"""Tests for v9.7 — business-logic fixes + funding-rate composite wiring.

Covers:
- funding rate as an 8th composite component (TODO #1)
- consensus direction parsing with word boundaries (UPDATE != UP)
- "insufficient data" LLM responses treated as unavailable
- execute_sell reports success; position state survives failed sells
- fill reconciliation dedup window covers the 2h stale-order lifetime
- dust fill residues are not journaled as trades
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from journal import TradeJournal
from strategy import (
    Strategist,
    _parse_consensus,
    _parse_funding,
    _sentiment_available,
)


@pytest.fixture
def journal(tmp_path):
    return TradeJournal(db_path=str(tmp_path / "trades.db"))


def _make_data(**overrides):
    from models import MarketData

    defaults = {
        "ticker": "BTC/USDC",
        "current_price": 44000.0,
        "sentiment": "NEUTRAL (Score: 5/10): Mixed onchain signals.",
        "consensus": "Consensus unavailable.",
        "fear_greed": "20 (Extreme Fear)",
    }
    defaults.update(overrides)
    return MarketData(**defaults)


def _strong_technicals():
    from indicators import IndicatorSnapshot

    return IndicatorSnapshot(
        rsi=45.0,
        macd_histogram=0.12,
        bb_pct_b=0.50,
        atr_pct=2.5,
    )


# ---------------------------------------------------------------------------
# Funding rate — 8th composite component
# ---------------------------------------------------------------------------


class TestParseFunding:
    def test_missing_rate_has_no_data(self):
        score, has_data = _parse_funding(None)
        assert score == 0.0
        assert has_data is False

    def test_negative_funding_is_contrarian_bullish(self):
        score, has_data = _parse_funding(-0.0005)
        assert has_data is True
        assert score == pytest.approx(0.5)

    def test_positive_funding_is_bearish_and_clamped(self):
        score, has_data = _parse_funding(0.002)
        assert has_data is True
        assert score == -1.0

    def test_zero_rate_is_neutral_but_present(self):
        score, has_data = _parse_funding(0.0)
        assert score == 0.0
        assert has_data is True


class TestFundingInComposite:
    def test_default_weights_include_funding_and_sum_to_one(self):
        weights = Strategist.DEFAULT_WEIGHTS
        assert "funding" in weights
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_negative_funding_lifts_composite(self):
        strategist = Strategist()
        base = _make_data(indicators=_strong_technicals())
        funded = _make_data(indicators=_strong_technicals(), funding_rate=-0.001)
        sig_base = strategist.generate_signal(base)
        sig_funded = strategist.generate_signal(funded)
        assert sig_funded.composite > sig_base.composite

    def test_missing_funding_keeps_prior_behaviour(self):
        strategist = Strategist()
        data = _make_data(indicators=_strong_technicals())
        signal = strategist.generate_signal(data)
        assert "fund=" not in signal.reasoning

    def test_funding_token_in_reasoning_when_present(self):
        strategist = Strategist()
        data = _make_data(indicators=_strong_technicals(), funding_rate=-0.001)
        signal = strategist.generate_signal(data)
        assert "fund=+1.000" in signal.reasoning


# ---------------------------------------------------------------------------
# Consensus parsing — word boundaries
# ---------------------------------------------------------------------------


class TestParseConsensusWordBoundaries:
    def test_update_is_not_up(self):
        direction, _prob, score = _parse_consensus("UPDATE: mixed outlook, 70% uncertain.")
        assert direction == "NEUTRAL"
        assert score == 0.0

    def test_downtrend_is_not_down(self):
        direction, _prob, _score = _parse_consensus("DOWNTREND warning at 70% odds.")
        assert direction == "NEUTRAL"

    def test_plain_up_still_parses(self):
        direction, prob, score = _parse_consensus("UP, 68%, consensus bullish.")
        assert direction == "UP"
        assert prob == 68.0
        assert score > 0.0

    def test_plain_down_still_parses(self):
        direction, prob, score = _parse_consensus("DOWN, 70%, risk-off.")
        assert direction == "DOWN"
        assert prob == 70.0
        assert score < 0.0


# ---------------------------------------------------------------------------
# "Insufficient data" responses are unavailable, not neutral opinions
# ---------------------------------------------------------------------------


class TestInsufficientDataUnavailable:
    @pytest.mark.parametrize(
        "text",
        [
            "NEUTRAL (Score: 5/10): Insufficient recent data available.",
            "Insufficient real-time data available to form a consensus.",
        ],
    )
    def test_insufficient_marks_unavailable(self, text):
        assert _sentiment_available(text) is False


# ---------------------------------------------------------------------------
# execute_sell reports success — failed sells keep position state
# ---------------------------------------------------------------------------


def _sell_settings(mode: str = "LIVE"):
    settings = MagicMock()
    settings.trading_mode = mode
    settings.quote_currency = "USDC"
    return settings


class TestExecuteSellReturnsSuccess:
    def test_successful_live_sell_returns_true(self, journal):
        from order_executor import execute_sell

        exchange = MagicMock()
        exchange.get_free_balance.return_value = 0.5
        result = execute_sell(
            "ETH/USDC",
            price=2000.0,
            buy_price=1900.0,
            reason="test exit",
            settings=_sell_settings(),
            exchange=exchange,
            journal=journal,
            notifier=MagicMock(),
        )
        assert result is True
        exchange.place_maker_sell.assert_called_once()

    def test_no_balance_returns_false(self, journal):
        from order_executor import execute_sell

        exchange = MagicMock()
        exchange.get_free_balance.return_value = 0.0
        result = execute_sell(
            "ETH/USDC",
            price=2000.0,
            buy_price=1900.0,
            reason="test exit",
            settings=_sell_settings(),
            exchange=exchange,
            journal=journal,
            notifier=MagicMock(),
        )
        assert result is False
        exchange.place_maker_sell.assert_not_called()

    def test_failed_order_returns_false_and_journals_nothing(self, journal):
        from order_executor import execute_sell

        exchange = MagicMock()
        exchange.get_free_balance.return_value = 0.5
        exchange.place_maker_sell.side_effect = RuntimeError("EOrder:Insufficient funds")
        result = execute_sell(
            "ETH/USDC",
            price=2000.0,
            buy_price=1900.0,
            reason="test exit",
            settings=_sell_settings(),
            exchange=exchange,
            journal=journal,
            notifier=MagicMock(),
        )
        assert result is False
        row = journal._conn.execute(  # noqa: SLF001
            "SELECT COUNT(*) FROM trades WHERE side='SELL'",
        ).fetchone()
        assert row[0] == 0


class TestTakeProfitKeepsStateOnFailedSell:
    def test_position_state_survives_failed_sell(self, journal):
        from models import PositionState, StopMode
        from order_executor import check_take_profit

        journal.upsert_position_state(
            PositionState(
                ticker="ETH/USDC",
                entry_price=1900.0,
                high_watermark=2100.0,
                stop_mode=StopMode.FIXED,
                current_sl=1800.0,
                current_tp=1950.0,
                regime_at_entry="RANGING",
            ),
        )
        settings = _sell_settings()
        settings.use_atr_stops = False
        settings.stop_loss_pct = 0.02
        settings.take_profit_pct = 0.04
        settings.dynamic_risk_enabled = False

        exchange = MagicMock()
        exchange.get_free_balance.return_value = 0.5
        exchange.place_maker_sell.side_effect = RuntimeError("down")

        data = _make_data(
            ticker="ETH/USDC",
            current_price=2100.0,
            entry_price=1900.0,
        )
        sold = check_take_profit(
            "ETH/USDC",
            data,
            settings,
            exchange,
            journal,
            MagicMock(),
            None,
        )
        assert sold is False
        assert journal.get_position_state("ETH/USDC") is not None


# ---------------------------------------------------------------------------
# Fill reconciliation — dedup window + dust residues
# ---------------------------------------------------------------------------


def _fill(fill_id: str, side: str, price: float, qty: float) -> dict:
    return {
        "id": fill_id,
        "side": side,
        "price": price,
        "amount": qty,
        "cost": price * qty,
        "fee": {"cost": price * qty * 0.0016},
    }


class TestFillDedupWindow:
    def test_fill_after_15_minutes_still_deduped(self, journal):
        """Maker orders fill up to 2h after placement; 15min window missed them."""
        from order_executor import _process_single_fill

        journal.log_trade(
            account="Kraken",
            instrument="LINK/USDC",
            side="BUY",
            price=9.496,
            quantity=2.0153,
            value=19.14,
            currency="USDC",
            reasoning="placement",
            timestamp=datetime.now(UTC) - timedelta(minutes=40),
        )
        _process_single_fill(
            _fill("FILL-1", "buy", 9.496, 2.01536942),
            "LINK/USDC",
            journal,
            MagicMock(),
            "USDC",
            0.19,
        )
        count = journal._conn.execute(  # noqa: SLF001
            "SELECT COUNT(*) FROM trades WHERE side='BUY'",
        ).fetchone()[0]
        assert count == 1  # placement only — fill recognized as duplicate

    def test_dust_residue_fill_not_journaled(self, journal):
        from order_executor import _process_single_fill

        _process_single_fill(
            _fill("FILL-2", "buy", 9.496, 0.00006942),  # $0.00066 residue
            "LINK/USDC",
            journal,
            MagicMock(),
            "USDC",
            0.19,
        )
        count = journal._conn.execute(  # noqa: SLF001
            "SELECT COUNT(*) FROM trades",
        ).fetchone()[0]
        assert count == 0

    def test_genuine_late_fill_still_recorded(self, journal):
        """A fill with no matching placement (e.g. exchange SL) is recorded."""
        from order_executor import _process_single_fill

        _process_single_fill(
            _fill("FILL-3", "sell", 9.80, 2.0153),
            "LINK/USDC",
            journal,
            MagicMock(),
            "USDC",
            0.19,
        )
        count = journal._conn.execute(  # noqa: SLF001
            "SELECT COUNT(*) FROM trades WHERE side='SELL'",
        ).fetchone()[0]
        assert count == 1
