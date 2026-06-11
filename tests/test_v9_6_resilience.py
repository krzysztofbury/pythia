"""Tests for v9.6 — LLM resilience + entry-pipeline fixes.

Covers the 2026-06-11 incident fixes:
- sentiment is advisory, not a hard entry veto (block only BEARISH by default)
- NEUTRAL / unavailable sentiment no longer drags the composite negative
- sentiment+consensus LLM responses cached per ticker (credit burn)
- guardian sizes sub-tier-1 composites instead of silently killing them
- min-order round-up generalized for small accounts
- entry-drought alert so 7 silent weeks can never happen again
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from strategy import Strategist, _parse_sentiment, _sentiment_available

_ENV = {
    "KRAKEN_API_KEY": "test",
    "KRAKEN_PRIVATE_KEY": "test",
}


def _make_settings(**overrides):
    from config import Settings

    env = {**_ENV, **{k.upper(): str(v) for k, v in overrides.items()}}
    with patch.dict("os.environ", env, clear=False):
        s = Settings()
    for k, v in overrides.items():
        if hasattr(s, k):
            setattr(s, k, v)
    return s


def _make_buy_signal(composite: float):
    from models import TradeSignal

    return TradeSignal(
        action="BUY",
        confidence=abs(composite),
        composite=composite,
        reasoning="test",
    )


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
# _parse_sentiment — NEUTRAL must not drag the composite
# ---------------------------------------------------------------------------


class TestParseSentimentNeutralZero:
    @pytest.mark.parametrize("raw_score", [4, 5, 6])
    def test_neutral_band_scores_zero(self, raw_score):
        direction, score = _parse_sentiment(
            f"NEUTRAL (Score: {raw_score}/10): Mixed signals.",
        )
        assert direction == "NEUTRAL"
        assert score == 0.0

    def test_bullish_score_still_positive(self):
        direction, score = _parse_sentiment("BULLISH (Score: 8/10): inflows.")
        assert direction == "BULLISH"
        assert score > 0.0

    def test_bearish_score_still_negative(self):
        direction, score = _parse_sentiment("BEARISH (Score: 2/10): outflows.")
        assert direction == "BEARISH"
        assert score < 0.0


# ---------------------------------------------------------------------------
# Sentiment availability — failure fallbacks excluded from effective weight
# ---------------------------------------------------------------------------


class TestSentimentAvailability:
    @pytest.mark.parametrize(
        "text",
        [
            "NEUTRAL (Score: 5/10): Sentiment oracle temporarily unavailable.",
            "NEUTRAL (Score: 5/10): Analysis failed.",
            "NEUTRAL (Score: 5/10): Timeout.",
            "Consensus unavailable.",
            "",
        ],
    )
    def test_failure_fallbacks_are_unavailable(self, text):
        assert _sentiment_available(text) is False

    @pytest.mark.parametrize(
        "text",
        [
            "NEUTRAL (Score: 5/10): Mixed onchain signals.",
            "BULLISH (Score: 8/10): strong inflows.",
            "UP, 68%, Twitter consensus bullish.",
        ],
    )
    def test_genuine_responses_are_available(self, text):
        assert _sentiment_available(text) is True

    def test_unavailable_sentiment_not_diluting_composite(self):
        """Dead sentiment weight redistributes: composite magnitude grows."""
        strategist = Strategist()
        strategist.entry_threshold = 0.05

        genuine = _make_data(
            sentiment="NEUTRAL (Score: 5/10): Mixed onchain signals.",
            indicators=_strong_technicals(),
        )
        unavailable = _make_data(
            sentiment="NEUTRAL (Score: 5/10): Analysis failed.",
            indicators=_strong_technicals(),
        )
        sig_genuine = strategist.generate_signal(genuine)
        sig_unavailable = strategist.generate_signal(unavailable)
        assert abs(sig_unavailable.composite) > abs(sig_genuine.composite)


# ---------------------------------------------------------------------------
# Entry gate — sentiment is advisory; only BEARISH blocks by default
# ---------------------------------------------------------------------------


class TestEntryGateSentimentAdvisory:
    def setup_method(self):
        self.strategist = Strategist()
        # Decouple from exact composite scoring — gate behavior is the subject.
        self.strategist.entry_threshold = 0.05

    def test_neutral_sentiment_allows_buy(self):
        data = _make_data(indicators=_strong_technicals())
        signal = self.strategist.generate_signal(data)
        assert signal.action == "BUY"

    def test_bearish_sentiment_blocks_buy(self):
        data = _make_data(
            sentiment="BEARISH (Score: 2/10): heavy outflows.",
            indicators=_strong_technicals(),
        )
        signal = self.strategist.generate_signal(data)
        assert signal.action == "HOLD"
        assert "sentiment" in signal.reasoning.lower()

    def test_bearish_consensus_blocks_buy(self):
        data = _make_data(
            consensus="DOWN, 70%, broad risk-off consensus.",
            indicators=_strong_technicals(),
        )
        signal = self.strategist.generate_signal(data)
        assert signal.action == "HOLD"

    def test_legacy_mode_requires_bullish(self):
        self.strategist.require_bullish_sentiment = True
        data = _make_data(indicators=_strong_technicals())
        signal = self.strategist.generate_signal(data)
        assert signal.action == "HOLD"

    def test_legacy_mode_bullish_passes(self):
        self.strategist.require_bullish_sentiment = True
        data = _make_data(
            sentiment="BULLISH (Score: 8/10): strong inflows.",
            indicators=_strong_technicals(),
        )
        signal = self.strategist.generate_signal(data)
        assert signal.action == "BUY"


# ---------------------------------------------------------------------------
# Guardian — sub-tier-1 composites sized at tier-1, not silently killed
# ---------------------------------------------------------------------------


class TestGuardianSubTierSizing:
    def test_sub_tier_entry_sized_at_tier_1(self):
        """Strategy approved the BUY (regime-adjusted threshold) — size it."""
        from risk_guardian import RiskGuardian

        settings = _make_settings()
        settings.max_position_size_usd = 50.0
        exchange = MagicMock()
        exchange.get_free_balance.return_value = 1000.0
        guardian = RiskGuardian(settings, exchange)
        amount = guardian.check_buy(
            "BTC/USDC",
            price=50_000.0,
            trade_signal=_make_buy_signal(0.20),
            available_slots=1,
        )
        # base 1000 x 0.6 = 600 → concentration 400 → hard cap 50 binds
        assert amount == pytest.approx(50.0 / 50_000.0, rel=1e-6)

    def test_sub_tier_dca_still_rejected(self):
        from risk_guardian import RiskGuardian

        settings = _make_settings()
        exchange = MagicMock()
        exchange.get_free_balance.return_value = 1000.0
        guardian = RiskGuardian(settings, exchange)
        amount = guardian.check_buy(
            "BTC/USDC",
            price=50_000.0,
            trade_signal=_make_buy_signal(0.20),
            available_slots=1,
            is_dca=True,
        )
        assert amount == 0.0


class TestGuardianMinOrderBump:
    def test_small_account_tier_1_bumped_to_min_order(self):
        """Balance $80 / 7 slots x 0.6 = $6.86 → bump to $10 (Kraken min)."""
        from risk_guardian import RiskGuardian

        settings = _make_settings()
        exchange = MagicMock()
        exchange.get_free_balance.return_value = 80.0
        guardian = RiskGuardian(settings, exchange)
        amount = guardian.check_buy(
            "BTC/USDC",
            price=50_000.0,
            trade_signal=_make_buy_signal(0.30),
            available_slots=7,
        )
        assert amount == pytest.approx(10.0 / 50_000.0, rel=1e-6)

    def test_dust_allocation_still_rejected(self):
        """Far-below-minimum target is not bumped (5x the intended risk)."""
        from risk_guardian import RiskGuardian

        settings = _make_settings()
        exchange = MagicMock()
        exchange.get_free_balance.return_value = 50.0
        guardian = RiskGuardian(settings, exchange)
        amount = guardian.check_buy(
            "BTC/USDC",
            price=50_000.0,
            trade_signal=_make_buy_signal(0.30),
            available_slots=7,
        )
        # 50/7 x 0.6 = $4.29 < $5 round-up floor → reject
        assert amount == 0.0

    def test_bump_respects_budget(self):
        from risk_guardian import RiskGuardian

        settings = _make_settings()
        exchange = MagicMock()
        exchange.get_free_balance.return_value = 80.0
        guardian = RiskGuardian(settings, exchange)
        amount = guardian.check_buy(
            "BTC/USDC",
            price=50_000.0,
            trade_signal=_make_buy_signal(0.30),
            available_slots=7,
            budget_usd=5.0,
        )
        assert amount == 0.0


# ---------------------------------------------------------------------------
# SentimentOracle — per-ticker TTL cache + provider-agnostic keys
# ---------------------------------------------------------------------------


def _fake_litellm_response(text: str):
    msg = MagicMock()
    msg.content = text
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


class TestSentimentCache:
    def _oracle(self, ttl: float = 3600.0):
        from sentiment import SentimentOracle

        return SentimentOracle(
            xai_key=None,
            model_name="gemini/gemini-3.5-flash",
            extra_api_keys={"gemini": "fake"},
            cache_ttl_seconds=ttl,
        )

    def test_sentiment_cached_within_ttl(self):
        oracle = self._oracle()
        with patch("litellm.completion") as mock_completion:
            mock_completion.return_value = _fake_litellm_response(
                "BULLISH (Score: 8/10): inflows",
            )
            first = oracle.analyze_ticker_sentiment("BTC/USDC")
            second = oracle.analyze_ticker_sentiment("BTC/USDC")
        assert first == second
        assert mock_completion.call_count == 1

    def test_cache_is_per_ticker(self):
        oracle = self._oracle()
        with patch("litellm.completion") as mock_completion:
            mock_completion.return_value = _fake_litellm_response(
                "BULLISH (Score: 8/10): inflows",
            )
            oracle.analyze_ticker_sentiment("BTC/USDC")
            oracle.analyze_ticker_sentiment("ETH/USDC")
        assert mock_completion.call_count == 2

    def test_failures_not_cached(self):
        oracle = self._oracle()
        with patch("litellm.completion", side_effect=RuntimeError("down")):
            out = oracle.analyze_ticker_sentiment("BTC/USDC")
        assert out.startswith("NEUTRAL")
        with patch("litellm.completion") as mock_completion:
            mock_completion.return_value = _fake_litellm_response(
                "BULLISH (Score: 8/10): recovered",
            )
            out = oracle.analyze_ticker_sentiment("BTC/USDC")
        assert out.startswith("BULLISH")
        assert mock_completion.call_count == 1

    def test_consensus_cached_within_ttl(self):
        oracle = self._oracle()
        with patch("litellm.completion") as mock_completion:
            mock_completion.return_value = _fake_litellm_response(
                "UP, 65%, consensus bullish",
            )
            oracle.get_consensus_odds("BTC/USDC")
            oracle.get_consensus_odds("BTC/USDC")
        assert mock_completion.call_count == 1

    def test_zero_ttl_disables_cache(self):
        oracle = self._oracle(ttl=0.0)
        with patch("litellm.completion") as mock_completion:
            mock_completion.return_value = _fake_litellm_response(
                "BULLISH (Score: 8/10): inflows",
            )
            oracle.analyze_ticker_sentiment("BTC/USDC")
            oracle.analyze_ticker_sentiment("BTC/USDC")
        assert mock_completion.call_count == 2

    def test_oracle_without_xai_key(self):
        oracle = self._oracle()
        assert oracle._llm._primary == "gemini/gemini-3.5-flash"  # noqa: SLF001


# ---------------------------------------------------------------------------
# Config — Gemini primary, xAI optional, retention bumped
# ---------------------------------------------------------------------------


class TestConfigDefaults:
    def test_sentiment_model_defaults_to_gemini(self):
        settings = _make_settings()
        assert settings.sentiment_model == "gemini/gemini-3.5-flash"

    def test_xai_key_now_optional(self):
        settings = _make_settings()
        assert settings.kraken_api_key is not None

    def test_gate_mode_defaults_to_advisory(self):
        settings = _make_settings()
        assert settings.require_bullish_sentiment is False

    def test_log_retention_default_30_days(self):
        settings = _make_settings()
        assert settings.log_retention_days == 30

    def test_sentiment_cache_default_one_hour(self):
        settings = _make_settings()
        assert settings.sentiment_cache_seconds == 3600

    def test_entry_drought_alert_default_72h(self):
        settings = _make_settings()
        assert settings.no_entry_alert_hours == 72.0


# ---------------------------------------------------------------------------
# Slot computation — capital divided by real max positions, not universe size
# ---------------------------------------------------------------------------


class TestSlotComputation:
    def test_slots_capped_by_max_total_positions(self):
        from portfolio_manager import compute_available_slots

        assert compute_available_slots(n_available=7, n_owned=0, max_total=5) == 5

    def test_slots_reduced_by_owned(self):
        from portfolio_manager import compute_available_slots

        assert compute_available_slots(n_available=5, n_owned=3, max_total=5) == 2

    def test_slots_never_below_one(self):
        from portfolio_manager import compute_available_slots

        assert compute_available_slots(n_available=0, n_owned=5, max_total=5) == 1

    def test_fewer_available_than_cap(self):
        from portfolio_manager import compute_available_slots

        assert compute_available_slots(n_available=2, n_owned=0, max_total=5) == 2


# ---------------------------------------------------------------------------
# Entry-drought alert — silence is a failure mode
# ---------------------------------------------------------------------------


class TestEntryDroughtAlert:
    def _journal(self, tmp_path):
        from journal import TradeJournal

        return TradeJournal(db_path=str(tmp_path / "trades.db"))

    def _log_buy(self, journal, *, days_ago: float):
        ts = datetime.now(UTC) - timedelta(days=days_ago)
        journal.log_trade(
            account="Kraken",
            instrument="BTC/USDC",
            side="BUY",
            price=50_000.0,
            quantity=0.0002,
            value=10.0,
            currency="USDC",
            reasoning="test",
            timestamp=ts,
        )

    def test_alert_fires_when_no_recent_entry(self, tmp_path):
        from main import check_entry_drought

        journal = self._journal(tmp_path)
        self._log_buy(journal, days_ago=10.0)
        settings = _make_settings(trading_mode="LIVE")
        notifier = MagicMock()
        fired = check_entry_drought(journal, settings, notifier, _state={})
        assert fired is True
        notifier.notify_error.assert_called_once()

    def test_no_alert_when_recent_entry(self, tmp_path):
        from main import check_entry_drought

        journal = self._journal(tmp_path)
        self._log_buy(journal, days_ago=0.5)
        settings = _make_settings(trading_mode="LIVE")
        notifier = MagicMock()
        fired = check_entry_drought(journal, settings, notifier, _state={})
        assert fired is False
        notifier.notify_error.assert_not_called()

    def test_no_alert_in_paper_mode(self, tmp_path):
        from main import check_entry_drought

        journal = self._journal(tmp_path)
        self._log_buy(journal, days_ago=10.0)
        settings = _make_settings(trading_mode="PAPER")
        notifier = MagicMock()
        fired = check_entry_drought(journal, settings, notifier, _state={})
        assert fired is False

    def test_alert_throttled_to_once_per_day(self, tmp_path):
        from main import check_entry_drought

        journal = self._journal(tmp_path)
        self._log_buy(journal, days_ago=10.0)
        settings = _make_settings(trading_mode="LIVE")
        notifier = MagicMock()
        state: dict = {}
        assert check_entry_drought(journal, settings, notifier, _state=state) is True
        assert check_entry_drought(journal, settings, notifier, _state=state) is False
        assert notifier.notify_error.call_count == 1

    def test_alert_disabled_when_zero(self, tmp_path):
        from main import check_entry_drought

        journal = self._journal(tmp_path)
        self._log_buy(journal, days_ago=10.0)
        settings = _make_settings(trading_mode="LIVE")
        settings.no_entry_alert_hours = 0.0
        notifier = MagicMock()
        assert check_entry_drought(journal, settings, notifier, _state={}) is False
