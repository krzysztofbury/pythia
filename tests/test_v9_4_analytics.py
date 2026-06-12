"""Tests for v9.4 — per-asset stats, MAE/MFE, factor attribution."""

import datetime as dt

import pytest


def _seed_trade(journal, *, ticker, side, price, timestamp, profit=0.0, reasoning=""):
    """Insert a trade row directly (bypasses log_trade's now() timestamp)."""
    day_name = dt.datetime.fromisoformat(timestamp).strftime("%A")
    date = timestamp[:10]
    journal._conn.execute(  # noqa: SLF001
        "INSERT INTO trades "
        "(timestamp, date, day_of_week, account, instrument, side, "
        " price, quantity, value, currency, profit, tax, reasoning, "
        " fee, fill_source, exchange_fill_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            timestamp,
            date,
            day_name,
            "Kraken",
            ticker,
            side,
            price,
            1.0,
            price,
            "USDC",
            profit,
            0.0,
            reasoning,
            0.0,
            "bot",
            None,
        ),
    )
    journal._conn.commit()  # noqa: SLF001


def _seed_snapshot(journal, *, ticker, price, timestamp):
    journal._conn.execute(  # noqa: SLF001
        "INSERT INTO market_snapshots "
        "(timestamp, ticker, price, sentiment, consensus, fear_greed, "
        " polymarket_odds, indicators_json, regime) "
        "VALUES (?, ?, ?, '', '', '', '', '{}', 'RANGING')",
        (timestamp, ticker, price),
    )
    journal._conn.commit()  # noqa: SLF001


def _seed_signal(journal, *, ticker, action, composite, reasoning, timestamp):
    journal._conn.execute(  # noqa: SLF001
        "INSERT INTO signals "
        "(timestamp, ticker, action, confidence, composite, vol_regime, reasoning, llm_critique) "
        "VALUES (?, ?, ?, ?, ?, 'NORMAL', ?, '')",
        (timestamp, ticker, action, abs(composite), composite, reasoning),
    )
    journal._conn.commit()  # noqa: SLF001


# ---------------------------------------------------------------------------
# Per-asset metrics
# ---------------------------------------------------------------------------


class TestPerAssetMetrics:
    def test_empty_journal_returns_empty_list(self, tmp_path):
        from journal import TradeJournal
        from signal_analytics import SignalAnalyzer

        journal = TradeJournal(db_path=str(tmp_path / "empty.db"))
        stats = SignalAnalyzer(journal=journal).per_asset_metrics()
        assert stats == []

    def test_aggregates_sell_trades_per_ticker(self, tmp_path):
        from journal import TradeJournal
        from signal_analytics import SignalAnalyzer

        journal = TradeJournal(db_path=str(tmp_path / "j.db"))

        # Three BTC sells: +1, -0.5, +2 → total +2.5, win rate 2/3
        _seed_trade(
            journal,
            ticker="BTC/USDC",
            side="SELL",
            price=100.0,
            timestamp="2026-03-01T00:00:00+00:00",
            profit=1.0,
        )
        _seed_trade(
            journal,
            ticker="BTC/USDC",
            side="SELL",
            price=100.0,
            timestamp="2026-03-02T00:00:00+00:00",
            profit=-0.5,
        )
        _seed_trade(
            journal,
            ticker="BTC/USDC",
            side="SELL",
            price=100.0,
            timestamp="2026-03-03T00:00:00+00:00",
            profit=2.0,
        )
        # Two ETH sells: -1, +0.5 → total -0.5, win rate 1/2
        _seed_trade(
            journal,
            ticker="ETH/USDC",
            side="SELL",
            price=2000.0,
            timestamp="2026-03-04T00:00:00+00:00",
            profit=-1.0,
        )
        _seed_trade(
            journal,
            ticker="ETH/USDC",
            side="SELL",
            price=2000.0,
            timestamp="2026-03-05T00:00:00+00:00",
            profit=0.5,
        )
        # A BUY for a third ticker that should be ignored.
        _seed_trade(
            journal,
            ticker="LINK/USDC",
            side="BUY",
            price=8.0,
            timestamp="2026-03-06T00:00:00+00:00",
            profit=0.0,
        )

        stats = SignalAnalyzer(journal=journal).per_asset_metrics()
        assert len(stats) == 2
        by_ticker = {s.ticker: s for s in stats}
        btc = by_ticker["BTC/USDC"]
        assert btc.trades == 3
        assert btc.wins == 2
        assert btc.losses == 1
        assert btc.win_rate == pytest.approx(2 / 3)
        assert btc.total_pnl == pytest.approx(2.5)
        assert btc.best_trade == pytest.approx(2.0)
        assert btc.worst_trade == pytest.approx(-0.5)
        assert stats[0].ticker == "BTC/USDC"  # sorted by P&L desc

        eth = by_ticker["ETH/USDC"]
        assert eth.trades == 2
        assert eth.total_pnl == pytest.approx(-0.5)


# ---------------------------------------------------------------------------
# MAE / MFE
# ---------------------------------------------------------------------------


class TestMaeMfe:
    def test_mae_mfe_uses_snapshot_prices(self, tmp_path):
        from journal import TradeJournal
        from signal_analytics import SignalAnalyzer

        journal = TradeJournal(db_path=str(tmp_path / "j.db"))

        _seed_trade(
            journal,
            ticker="BTC/USDC",
            side="BUY",
            price=100.0,
            timestamp="2026-03-01T00:00:00+00:00",
        )
        _seed_snapshot(
            journal,
            ticker="BTC/USDC",
            price=95.0,
            timestamp="2026-03-01T06:00:00+00:00",
        )
        _seed_snapshot(
            journal,
            ticker="BTC/USDC",
            price=110.0,
            timestamp="2026-03-01T12:00:00+00:00",
        )
        _seed_snapshot(
            journal,
            ticker="BTC/USDC",
            price=103.0,
            timestamp="2026-03-01T18:00:00+00:00",
        )
        _seed_trade(
            journal,
            ticker="BTC/USDC",
            side="SELL",
            price=108.0,
            timestamp="2026-03-02T00:00:00+00:00",
            profit=8.0,
        )

        excursions = SignalAnalyzer(journal=journal).compute_mae_mfe()
        assert len(excursions) == 1
        exc = excursions[0]
        assert exc.entry_price == pytest.approx(100.0)
        assert exc.exit_price == pytest.approx(108.0)
        # Low during hold = 95 → MAE = (95-100)/100 = -5%
        assert exc.mae_pct == pytest.approx(-5.0)
        # High during hold = 110 → MFE = (110-100)/100 = +10%
        assert exc.mfe_pct == pytest.approx(10.0)

    def test_mae_mfe_without_snapshots_uses_entry_exit_only(self, tmp_path):
        from journal import TradeJournal
        from signal_analytics import SignalAnalyzer

        journal = TradeJournal(db_path=str(tmp_path / "j.db"))
        _seed_trade(
            journal,
            ticker="ETH/USDC",
            side="BUY",
            price=2000.0,
            timestamp="2026-03-01T00:00:00+00:00",
        )
        _seed_trade(
            journal,
            ticker="ETH/USDC",
            side="SELL",
            price=2050.0,
            timestamp="2026-03-02T00:00:00+00:00",
            profit=50.0,
        )

        excursions = SignalAnalyzer(journal=journal).compute_mae_mfe()
        assert len(excursions) == 1
        exc = excursions[0]
        # No snapshots → MAE is 0 (entry is min) and MFE is profit (exit is max).
        assert exc.mae_pct == pytest.approx(0.0)
        assert exc.mfe_pct == pytest.approx(2.5)

    def test_mae_mfe_filter_by_ticker(self, tmp_path):
        from journal import TradeJournal
        from signal_analytics import SignalAnalyzer

        journal = TradeJournal(db_path=str(tmp_path / "j.db"))
        _seed_trade(
            journal,
            ticker="BTC/USDC",
            side="BUY",
            price=100.0,
            timestamp="2026-03-01T00:00:00+00:00",
        )
        _seed_trade(
            journal,
            ticker="BTC/USDC",
            side="SELL",
            price=105.0,
            timestamp="2026-03-02T00:00:00+00:00",
            profit=5.0,
        )
        _seed_trade(
            journal,
            ticker="ETH/USDC",
            side="BUY",
            price=2000.0,
            timestamp="2026-03-01T00:00:00+00:00",
        )
        _seed_trade(
            journal,
            ticker="ETH/USDC",
            side="SELL",
            price=1900.0,
            timestamp="2026-03-02T00:00:00+00:00",
            profit=-100.0,
        )

        eth_only = SignalAnalyzer(journal=journal).compute_mae_mfe(ticker="ETH/USDC")
        assert len(eth_only) == 1
        assert eth_only[0].ticker == "ETH/USDC"


# ---------------------------------------------------------------------------
# Factor attribution
# ---------------------------------------------------------------------------


class TestFactorAttribution:
    def test_parses_components_from_reasoning(self):
        from signal_analytics import _parse_component_scores

        reasoning = (
            "Algorithmic BUY: composite=+0.301. tech=+0.200 mom=-0.050 fc=+0.000 "
            "sent=+0.333 cons=+0.200 poly=+0.000 fg=+0.160"
        )
        scores = _parse_component_scores(reasoning)
        assert scores["technical"] == pytest.approx(0.200)
        assert scores["momentum"] == pytest.approx(-0.050)
        assert scores["sentiment"] == pytest.approx(0.333)
        assert scores["consensus"] == pytest.approx(0.200)
        assert scores["polymarket"] == pytest.approx(0.000)
        assert scores["fear_greed"] == pytest.approx(0.160)

    def test_unparseable_reasoning_returns_empty_dict(self):
        from signal_analytics import _parse_component_scores

        assert _parse_component_scores("no scores in this text") == {}
        assert _parse_component_scores("") == {}

    def test_pearson_helper_basic(self):
        from signal_analytics import _pearson

        # Perfect positive correlation
        assert _pearson([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0)
        # Perfect negative correlation
        assert _pearson([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)
        # Zero variance in one series
        assert _pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) == 0.0

    def test_attribution_pairs_signals_with_sells(self, tmp_path):
        from journal import TradeJournal
        from signal_analytics import SignalAnalyzer

        journal = TradeJournal(db_path=str(tmp_path / "j.db"))

        # Scenario: sentiment positively predicts profit, consensus does not.
        cases = [
            ("2026-03-01", 0.8, 0.1, 5.0),  # high sent → profit
            ("2026-03-02", 0.7, 0.2, 3.0),  # high sent → profit
            ("2026-03-03", -0.5, 0.1, -4.0),  # low sent → loss
            ("2026-03-04", -0.6, 0.2, -2.0),  # low sent → loss
        ]
        for idx, (date, sent, cons, profit) in enumerate(cases, start=1):
            ts = f"{date}T00:00:00+00:00"
            reasoning = f"tech=+0.000 sent={sent:+.3f} cons={cons:+.3f} poly=+0.000"
            _seed_signal(
                journal,
                ticker="BTC/USDC",
                action="BUY",
                composite=sent,
                reasoning=reasoning,
                timestamp=ts,
            )
            sell_ts = f"{date}T12:00:00+00:00"
            _seed_trade(
                journal,
                ticker="BTC/USDC",
                side="SELL",
                price=100.0 + idx,
                timestamp=sell_ts,
                profit=profit,
            )

        factors = SignalAnalyzer(journal=journal).factor_attribution()
        by_component = {f.component: f for f in factors}
        # Sentiment strongly correlated with profit
        assert by_component["sentiment"].correlation_with_pnl > 0.9
        # Consensus essentially uncorrelated (small variance + noise)
        assert abs(by_component["consensus"].correlation_with_pnl) < 0.5
        # Technical is constant (0) → variance 0 → corr 0
        assert by_component["technical"].correlation_with_pnl == 0.0
        # Sample sizes consistent
        for f in factors:
            assert f.sample_size == 4


class TestReasoningIncludesComponentTokens:
    """Regression: real Strategist output must carry per-component tokens.

    C1 fix: the factor-attribution regex needs tokens like `tech=+0.025` in
    the persisted reasoning string; without them the feature returns [].
    """

    def test_generate_signal_emits_component_tokens(self):
        from models import MarketData
        from signal_analytics import _parse_component_scores
        from strategy import Strategist

        data = MarketData(
            ticker="BTC/USDC",
            current_price=100.0,
            sentiment="BULLISH (Score: 7/10)",
            consensus="UP, 60%",
            fear_greed="30 (Fear)",
            polymarket_odds="YES: 55%",
            has_position=False,
        )
        signal = Strategist().generate_signal(data)
        scores = _parse_component_scores(signal.reasoning)
        # All seven components should be parseable from reasoning.
        for key in (
            "technical",
            "momentum",
            "forecast",
            "sentiment",
            "consensus",
            "polymarket",
            "fear_greed",
        ):
            assert key in scores, f"Missing {key} in reasoning: {signal.reasoning}"


class TestSignalAnalyzerRun:
    def test_run_returns_all_three_analyses(self, tmp_path):
        from journal import TradeJournal
        from signal_analytics import SignalAnalyzer

        journal = TradeJournal(db_path=str(tmp_path / "j.db"))
        _seed_trade(
            journal,
            ticker="BTC/USDC",
            side="BUY",
            price=100.0,
            timestamp="2026-03-01T00:00:00+00:00",
        )
        _seed_trade(
            journal,
            ticker="BTC/USDC",
            side="SELL",
            price=110.0,
            timestamp="2026-03-02T00:00:00+00:00",
            profit=10.0,
        )

        summary = SignalAnalyzer(journal=journal).run()
        assert len(summary.per_asset) == 1
        assert len(summary.recent_excursions) == 1
        # No signals seeded → no factor attribution yet.
        assert summary.factors == []
