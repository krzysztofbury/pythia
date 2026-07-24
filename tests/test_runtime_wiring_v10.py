"""Runtime-level v10 wiring tests with real execution primitives."""
# ruff: noqa: D102, D103

from __future__ import annotations

import os
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

from analytics import PortfolioAnalyzer
from config import Settings
from execution_coordinator import ExecutionCoordinator
from execution_store import ExecutionStore
from journal import TradeJournal
from llm_challenger import build_challenger_from_settings
from main import _apply_quantitative_context, run_cycle
from models import FullCycleContext, MarketData, TradeSignal
from paper_exchange import MarketRules, PaperExchange
from risk_guardian import RiskGuardian
from runtime_state import RuntimeState
from sentiment import SentimentOracle


class MarketExchange:
    """Network-free market-data venue used by the runtime orchestration test."""

    def __init__(self) -> None:
        """Initialize a narrow synthetic spread."""
        self.bid = 99.8
        self.ask = 100.2
        self.last = 100.0

    def symbol_exists(self, symbol: str) -> bool:
        return symbol == "BTC/USDC"

    def get_market_structure(self, _symbol: str) -> dict[str, float]:
        return {
            "bid": self.bid,
            "ask": self.ask,
            "spread_pct": (self.ask - self.bid) / self.bid,
        }

    def fetch_price(self, _symbol: str) -> float:
        return self.last


def _settings(tmp_path: Path) -> Settings:
    values = {
        "kraken_api_key": "test",
        "kraken_private_key": "test",
        "trading_universe": "BTC/USDC",
        "paper_state_db_path": str(tmp_path / "execution.db"),
        "live_state_db_path": str(tmp_path / "live-execution.db"),
        "analytics_db_path": str(tmp_path / "analytics.db"),
        "llm_budget_db_path": str(tmp_path / "llm.db"),
        "paper_exchange_state_path": str(tmp_path / "paper.json"),
        "singleton_lock_path": str(tmp_path / "runtime.lock"),
        "healthcheck_file": str(tmp_path / "health.json"),
        "garch_enabled": False,
        "arima_enabled": False,
        "mtf_enabled": False,
        "polymarket_enabled": False,
        "funding_rates_enabled": False,
        "max_total_crypto_positions": 1,
        "max_correlated_positions": 1,
        "max_entries_per_cycle": 1,
        "use_atr_stops": False,
    }
    with patch.dict(os.environ, {}, clear=True):
        return Settings(_env_file=None, **values)


def _paper() -> PaperExchange:
    return PaperExchange(
        {"USDC": Decimal(1000), "BTC": Decimal(0)},
        {
            "BTC/USDC": MarketRules(
                "BTC",
                "USDC",
                price_increment="0.00000000000000000001",
                amount_increment="0.00000000000000000001",
                min_notional="10",
                min_amount="0.00000000000000000001",
            ),
        },
        start_timestamp=1,
    )


def _market_data(price: float) -> dict[str, MarketData]:
    return {"BTC/USDC": MarketData(ticker="BTC/USDC", current_price=price)}


def test_mocked_run_cycle_recovers_before_submitting_without_placement_fill(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    market = MarketExchange()
    paper = _paper()
    store = ExecutionStore(settings.state_db_path, "paper:kraken:test")
    journal = TradeJournal(settings.analytics_db_path)
    runtime = RuntimeState()
    coordinator = ExecutionCoordinator(store, paper, live_entries_enabled=True)
    guardian = RiskGuardian(settings, paper, market_exchange=market, execution_store=store)
    strategist = MagicMock()
    strategist.generate_signal.return_value = TradeSignal(
        action="BUY",
        confidence=0.8,
        composite=0.8,
        reasoning="deterministic test entry",
    )
    oracle = MagicMock()
    oracle.get_fear_and_greed_index.return_value = "50 (Neutral)"
    notifier = MagicMock()

    with patch("main.collect_all_market_data", return_value=_market_data(market.last)):
        run_cycle(
            settings,
            market,  # type: ignore[arg-type]
            oracle,
            strategist,
            guardian,
            journal,
            notifier,
            coordinator=coordinator,
            execution_store=store,
            runtime=runtime,
            execution_venue=paper,
            full_cycle_context=FullCycleContext(),
            cycle_id="cycle-one",
            force_full_cycle=True,
        )

    orders = store.list_orders()
    assert len(orders) == 1
    assert orders[0].intent_id == "cycle-one:entry:BTC/USDC"
    assert store.get_position("BTC/USDC") is None
    assert journal.get_trade_count("BTC/USDC") == 0
    notifier.notify_trade.assert_not_called()
    assert paper.get_free_balance("USDC") < paper.get_total_balance("USDC"), (
        paper.get_closed_orders(),
        paper.get_open_orders(),
        orders,
        store.list_reconciliation_issues(),
    )
    store.close()
    journal.close()


def test_paper_two_cycles_place_then_fill_reconcile_and_notify(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    market = MarketExchange()
    paper = _paper()
    store = ExecutionStore(settings.state_db_path, "paper:kraken:test")
    journal = TradeJournal(settings.analytics_db_path)
    runtime = RuntimeState()
    coordinator = ExecutionCoordinator(store, paper, live_entries_enabled=True)
    guardian = RiskGuardian(settings, paper, market_exchange=market, execution_store=store)
    strategist = MagicMock()
    strategist.generate_signal.return_value = TradeSignal(
        action="BUY",
        confidence=0.8,
        composite=0.8,
        reasoning="deterministic test entry",
    )
    oracle = MagicMock()
    oracle.get_fear_and_greed_index.return_value = "50 (Neutral)"
    notifier = MagicMock()
    context = FullCycleContext()

    with patch("main.collect_all_market_data", return_value=_market_data(market.last)):
        run_cycle(
            settings,
            market,  # type: ignore[arg-type]
            oracle,
            strategist,
            guardian,
            journal,
            notifier,
            coordinator=coordinator,
            execution_store=store,
            runtime=runtime,
            execution_venue=paper,
            portfolio_analyzer=PortfolioAnalyzer(journal),
            full_cycle_context=context,
            cycle_id="cycle-one",
            force_full_cycle=True,
            paper_state_path=settings.paper_exchange_state_path,
        )
    assert journal.get_trade_count("BTC/USDC") == 0

    market.bid = 99.6
    market.ask = 99.8
    market.last = 99.7
    with patch("main.collect_all_market_data", return_value=_market_data(market.last)):
        run_cycle(
            settings,
            market,  # type: ignore[arg-type]
            oracle,
            strategist,
            guardian,
            journal,
            notifier,
            coordinator=coordinator,
            execution_store=store,
            runtime=runtime,
            execution_venue=paper,
            full_cycle_context=context,
            cycle_id="cycle-two",
            force_full_cycle=True,
            paper_state_path=settings.paper_exchange_state_path,
        )

    position = store.get_position("BTC/USDC")
    assert position is not None
    assert position.quantity > 0
    assert journal.get_trade_count("BTC/USDC") == 1
    notifier.notify_trade.assert_called_once()
    assert coordinator.ready is True, store.list_reconciliation_issues()
    assert Path(settings.paper_exchange_state_path).exists()
    store.close()
    journal.close()


def test_disabled_sentiment_and_challenger_make_no_paid_calls(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.challenger_enabled = True
    oracle = SentimentOracle(None, "gemini/test", enabled=False)

    with (
        patch("litellm.completion") as completion,
        patch("sentiment.requests.get") as http_get,
    ):
        assert oracle.analyze_ticker_sentiment("BTC/USDC").startswith("NEUTRAL")
        assert oracle.get_consensus_odds("BTC/USDC").startswith("Consensus unavailable")
        assert oracle.get_fear_and_greed_index().endswith("(sentiment disabled)")
        challenger = build_challenger_from_settings(settings)

    assert challenger.enabled is False
    completion.assert_not_called()
    http_get.assert_not_called()


def test_fast_cycle_reuses_only_fresh_full_quantitative_context(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.garch_enabled = True
    settings.arima_enabled = True
    volatility = MagicMock()
    forecast = MagicMock()
    full_data = MarketData(
        ticker="BTC/USDC",
        current_price=100.0,
        volatility=volatility,
        forecast=forecast,
    )
    context = FullCycleContext()
    _apply_quantitative_context(
        settings,
        {"BTC/USDC": full_data},
        context,
        100.0,
        is_full_cycle=True,
    )

    fast_data = MarketData(ticker="BTC/USDC", current_price=101.0)
    _apply_quantitative_context(
        settings,
        {"BTC/USDC": fast_data},
        context,
        200.0,
        is_full_cycle=False,
    )
    assert fast_data.quantitative_context_fresh is True
    assert fast_data.volatility is volatility
    assert fast_data.forecast is forecast

    stale_data = MarketData(ticker="BTC/USDC", current_price=102.0)
    _apply_quantitative_context(
        settings,
        {"BTC/USDC": stale_data},
        context,
        100.0 + settings.full_cycle_seconds + 1,
        is_full_cycle=False,
    )
    assert stale_data.quantitative_context_fresh is False
    assert stale_data.volatility is None
    assert stale_data.forecast is None

    failed_full_data = MarketData(ticker="BTC/USDC", current_price=103.0)
    _apply_quantitative_context(
        settings,
        {"BTC/USDC": failed_full_data},
        context,
        5_000.0,
        is_full_cycle=True,
    )
    following_fast_data = MarketData(ticker="BTC/USDC", current_price=104.0)
    _apply_quantitative_context(
        settings,
        {"BTC/USDC": following_fast_data},
        context,
        5_100.0,
        is_full_cycle=False,
    )
    assert following_fast_data.quantitative_context_fresh is False
