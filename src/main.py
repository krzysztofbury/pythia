"""Oracle Pythia v10 boot, exact recovery, and cycle orchestration."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import logging.handlers
import os
import signal
import sys
import time
import uuid
from decimal import Decimal
from pathlib import Path

from analytics import PortfolioAnalyzer
from cache import CachedExchange
from config import Settings
from data_pipeline import collect_all_market_data
from decision_engine import evaluate_entry, evaluate_exit
from exchange import KrakenExchange
from execution_coordinator import ExecutionCoordinator, RecoverySummary
from execution_store import ExecutionStore, PositionSnapshot
from forecast import PriceForecaster
from health import HealthReporter, HealthState
from indicators import TechnicalAnalyzer
from journal import TradeJournal
from llm_budget import LLMBudgetManager
from llm_challenger import LLMChallenger, build_challenger_from_settings
from models import FullCycleContext, MarketData
from notifier import NullNotifier, WebhookNotifier
from order_executor import ensure_stop_loss
from paper_exchange import MarketRules, PaperExchange
from portfolio_manager import classify_tickers, compute_available_slots, reserved_entry_tickers
from redaction import RedactingFilter, RedactingFormatter, Redactor
from risk_guardian import RiskGuardian, check_daily_loss_limit
from runtime_state import MutationRejectedError, RuntimeState
from sentiment import SentimentOracle
from singleton import ProcessSingleton
from strategy import Strategist
from volatility import VolatilityAnalyzer

logger = logging.getLogger("OraclePythia")

APP_VERSION = "10.0.0"
_DROUGHT_ALERT_THROTTLE_SECONDS = 24 * 3600.0
_drought_alert_state: dict[str, float] = {}


def setup_logging(settings: Settings) -> Redactor:
    """Install one central redacting console/file logging pipeline."""
    redactor = Redactor(_configured_secrets(settings))
    formatter = RedactingFormatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        redactor=redactor,
    )
    redacting_filter = RedactingFilter(redactor=redactor)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    console.addFilter(redacting_filter)
    root.addHandler(console)

    log_path = Path("logs/bot.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.TimedRotatingFileHandler(
        log_path,
        when="midnight",
        backupCount=settings.log_retention_days,
        encoding="utf-8",
        utc=True,
    )
    file_handler.setFormatter(formatter)
    file_handler.addFilter(redacting_filter)
    root.addHandler(file_handler)
    return redactor


def _configured_secrets(settings: Settings) -> list[object]:
    values: list[object] = []
    for name in settings.__class__.model_fields:
        value = getattr(settings, name)
        if value is not None and any(
            marker in name for marker in ("key", "secret", "ack", "account_id", "url", "proxy")
        ):
            values.append(value)
    return values


def check_entry_drought(
    journal: TradeJournal,
    settings: Settings,
    notifier: WebhookNotifier | NullNotifier,
    *,
    _state: dict[str, float] | None = None,
) -> bool:
    """Alert at most daily when an armed LIVE process has no reconciled BUY fills."""
    state = _drought_alert_state if _state is None else _state
    if settings.trading_mode != "LIVE" or settings.no_entry_alert_hours <= 0:
        return False
    last_entry = journal.get_last_entry_timestamp(account=f"Kraken-{settings.trading_mode}")
    now = dt.datetime.now(tz=dt.UTC)
    threshold_seconds = settings.no_entry_alert_hours * 3600.0
    if last_entry is not None:
        if (now - last_entry).total_seconds() < threshold_seconds:
            return False
    else:
        # Fresh journal (v10 cutover, restart): measure the drought from the
        # first armed check, not from epoch — otherwise every cold start
        # alerts "for ever" immediately and trains the operator to ignore it.
        none_since = state.setdefault("none_since_monotonic", time.monotonic())
        if time.monotonic() - none_since < threshold_seconds:
            return False
    last_alert = state.get("last_alert_monotonic")
    if last_alert is not None and time.monotonic() - last_alert < _DROUGHT_ALERT_THROTTLE_SECONDS:
        return False
    idle_desc = f"{(now - last_entry).days}d" if last_entry is not None else "ever"
    message = (
        f"Entry drought: no reconciled BUY fill for {idle_desc} "
        f"(threshold {settings.no_entry_alert_hours:.0f}h)."
    )
    logger.warning(message)
    notifier.notify_error(message)
    state["last_alert_monotonic"] = time.monotonic()
    return True


def run_cycle(  # noqa: C901, PLR0912, PLR0913, PLR0915
    settings: Settings,
    exchange: KrakenExchange,
    oracle: SentimentOracle,
    strategist: Strategist,
    guardian: RiskGuardian,
    journal: TradeJournal,
    notifier: WebhookNotifier | NullNotifier,
    tech_analyzer: TechnicalAnalyzer | None = None,
    vol_analyzer: VolatilityAnalyzer | None = None,
    price_forecaster: PriceForecaster | None = None,
    portfolio_analyzer: PortfolioAnalyzer | None = None,
    _legacy_paper_portfolio: object | None = None,
    challenger: LLMChallenger | None = None,
    *,
    coordinator: ExecutionCoordinator,
    execution_store: ExecutionStore,
    runtime: RuntimeState,
    execution_venue: KrakenExchange | PaperExchange,
    health: HealthReporter | None = None,
    full_cycle_context: FullCycleContext | None = None,
    cycle_id: str | None = None,
    force_full_cycle: bool | None = None,
    paper_state_path: str | Path | None = None,
) -> None:
    """Run one recovery-first cycle against an exact execution venue."""
    del _legacy_paper_portfolio
    cycle_start = time.monotonic()
    cycle_id = cycle_id or f"cycle-{time.time_ns()}"
    context = full_cycle_context or FullCycleContext()
    is_full_cycle = (
        context.observed_at is None
        or cycle_start - context.observed_at >= settings.full_cycle_seconds
        if force_full_cycle is None
        else force_full_cycle
    )
    if health is not None:
        health.mark_cycle_started(settings.timeout_seconds)
    logger.info(
        "Starting %s cycle id=%s mode=%s",
        "FULL" if is_full_cycle else "FAST",
        cycle_id,
        settings.trading_mode,
    )

    market_exchange = CachedExchange(exchange)
    execution_exchange = CachedExchange(execution_venue)
    guardian.exchange = execution_exchange
    guardian.market_exchange = market_exchange
    guardian.execution_store = execution_store

    try:
        if isinstance(execution_venue, PaperExchange):
            _feed_paper_market_events(settings, market_exchange, execution_venue)

        recovery = _recover_runtime(
            settings,
            coordinator,
            execution_store,
            journal,
            notifier,
            runtime,
            health,
        )
        if recovery is None:
            logger.warning("Cycle aborted because exact recovery did not complete.")
            return
        try:
            with runtime.mutation_section():
                coordinator.cancel_stale_entries(settings.stale_order_max_age_seconds)
        except MutationRejectedError:
            logger.info("Stale cancellation skipped because the runtime is draining.")
        if not coordinator.ready:
            runtime.mark_not_ready()

        if is_full_cycle:
            context.fear_greed = oracle.get_fear_and_greed_index()
            context.polymarket_markets = (
                oracle.fetch_polymarket_markets() if settings.polymarket_enabled else []
            )
        fear_greed = context.fear_greed
        polymarket_markets = context.polymarket_markets or []

        owned_tickers, available_tickers = classify_tickers(
            settings,
            exchange,
            execution_store=execution_store,
        )
        reserved_tickers = reserved_entry_tickers(execution_store)
        logger.info(
            "Universe: owned=%s reserved=%s available=%s",
            owned_tickers,
            sorted(reserved_tickers),
            available_tickers,
        )
        available_slots = compute_available_slots(
            len(available_tickers),
            len(owned_tickers) + len(reserved_tickers),
            settings.max_total_crypto_positions,
        )
        entry_slots_remaining = max(
            settings.max_total_crypto_positions - len(owned_tickers) - len(reserved_tickers),
            0,
        )
        positions = {
            position.symbol: position for position in execution_store.list_open_positions()
        }
        has_position_map = {ticker: ticker in positions for ticker in settings.universe_list}
        buy_price_map = {
            ticker: float(position.average_cost) if (position := positions.get(ticker)) else None
            for ticker in settings.universe_list
        }
        all_tickers = owned_tickers + available_tickers
        all_data = collect_all_market_data(
            all_tickers,
            settings,
            market_exchange,
            oracle,
            journal,
            fear_greed,
            polymarket_markets,
            tech_analyzer,
            vol_analyzer if is_full_cycle else None,
            price_forecaster if is_full_cycle else None,
            has_position_map,
            buy_price_map,
        )
        _apply_quantitative_context(
            settings,
            all_data,
            context,
            cycle_start,
            is_full_cycle=is_full_cycle,
        )
        _apply_position_ages(all_data, positions)

        exit_order_count_before = _exit_order_count(execution_store)
        for ticker in owned_tickers:
            data = all_data.get(ticker)
            if data is None:
                continue
            try:
                if coordinator.ready:
                    evaluate_exit(
                        ticker,
                        data,
                        settings,
                        market_exchange,
                        strategist,
                        guardian,
                        journal,
                        notifier,
                        available_slots,
                        challenger=challenger,
                        coordinator=coordinator,
                        execution_store=execution_store,
                        runtime=runtime,
                        cycle_id=cycle_id,
                    )
                elif coordinator.protection_remediation_only:
                    # Protection-gap issues must not freeze exit management:
                    # restore the exchange stop first, then let the exit path
                    # retry — a successful protected exit resolves stale
                    # handoff failures for the symbol.
                    ensure_stop_loss(
                        ticker,
                        data,
                        settings,
                        market_exchange,
                        coordinator=coordinator,
                        execution_store=execution_store,
                        runtime=runtime,
                    )
                    evaluate_exit(
                        ticker,
                        data,
                        settings,
                        market_exchange,
                        strategist,
                        guardian,
                        journal,
                        notifier,
                        available_slots,
                        challenger=challenger,
                        coordinator=coordinator,
                        execution_store=execution_store,
                        runtime=runtime,
                        cycle_id=cycle_id,
                    )
                else:
                    logger.warning(
                        "Exit mutation skipped for %s while exact recovery is unresolved.",
                        ticker,
                    )
            except Exception as error:  # noqa: BLE001
                logger.warning("Exit evaluation failed for %s: %s", ticker, error)
                notifier.notify_error(f"Exit evaluation failed for {ticker}: {error}")

        exit_activity = _exit_order_count(execution_store) > exit_order_count_before
        active_exit = any(
            order["purpose"] == "EXIT"
            for order in execution_store.list_order_contexts(active_only=True)
        )
        if exit_activity or active_exit:
            if (
                _recover_runtime(
                    settings,
                    coordinator,
                    execution_store,
                    journal,
                    notifier,
                    runtime,
                    health,
                )
                is None
            ):
                logger.warning("Entries blocked because post-exit recovery failed.")
            exit_activity = True

        # Exact post-exit fills must affect the daily-loss gate before another entry.
        daily_loss_ok = check_daily_loss_limit(execution_store, settings)
        if daily_loss_ok and runtime.ready and coordinator.ready and not exit_activity:
            entries_this_cycle = 0
            try:
                cycle_budget_usd = float(
                    execution_exchange.get_free_balance(settings.quote_currency),
                )
            except Exception:
                logger.exception("Execution venue balance read failed; all entries are blocked.")
                cycle_budget_usd = 0.0
            exposure_tickers = [*owned_tickers, *sorted(reserved_tickers)]
            for ticker in available_tickers:
                if entries_this_cycle >= settings.max_entries_per_cycle:
                    break
                if entry_slots_remaining <= 0:
                    break
                data = all_data.get(ticker)
                if data is None:
                    continue
                try:
                    submitted, reserved_usd = evaluate_entry(
                        ticker,
                        data,
                        settings,
                        market_exchange,
                        strategist,
                        guardian,
                        journal,
                        notifier,
                        available_slots,
                        exposure_tickers,
                        challenger=challenger,
                        budget_usd=cycle_budget_usd,
                        coordinator=coordinator,
                        execution_store=execution_store,
                        runtime=runtime,
                        cycle_id=cycle_id,
                    )
                    if submitted:
                        entries_this_cycle += 1
                        entry_slots_remaining -= 1
                        available_slots = max(entry_slots_remaining, 1)
                        exposure_tickers.append(ticker)
                        cycle_budget_usd = max(cycle_budget_usd - reserved_usd, 0.0)
                except Exception as error:  # noqa: BLE001
                    logger.warning("Entry evaluation failed for %s: %s", ticker, error)
                    notifier.notify_error(f"Entry evaluation failed for {ticker}: {error}")
        else:
            logger.info("Entries blocked by daily loss or reconciliation readiness.")

        if portfolio_analyzer is not None:
            portfolio_analyzer.log_summary()
        if settings.live_entries_enabled:
            check_entry_drought(journal, settings, notifier)
        if health is not None:
            health.mark_cycle_completed()
            health.set_state(
                HealthState.READY if runtime.ready and coordinator.ready else HealthState.DEGRADED,
                error_codes=()
                if runtime.ready and coordinator.ready
                else ("RECONCILIATION_PENDING",),
            )
        duration_seconds = time.monotonic() - cycle_start
        logger.info("Cycle complete id=%s duration=%.1fs", cycle_id, duration_seconds)
        notifier.notify_cycle_summary(f"Cycle {cycle_id} completed in {duration_seconds:.1f}s")
    finally:
        if isinstance(execution_venue, PaperExchange) and paper_state_path is not None:
            _save_paper_exchange(execution_venue, paper_state_path, execution_store)


def _recover_runtime(  # noqa: PLR0913
    settings: Settings,
    coordinator: ExecutionCoordinator,
    store: ExecutionStore,
    journal: TradeJournal,
    notifier: WebhookNotifier | NullNotifier,
    runtime: RuntimeState,
    health: HealthReporter | None,
) -> RecoverySummary | None:
    """Run exact recovery, publish fills, and derive runtime readiness."""
    if health is not None and not runtime.ready:
        health.set_state(HealthState.RECOVERING)
    try:
        summary = coordinator.recover(
            settings.universe_list,
            Decimal(str(settings.reconciliation_balance_tolerance)),
        )
        _publish_reconciled_fills(summary, store, journal, notifier, settings)
        if health is not None:
            health.mark_reconciliation()
    except Exception:
        logger.exception("Exact recovery failed; entries remain blocked.")
        runtime.mark_not_ready()
        if health is not None:
            health.set_state(HealthState.DEGRADED, error_codes=("RECOVERY_FAILED",))
        return None
    if summary.readiness.ready:
        runtime.mark_ready()
    else:
        runtime.mark_not_ready()
        logger.warning("Exact recovery has %d unresolved issue(s).", len(summary.readiness.issues))
    return summary


def _publish_reconciled_fills(
    summary: RecoverySummary,
    store: ExecutionStore,
    journal: TradeJournal,
    notifier: WebhookNotifier | NullNotifier,
    settings: Settings,
) -> None:
    """Mirror every durable, previously unpublished fill into analytics."""
    _ = summary
    rows = store.connection.execute(
        "SELECT fill_id, symbol, side, price, quantity, cost, fee, executed_at_ms "
        "FROM fills ORDER BY executed_at_ms, fill_id COLLATE BINARY",
    ).fetchall()
    for row in rows:
        fill_id = str(row["fill_id"])
        if journal.is_fill_recorded(fill_id):
            continue
        realized_row = store.connection.execute(
            "SELECT realized_pnl FROM position_events WHERE fill_id = ?",
            (fill_id,),
        ).fetchone()
        realized = Decimal(0) if realized_row is None else Decimal(str(realized_row[0]))
        side = str(row["side"])
        price = Decimal(str(row["price"]))
        quantity = Decimal(str(row["quantity"]))
        fee = Decimal(str(row["fee"]))
        symbol = str(row["symbol"])
        quote = symbol.split("/", maxsplit=1)[1]
        timestamp = dt.datetime.fromtimestamp(int(row["executed_at_ms"]) / 1000, tz=dt.UTC)
        tax = max(realized, Decimal(0)) * Decimal(str(settings.tax_rate))
        journal.log_trade(
            account=f"Kraken-{settings.trading_mode}",
            instrument=symbol,
            side=side,
            price=float(price),
            quantity=float(quantity),
            value=float(Decimal(str(row["cost"]))),
            currency=quote,
            reasoning=f"Exact reconciled fill [{fill_id}]",
            profit=float(realized),
            tax=float(tax),
            fee=float(fee),
            fill_source="exchange",
            exchange_fill_id=fill_id,
            timestamp=timestamp,
        )
        notifier.notify_trade(
            symbol,
            side,
            float(price),
            float(quantity),
            f"Exact reconciled fill [{fill_id}]",
        )
        if store.get_position(symbol) is None:
            journal.delete_position_state(symbol)


def _exit_order_count(store: ExecutionStore) -> int:
    """Return durable exit intent count for current-cycle mutation detection."""
    return sum(1 for order in store.list_order_contexts() if order["purpose"] == "EXIT")


def _apply_quantitative_context(
    settings: Settings,
    data_by_ticker: dict[str, MarketData],
    context: FullCycleContext,
    observed_at: float,
    *,
    is_full_cycle: bool,
) -> None:
    """Cache FULL outputs or inject only a still-fresh cache into FAST data."""
    assert context.volatility_by_ticker is not None  # noqa: S101
    assert context.forecast_by_ticker is not None  # noqa: S101
    volatility_cache = context.volatility_by_ticker
    forecast_cache = context.forecast_by_ticker
    if is_full_cycle:
        context.observed_at = observed_at
        volatility_cache.clear()
        forecast_cache.clear()
        for ticker, data in data_by_ticker.items():
            if data.volatility is not None:
                volatility_cache[ticker] = data.volatility
            if data.forecast is not None:
                forecast_cache[ticker] = data.forecast
    age = None if context.observed_at is None else observed_at - context.observed_at
    cache_fresh = age is not None and 0 <= age <= settings.full_cycle_seconds
    for ticker, data in data_by_ticker.items():
        if not is_full_cycle and cache_fresh:
            data.volatility = volatility_cache.get(ticker)  # type: ignore[assignment]
            data.forecast = forecast_cache.get(ticker)  # type: ignore[assignment]
        required_volatility = settings.garch_enabled
        required_forecast = settings.arima_enabled
        data.quantitative_context_fresh = (
            cache_fresh
            and (not required_volatility or data.volatility is not None)
            and (not required_forecast or data.forecast is not None)
        )
        data.quantitative_context_observed_at = context.observed_at


def _apply_position_ages(
    data_by_ticker: dict[str, MarketData],
    positions: dict[str, PositionSnapshot],
) -> None:
    """Derive entry price and age only from reconciled ExecutionStore positions."""
    now_ms = int(time.time() * 1000)
    for ticker, data in data_by_ticker.items():
        position = positions.get(ticker)
        if position is None:
            data.has_position = False
            data.entry_price = None
            data.position_age_hours = None
            continue
        data.has_position = True
        data.entry_price = float(position.average_cost)
        data.position_age_hours = max(now_ms - position.opened_at_ms, 0) / 3_600_000


def _feed_paper_market_events(
    settings: Settings,
    market_exchange: CachedExchange,
    paper_exchange: PaperExchange,
) -> None:
    """Advance PAPER orders with fresh quotes before any new decision is evaluated."""
    state_clock = int(paper_exchange.to_state()["sequences"]["clock"])
    base_timestamp = max(int(time.time() * 1000) - len(settings.universe_list), state_clock + 1)
    for index, symbol in enumerate(settings.universe_list):
        structure = market_exchange.get_market_structure(symbol)
        paper_exchange.process_market_event(
            symbol,
            bid=structure["bid"],
            ask=structure["ask"],
            last=market_exchange.fetch_price(symbol),
            timestamp=base_timestamp + index,
        )


def _paper_markets(settings: Settings) -> dict[str, MarketRules]:
    """Build deterministic high-precision PAPER rules for the configured universe."""
    markets: dict[str, MarketRules] = {}
    for symbol in settings.universe_list:
        base, quote = symbol.split("/", maxsplit=1)
        markets[symbol] = MarketRules(
            base,
            quote,
            price_increment="0.000000000000000000000001",
            amount_increment="0.000000000000000000000001",
            min_notional="10",
            min_amount="0.000000000000000000000001",
        )
    return markets


def _load_paper_exchange(settings: Settings) -> PaperExchange:
    path = Path(settings.paper_exchange_state_path)
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        exchange = PaperExchange.from_state(state)
        if any(not exchange.symbol_exists(symbol) for symbol in settings.universe_list):
            message = "paper exchange checkpoint does not match the configured universe"
            raise ValueError(message)
        return exchange
    balances = {settings.quote_currency: Decimal(str(settings.paper_initial_balance_usd))}
    balances.update(
        {symbol.split("/", maxsplit=1)[0]: Decimal(0) for symbol in settings.universe_list},
    )
    return PaperExchange(
        balances,
        _paper_markets(settings),
        start_timestamp=int(time.time() * 1000),
    )


def _save_paper_exchange(
    exchange: PaperExchange,
    state_path: str | Path,
    execution_store: ExecutionStore | None = None,
) -> None:
    """Atomically checkpoint PAPER cash, reservations, orders, and sequences."""
    if execution_store is not None:
        pending_fill_ids = {
            str(fill["id"])
            for fill in exchange.fetch_my_trades()
            if execution_store.connection.execute(
                "SELECT 1 FROM fills WHERE fill_id = ?",
                (str(fill["id"]),),
            ).fetchone()
            is None
        }
        if pending_fill_ids:
            message = "PAPER checkpoint refused because exact fills are not durable"
            raise RuntimeError(message)
    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as state_file:
            json.dump(exchange.to_state(), state_file, sort_keys=True, separators=(",", ":"))
            state_file.write("\n")
            state_file.flush()
            os.fsync(state_file.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _execution_scope(settings: Settings) -> str:
    if settings.trading_mode == "PAPER":
        return "paper:kraken:local"
    assert settings.trading_account_id is not None  # noqa: S101
    account = settings.trading_account_id.get_secret_value()
    digest = hashlib.sha256(account.encode()).hexdigest()[:16]
    return f"live:kraken:{digest}"


def main() -> int:  # noqa: C901, PLR0912, PLR0915
    """Initialize all v10 resources, run until drain, and return a process exit code."""
    try:
        settings = Settings()
    except Exception:
        logging.basicConfig(level=logging.INFO)
        logger.critical("Failed to load settings.", exc_info=True)
        return 1

    redactor = setup_logging(settings)
    logger.info("Oracle Pythia v10 booting mode=%s", settings.trading_mode)
    singleton: ProcessSingleton | None = None
    runtime: RuntimeState | None = None
    health: HealthReporter | None = None
    store: ExecutionStore | None = None
    journal: TradeJournal | None = None
    paper_exchange: PaperExchange | None = None
    exchange: KrakenExchange | None = None
    exit_code = 0
    try:
        mode_lock = Path(settings.singleton_lock_path).with_name(
            f"{Path(settings.singleton_lock_path).stem}-{settings.trading_mode.lower()}.lock",
        )
        singleton = ProcessSingleton(mode_lock)
        singleton.acquire()
        runtime = RuntimeState()
        signal.signal(signal.SIGTERM, runtime.handle_signal)
        signal.signal(signal.SIGINT, runtime.handle_signal)
        health = HealthReporter(
            settings.healthcheck_file,
            app_version=APP_VERSION,
            build_version=os.environ.get("BUILD_VERSION", "local"),
        )
        exchange = KrakenExchange(settings.kraken_api_key, settings.kraken_private_key)
        store = ExecutionStore(settings.state_db_path, _execution_scope(settings))
        journal = TradeJournal(db_path=settings.analytics_db_path)
        budget = LLMBudgetManager(
            settings.llm_budget_db_path,
            daily_cap_usd=settings.llm_daily_shadow_budget_usd,
            monthly_cap_usd=settings.llm_monthly_budget_usd,
            shadow_cap_usd=settings.llm_monthly_budget_usd,
        )
        execution_venue: KrakenExchange | PaperExchange = exchange
        if settings.trading_mode == "PAPER":
            paper_exchange = _load_paper_exchange(settings)
            execution_venue = paper_exchange
        coordinator = ExecutionCoordinator(
            store,
            execution_venue,
            live_entries_enabled=(
                settings.trading_mode == "PAPER" or settings.live_entries_enabled
            ),
        )
        notifier: WebhookNotifier | NullNotifier = (
            WebhookNotifier(settings.webhook_url, settings.error_webhook_url, redactor=redactor)
            if settings.webhook_url
            else NullNotifier()
        )
        extra_api_keys = {
            provider: key.get_secret_value()
            for provider, key in (
                ("gemini", settings.gemini_api_key),
                ("anthropic", settings.anthropic_api_key),
                ("openai", settings.openai_api_key),
            )
            if key is not None
        }
        oracle = SentimentOracle(
            settings.xai_api_key,
            settings.sentiment_model,
            settings.cmc_api_key,
            warp_proxy=settings.warp_proxy,
            fallback_model=settings.sentiment_fallback_model,
            extra_api_keys=extra_api_keys,
            cache_ttl_seconds=settings.sentiment_cache_seconds,
            enabled=settings.sentiment_enabled,
            budget_manager=budget,
            max_tokens=settings.llm_max_tokens,
            timeout_seconds=settings.llm_timeout_seconds,
        )
        guardian = RiskGuardian(
            settings,
            execution_venue,
            market_exchange=exchange,
            execution_store=store,
        )
        challenger = build_challenger_from_settings(settings, budget)
        strategist = Strategist(settings=settings)
        technical = TechnicalAnalyzer()
        volatility = VolatilityAnalyzer()
        forecaster = PriceForecaster(max_order=settings.arima_max_order)
        portfolio_analyzer = PortfolioAnalyzer(journal=journal)
        full_context = FullCycleContext()

        _recover_runtime(settings, coordinator, store, journal, notifier, runtime, health)
        while not runtime.shutdown_event.is_set():
            try:
                run_cycle(
                    settings,
                    exchange,
                    oracle,
                    strategist,
                    guardian,
                    journal,
                    notifier,
                    technical,
                    volatility,
                    forecaster,
                    portfolio_analyzer,
                    challenger=challenger,
                    coordinator=coordinator,
                    execution_store=store,
                    runtime=runtime,
                    execution_venue=execution_venue,
                    health=health,
                    full_cycle_context=full_context,
                    paper_state_path=(
                        settings.paper_exchange_state_path
                        if settings.trading_mode == "PAPER"
                        else None
                    ),
                )
            except Exception:
                logger.exception("Cycle failed with an unhandled error.")
                notifier.notify_error("Cycle failed with an unhandled error.")
                runtime.mark_not_ready()
                health.set_state(HealthState.DEGRADED, error_codes=("CYCLE_FAILED",))
            singleton.refresh()
            health.heartbeat()
            if runtime.shutdown_event.wait(settings.entry_cycle_seconds):
                break
    except Exception:
        logger.critical("Component initialization failed.", exc_info=True)
        exit_code = 1
    finally:
        if runtime is not None:
            if health is not None:
                health.set_state(HealthState.DRAINING)
            drained = runtime.stop(settings.shutdown_drain_timeout_seconds)
            if not drained:
                logger.error("Mutation drain timed out; resources are closing after the bound.")
                exit_code = 1
        if paper_exchange is not None:
            try:
                _save_paper_exchange(
                    paper_exchange,
                    settings.paper_exchange_state_path,
                    store,
                )
            except Exception:
                logger.exception("Final PAPER checkpoint failed.")
                exit_code = 1
        if journal is not None:
            journal.close()
        if store is not None:
            store.close()
        if exchange is not None:
            close_exchange = getattr(exchange._client, "close", None)  # noqa: SLF001
            if callable(close_exchange):
                close_exchange()
        if singleton is not None:
            singleton.release()
        if health is not None:
            health.set_state(HealthState.STOPPED)
    logger.info("Oracle Pythia stopped exit_code=%d", exit_code)
    logging.shutdown()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
