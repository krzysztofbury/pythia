"""Decision engine — evaluates entry/exit for each ticker using strategy + guardian."""

from __future__ import annotations

import contextlib
import dataclasses
import logging
from decimal import Decimal
from typing import TYPE_CHECKING

from execution_coordinator import SubmissionOutcome
from models import PositionState, StopMode, fmt_price
from order_executor import (
    check_take_profit,
    compute_protection_prices,
    ensure_stop_loss,
    execute_dca,
    execute_partial_sell,
    execute_sell,
)
from regime import RegimeClassifier
from risk_guardian import (
    REGULATORY_DANGER_PHRASES,
    RiskGuardian,
    check_correlation_limit,
)
from stop_manager import StopManager

if TYPE_CHECKING:
    from cache import CachedExchange
    from config import Settings
    from execution_coordinator import ExecutionCoordinator
    from execution_store import ExecutionStore
    from journal import TradeJournal
    from llm_challenger import Critique, LLMChallenger
    from models import MarketData, TradeSignal
    from notifier import NullNotifier, WebhookNotifier
    from paper_portfolio import PaperPortfolio
    from runtime_state import RuntimeState
    from strategy import Strategist

logger = logging.getLogger("OraclePythia")


def _run_challenger_and_format(
    challenger: LLMChallenger | None,
    signal: TradeSignal,
    data: MarketData,
    notifier: WebhookNotifier | NullNotifier,
) -> str:
    """Run the challenger (if provided) and return a journal-safe string.

    Notifies on HIGH concern. Never raises — defensive even if ``review``
    breaks its "never raise" contract in the future, so a challenger bug
    can't kill a trading cycle.
    """
    if challenger is None:
        return ""
    try:
        critique: Critique | None = challenger.review(signal, data)
    except Exception:
        logger.exception(
            "LLMChallenger.review raised unexpectedly; continuing without critique",
        )
        return ""
    if critique is None:
        return ""
    if critique.concern_level == "HIGH":
        try:
            notifier.notify_error(
                f"LLM Challenger HIGH concern on {data.ticker} {signal.action}: "
                f"{critique.thesis[:200]}",
            )
        except Exception:  # noqa: BLE001
            logger.debug(
                "notifier.notify_error failed for challenger HIGH concern",
                exc_info=True,
            )
    return critique.format_for_journal()


def _log_signal_after_execution(  # noqa: PLR0913
    ticker: str,
    signal: TradeSignal,
    data: MarketData,
    journal: TradeJournal,
    notifier: WebhookNotifier | NullNotifier,
    challenger: LLMChallenger | None,
) -> None:
    """Persist the algorithmic decision, then run shadow commentary off-path."""
    critique_text = _run_challenger_and_format(challenger, signal, data, notifier)
    vol_regime = data.volatility.regime.value if data.volatility else "UNAVAILABLE"
    journal.log_signal(
        ticker,
        signal.action,
        signal.confidence,
        signal.composite,
        vol_regime,
        signal.reasoning,
        llm_critique=critique_text,
    )


def evaluate_exit(  # noqa: C901, PLR0913
    ticker: str,
    data: MarketData,
    settings: Settings,
    exchange: CachedExchange,
    strategist: Strategist,
    guardian: RiskGuardian,
    journal: TradeJournal,
    notifier: WebhookNotifier | NullNotifier,
    available_slots: int,
    paper_portfolio: PaperPortfolio | None = None,
    challenger: LLMChallenger | None = None,
    *,
    coordinator: ExecutionCoordinator | None = None,
    execution_store: ExecutionStore | None = None,
    runtime: RuntimeState | None = None,
    cycle_id: str | None = None,
) -> None:
    """Evaluate whether to exit an owned position, or DCA into it."""
    logger.info("Exit evaluation: %s", ticker)

    stop_manager = StopManager(settings)
    pos_state = journal.get_position_state(ticker)
    if coordinator is not None and pos_state is None and execution_store is not None:
        position = execution_store.get_position(ticker)
        if position is not None:
            stop_loss_price, take_profit_price = compute_protection_prices(
                float(position.average_cost),
                settings,
                data,
            )
            pos_state = PositionState(
                ticker=ticker,
                entry_price=float(position.average_cost),
                high_watermark=max(float(position.average_cost), data.current_price),
                stop_mode=StopMode.FIXED,
                current_sl=stop_loss_price,
                current_tp=take_profit_price,
                regime_at_entry="RECONCILED",
            )
            journal.upsert_position_state(pos_state)

    if pos_state and data.current_price > 0:
        journal.update_high_watermark(ticker, data.current_price)
        pos_state.high_watermark = max(pos_state.high_watermark, data.current_price)

        atr = data.indicators.atr if data.indicators else None
        pnl_pct = data.pnl_pct or 0.0
        age_minutes = (data.position_age_hours or 0.0) * 60.0

        stop_action = stop_manager.adjust(
            pos_state,
            data.current_price,
            atr,
            pnl_pct,
            age_minutes,
        )

        if stop_action.action == "SELL":
            logger.info("StopManager SELL for %s: %s", ticker, stop_action.reason)
            sold = execute_sell(
                ticker,
                data.current_price,
                data.entry_price,
                f"StopManager: {stop_action.reason}",
                settings,
                exchange,
                journal,
                notifier,
                paper_portfolio,
                coordinator=coordinator,
                execution_store=execution_store,
                runtime=runtime,
                cycle_id=cycle_id,
            )
            # v9.7: keep stop tracking when the sell did not go through —
            # the coins are still held and the next cycle must retry.
            if sold and coordinator is None:
                journal.delete_position_state(ticker)
            return

        if stop_action.action == "PARTIAL_SELL":
            logger.info("StopManager PARTIAL SELL for %s: %s", ticker, stop_action.reason)
            success = execute_partial_sell(
                ticker,
                data.current_price,
                stop_action.sell_pct,
                f"Partial: {stop_action.reason}",
                settings,
                exchange,
                journal,
                notifier,
                paper_portfolio,
                coordinator=coordinator,
            )
            if success:
                pos_state.partial_exits += 1
                journal.upsert_position_state(pos_state)

        if stop_action.updated_sl != pos_state.current_sl:
            pos_state.current_sl = stop_action.updated_sl
            journal.upsert_position_state(pos_state)

    if check_take_profit(
        ticker,
        data,
        settings,
        exchange,
        journal,
        notifier,
        paper_portfolio,
        coordinator=coordinator,
        execution_store=execution_store,
        runtime=runtime,
        cycle_id=cycle_id,
    ):
        return

    ensure_stop_loss(
        ticker,
        data,
        settings,
        exchange,
        coordinator=coordinator,
        execution_store=execution_store,
        runtime=runtime,
        desired_stop_price=None if pos_state is None else pos_state.current_sl,
    )

    strategy_data = _strategy_data(data, settings)
    trade_signal = strategist.generate_signal(strategy_data)
    logger.info(
        "Exit signal for %s: %s (Conf: %s) - %s",
        ticker,
        trade_signal.action,
        trade_signal.confidence,
        trade_signal.reasoning,
    )

    if trade_signal.action == "SELL" and trade_signal.confidence >= settings.min_sell_confidence:
        sold = execute_sell(
            ticker,
            data.current_price,
            data.entry_price,
            f"Strategic Exit: {trade_signal.reasoning}",
            settings,
            exchange,
            journal,
            notifier,
            paper_portfolio,
            coordinator=coordinator,
            execution_store=execution_store,
            runtime=runtime,
            cycle_id=cycle_id,
        )
        if sold and coordinator is None:
            journal.delete_position_state(ticker)
        _log_signal_after_execution(
            ticker,
            trade_signal,
            strategy_data,
            journal,
            notifier,
            challenger,
        )
        return

    execute_dca(
        ticker,
        data,
        trade_signal,
        settings,
        exchange,
        guardian,
        journal,
        notifier,
        available_slots,
        paper_portfolio,
        coordinator=coordinator,
    )
    _log_signal_after_execution(
        ticker,
        trade_signal,
        strategy_data,
        journal,
        notifier,
        challenger,
    )


def evaluate_entry(  # noqa: C901, PLR0911, PLR0912, PLR0913, PLR0915
    ticker: str,
    data: MarketData,
    settings: Settings,
    exchange: CachedExchange,
    strategist: Strategist,
    guardian: RiskGuardian,
    journal: TradeJournal,
    notifier: WebhookNotifier | NullNotifier,
    available_slots: int,
    owned_tickers: list[str],
    paper_portfolio: PaperPortfolio | None = None,
    challenger: LLMChallenger | None = None,
    budget_usd: float | None = None,
    *,
    coordinator: ExecutionCoordinator | None = None,
    execution_store: ExecutionStore | None = None,
    runtime: RuntimeState | None = None,
    cycle_id: str | None = None,
) -> tuple[bool, float]:
    """Evaluate whether to enter a new position.

    Returns ``(was_bought, target_usd_spent)`` — the caller uses the dollar
    amount to decrement an intra-cycle budget so a second entry in the same
    cycle cannot reuse the stale cached balance.
    """
    logger.info("Entry evaluation: %s", ticker)

    if coordinator is not None:
        if execution_store is None or runtime is None or not cycle_id:
            msg = "coordinator entries require execution_store, runtime, and cycle_id"
            raise ValueError(msg)
        if settings.trading_mode == "LIVE" and not settings.live_entries_enabled:
            logger.info("Entry blocked for %s: LIVE entries are not armed.", ticker)
            return (False, 0.0)
        if not runtime.ready or not coordinator.ready:
            logger.warning("Entry blocked for %s: runtime reconciliation is not ready.", ticker)
            return (False, 0.0)
        if _has_active_buy(execution_store, ticker):
            logger.info("Entry skipped for %s: an exact BUY order remains active.", ticker)
            return (False, 0.0)
    elif not data.quantitative_context_fresh:
        # Legacy direct tests predate explicit context freshness and retain
        # their isolated behavior; the coordinator production path is strict.
        pass

    if coordinator is not None and not data.quantitative_context_fresh:
        logger.warning("Entry blocked for %s: quantitative context is absent or stale.", ticker)
        return (False, 0.0)

    open_orders = [] if coordinator is not None else exchange.get_open_orders(ticker)
    if open_orders:
        logger.info(
            "Entry: Skipping %s - %d open orders remain.",
            ticker,
            len(open_orders),
        )
        return (False, 0.0)

    if not guardian.check_market_conditions(ticker):
        return (False, 0.0)

    if not check_correlation_limit(ticker, owned_tickers, settings):
        return (False, 0.0)

    price = data.current_price

    strategy_data = _strategy_data(data, settings)
    combined = (strategy_data.sentiment + " " + strategy_data.consensus).upper()
    if any(phrase in combined for phrase in REGULATORY_DANGER_PHRASES):
        logger.warning(
            "Entry: Regulatory Risk detected for %s. Abort.",
            ticker,
        )
        return (False, 0.0)

    logger.info(
        "Data for %s: Price=%s, Sentiment=%s..., Consensus=%s...",
        ticker,
        fmt_price(price),
        strategy_data.sentiment[:30],
        strategy_data.consensus[:30],
    )
    logger.info("Polymarket for %s: %s", ticker, data.polymarket_odds)

    trade_signal = strategist.generate_signal(strategy_data)
    logger.info(
        "Signal: %s (Conf: %s) - %s",
        trade_signal.action,
        trade_signal.confidence,
        trade_signal.reasoning,
    )

    if trade_signal.action != "BUY":
        _log_signal_after_execution(
            ticker,
            trade_signal,
            strategy_data,
            journal,
            notifier,
            challenger,
        )
        return (False, 0.0)

    limit_price = price * (1 - settings.limit_order_discount_pct)
    amount = guardian.check_buy(
        ticker,
        limit_price,
        trade_signal,
        available_slots,
        budget_usd=budget_usd,
    )
    if amount <= 0:
        return (False, 0.0)
    target_usd = amount * limit_price

    stop_loss_price, take_profit_price = compute_protection_prices(
        limit_price,
        settings,
        data,
    )

    if stop_loss_price <= 0 or stop_loss_price >= price:
        logger.warning(
            "Entry blocked for %s: protective stop %.8f is not below current price %.8f.",
            ticker,
            stop_loss_price,
            price,
        )
        _log_signal_after_execution(
            ticker,
            trade_signal,
            strategy_data,
            journal,
            notifier,
            challenger,
        )
        return (False, 0.0)

    if coordinator is not None:
        guard = runtime.mutation_section() if runtime is not None else contextlib.nullcontext()
        with guard:
            submission = coordinator.submit_entry(
                f"{cycle_id}:entry:{ticker}",
                ticker,
                Decimal(str(amount)),
                Decimal(str(limit_price)),
                stop_loss_price=Decimal(str(stop_loss_price)),
                payload={"reason": trade_signal.reasoning, "cycle_id": cycle_id},
            )
        if runtime is not None and not coordinator.ready:
            runtime.mark_not_ready()
        submitted = submission.outcome in {
            SubmissionOutcome.SUBMITTED,
            SubmissionOutcome.ALREADY_KNOWN,
        }
        _log_signal_after_execution(
            ticker,
            trade_signal,
            strategy_data,
            journal,
            notifier,
            challenger,
        )
        return (submitted, target_usd if submitted else 0.0)

    if settings.trading_mode == "LIVE":
        logger.info(">>> EXECUTING LIVE SNIPER BUY for %s <<<", ticker)
        exchange.place_maker_buy(ticker, limit_price, amount)

        try:
            exchange.place_protection_orders(
                ticker,
                amount,
                limit_price,
                stop_loss_price,
                take_profit_price,
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "Protection orders deferred for %s — buy not yet filled. "
                "SL will be placed on next cycle when position is detected.",
                ticker,
            )
    else:
        logger.info(
            ">>> PAPER SNIPER: BUY %s @ $%s <<<",
            ticker,
            limit_price,
        )
        if paper_portfolio:
            paper_portfolio.record_buy(ticker, limit_price, amount)

    journal.log_trade(
        account="Kraken",
        instrument=ticker,
        side="BUY",
        price=limit_price,
        quantity=amount,
        value=limit_price * amount,
        currency=settings.quote_currency,
        reasoning=trade_signal.reasoning,
        fee=limit_price * amount * 0.0025,
    )
    notifier.notify_trade(
        ticker,
        "BUY",
        limit_price,
        amount,
        trade_signal.reasoning,
    )

    regime_classifier = RegimeClassifier()
    regime = (
        regime_classifier.classify(
            adx=data.indicators.adx if data.indicators else None,
            ema_bullish_cross=data.indicators.ema_bullish_cross if data.indicators else None,
            roc=data.indicators.roc if data.indicators else None,
            bb_pct_b=data.indicators.bb_pct_b if data.indicators else None,
            obv_slope=data.indicators.obv_slope if data.indicators else None,
        )
        if data.indicators
        else None
    )

    stop_mode_str = (
        RegimeClassifier.get_stop_mode_for_regime(regime) if regime else StopMode.FIXED.value
    )
    stop_mode = StopMode(stop_mode_str)

    pos_state = PositionState(
        ticker=ticker,
        entry_price=limit_price,
        high_watermark=limit_price,
        stop_mode=stop_mode,
        current_sl=stop_loss_price,
        current_tp=take_profit_price,
        regime_at_entry=regime.value if regime else "RANGING",
    )
    journal.upsert_position_state(pos_state)

    return (True, target_usd)


def _strategy_data(data: MarketData, settings: Settings) -> MarketData:
    """Strip unadmitted external factors while retaining them for analytics."""
    # Mode-agnostic on purpose: PAPER must be able to exercise the exact
    # sentiment-influenced strategy path before the operator arms it in LIVE.
    # The flag still defaults off and requires SENTIMENT_ENABLED.
    llm_influence = settings.llm_live_influence_enabled
    polymarket_influence = settings.polymarket_strategy_influence_enabled
    if llm_influence and polymarket_influence:
        return data
    return dataclasses.replace(
        data,
        sentiment=(
            data.sentiment
            if llm_influence
            else "NEUTRAL (Score: 5/10): LLM strategy influence disabled."
        ),
        consensus=data.consensus if llm_influence else "Consensus unavailable.",
        polymarket_odds=(
            data.polymarket_odds
            if polymarket_influence
            else "Polymarket: No data available for strategy."
        ),
    )


def _has_active_buy(store: ExecutionStore, ticker: str) -> bool:
    """Check exact persisted BUY orders without fuzzy exchange matching."""
    return any(
        store.get_order_context(order.order_id)["symbol"] == ticker
        and store.get_order_context(order.order_id)["side"] == "BUY"
        for order in store.list_orders(active_only=True)
    )
