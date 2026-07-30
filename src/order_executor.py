"""Order execution — sell, partial sell, DCA, protection orders, fill reconciliation."""

from __future__ import annotations

import contextlib
import logging
from decimal import Decimal
from typing import TYPE_CHECKING

from execution_coordinator import SubmissionOutcome
from execution_store import OrderState
from models import MarketData, PositionState, StopMode, TradeSignal, fmt_price
from regime import RegimeClassifier
from risk_guardian import MIN_ORDER_USD, RiskGuardian

if TYPE_CHECKING:
    from cache import CachedExchange
    from config import Settings
    from exchange import KrakenExchange
    from execution_coordinator import ExecutionCoordinator
    from execution_store import ExecutionStore
    from journal import TradeJournal
    from notifier import NullNotifier, WebhookNotifier
    from paper_portfolio import PaperPortfolio
    from runtime_state import RuntimeState

logger = logging.getLogger("OraclePythia")

# Fills worth less than this are partial-fill residues, not positions.
DUST_FILL_USD = 1.0
# Must cover the stale-order lifetime (2h) plus reconciliation lag.
FILL_DEDUP_WINDOW_MINUTES = 180


def compute_protection_prices(
    base_price: float,
    settings: Settings,
    data: MarketData,
) -> tuple[float, float]:
    """Compute stop-loss and take-profit prices.

    Uses ATR-based stops when available (more adaptive to market conditions),
    falling back to fixed percentage.

    Returns (stop_loss_price, take_profit_price).
    """
    if settings.use_atr_stops and data.indicators and data.indicators.atr:
        atr = data.indicators.atr
        stop_loss_price = base_price - (atr * settings.atr_sl_multiplier)
        take_profit_price = base_price + (atr * settings.atr_tp_multiplier)

        if settings.dynamic_risk_enabled and data.volatility:
            stop_loss_price = base_price - (
                atr * settings.atr_sl_multiplier * data.volatility.suggested_sl_multiplier
            )
            take_profit_price = base_price + (
                atr * settings.atr_tp_multiplier * data.volatility.suggested_tp_multiplier
            )

        stop_loss_price = max(stop_loss_price, base_price * 0.5)
        assert stop_loss_price > 0, f"SL must be positive, got {stop_loss_price}"  # noqa: S101
        assert take_profit_price > base_price  # noqa: S101

        return stop_loss_price, take_profit_price

    stop_loss_price = base_price * (1 - settings.stop_loss_pct)
    take_profit_price = base_price * (1 + settings.take_profit_pct)

    if settings.dynamic_risk_enabled and data.volatility:
        stop_loss_multiplier = data.volatility.suggested_sl_multiplier
        take_profit_multiplier = data.volatility.suggested_tp_multiplier
        stop_loss_price = base_price * (1 - settings.stop_loss_pct * stop_loss_multiplier)
        take_profit_price = base_price * (1 + settings.take_profit_pct * take_profit_multiplier)

    return stop_loss_price, take_profit_price


def execute_sell(  # noqa: C901, PLR0911, PLR0912, PLR0913
    ticker: str,
    price: float,
    buy_price: float | None,
    reason: str,
    settings: Settings,
    exchange: CachedExchange,
    journal: TradeJournal,
    notifier: WebhookNotifier | NullNotifier,
    paper_portfolio: PaperPortfolio | None = None,
    *,
    coordinator: ExecutionCoordinator | None = None,
    execution_store: ExecutionStore | None = None,
    runtime: RuntimeState | None = None,
    cycle_id: str | None = None,
) -> bool:
    """Execute a sell order (shared by TP and AI-driven exits).

    Returns True only when an order was actually placed (or paper-recorded).
    Callers must NOT delete position state on False — v9.7: a failed sell
    used to drop stop tracking while the coins were still held.
    """
    assert price > 0, f"Sell price must be positive, got {price}"  # noqa: S101

    if coordinator is not None:
        if execution_store is None or not cycle_id:
            msg = "coordinator exits require execution_store and cycle_id"
            raise ValueError(msg)
        position = execution_store.get_position(ticker)
        if position is None:
            logger.info("Sell skipped for %s: no reconciled position.", ticker)
            return False
        if Decimal(str(price)) * position.quantity < Decimal(str(MIN_ORDER_USD)):
            logger.info("Sell skipped for %s: reconciled position is below venue minimum.", ticker)
            return False
        guard = runtime.mutation_section() if runtime is not None else contextlib.nullcontext()
        with guard:
            result = coordinator.submit_protected_exit(
                f"{cycle_id}:exit:{ticker}",
                ticker,
                position.quantity,
                Decimal(str(price)),
                payload={"reason": reason},
            )
        if runtime is not None and not coordinator.ready:
            runtime.mark_not_ready()
        if result.submission is not None and result.submission.outcome in {
            SubmissionOutcome.SUBMITTED,
            SubmissionOutcome.ALREADY_KNOWN,
        }:
            logger.info(
                "Exit for %s is pending exact fills; inventory remains authoritative.",
                ticker,
            )
        return result.submission is not None and result.submission.state is OrderState.FILLED

    if settings.trading_mode == "PAPER" and paper_portfolio:
        amount_to_sell = paper_portfolio.get_free_balance(
            ticker.split("/")[0],
        )
    else:
        amount_to_sell = exchange.get_free_balance(ticker.split("/")[0])

    if amount_to_sell < 0.0001:  # noqa: PLR2004
        logger.info("Sell skipped for %s: no significant balance.", ticker)
        return False

    usd_value = price * amount_to_sell
    if usd_value < MIN_ORDER_USD:
        logger.info(
            "Dust position detected for %s: %.8f (worth $%.2f < $%.0f min). Cleaning up.",
            ticker,
            amount_to_sell,
            usd_value,
            MIN_ORDER_USD,
        )
        journal.delete_position_state(ticker)
        return False

    profit = 0.0
    tax = 0.0
    if buy_price:
        estimated_sell_fee = price * amount_to_sell * 0.0025
        profit, tax = journal.compute_pnl_with_fees(
            ticker,
            price,
            amount_to_sell,
            estimated_sell_fee,
        )
        if profit > 0:
            logger.info(
                "Tax: Realized profit of %.2f %s (fee-adjusted). Tax liability: %.2f",
                profit,
                settings.quote_currency,
                tax,
            )

    if settings.trading_mode == "LIVE":
        logger.info(">>> EXECUTING LIVE SELL for %s <<<", ticker)
        try:
            exchange.cancel_all_orders(ticker)
            exchange.place_maker_sell(ticker, price, amount_to_sell)
        except Exception:  # noqa: BLE001
            logger.warning("Sell order failed for %s.", ticker, exc_info=True)
            return False
    else:
        logger.info(">>> PAPER: SELL %s @ $%s <<<", ticker, price)
        if paper_portfolio:
            paper_portfolio.record_sell(ticker, amount_to_sell)

    journal.log_trade(
        account="Kraken",
        instrument=ticker,
        side="SELL",
        price=price,
        quantity=amount_to_sell,
        value=price * amount_to_sell,
        currency=settings.quote_currency,
        reasoning=reason,
        profit=profit,
        tax=tax,
        fee=price * amount_to_sell * 0.0025,
    )
    notifier.notify_trade(ticker, "SELL", price, amount_to_sell, reason)
    return True


def execute_partial_sell(  # noqa: PLR0913
    ticker: str,
    price: float,
    sell_pct: float,
    reasoning: str,
    settings: Settings,
    exchange: CachedExchange,
    journal: TradeJournal,
    notifier: WebhookNotifier | NullNotifier,
    paper_portfolio: PaperPortfolio | None,
    *,
    coordinator: ExecutionCoordinator | None = None,
) -> bool:
    """Execute a partial position sell. Returns True if the order succeeded."""
    if coordinator is not None:
        logger.warning(
            "Partial exit disabled for %s: exact protection resizing is unavailable.",
            ticker,
        )
        return False
    base_currency = ticker.split("/")[0]

    if settings.trading_mode == "PAPER" and paper_portfolio:
        total_balance = paper_portfolio.get_free_balance(base_currency)
    else:
        total_balance = exchange.get_free_balance(base_currency)

    sell_amount = total_balance * sell_pct

    if sell_amount < 0.0001:  # noqa: PLR2004
        return False

    sell_price = price * (1 + settings.limit_order_discount_pct)

    if settings.trading_mode == "LIVE":
        logger.info(">>> EXECUTING LIVE PARTIAL SELL %s: %.1f%% <<<", ticker, sell_pct * 100)
        try:
            exchange.place_maker_sell(ticker, sell_price, sell_amount)
        except Exception:  # noqa: BLE001
            logger.warning("Partial sell failed for %s", ticker, exc_info=True)
            return False
    else:
        logger.info(
            ">>> PAPER PARTIAL SELL %s: %.1f%% @ $%s <<<",
            ticker,
            sell_pct * 100,
            sell_price,
        )
        if paper_portfolio:
            paper_portfolio.record_sell(ticker, sell_amount)

    fee = sell_price * sell_amount * 0.0025
    profit, tax = journal.compute_pnl_with_fees(ticker, sell_price, sell_amount, fee)

    journal.log_trade(
        account="Kraken",
        instrument=ticker,
        side="SELL",
        price=sell_price,
        quantity=sell_amount,
        value=sell_price * sell_amount,
        currency=settings.quote_currency,
        reasoning=reasoning,
        profit=profit,
        tax=tax,
        fee=fee,
    )
    notifier.notify_trade(ticker, "PARTIAL SELL", sell_price, sell_amount, reasoning)
    return True


def check_take_profit(  # noqa: PLR0913
    ticker: str,
    data: MarketData,
    settings: Settings,
    exchange: CachedExchange,
    journal: TradeJournal,
    notifier: WebhookNotifier | NullNotifier,
    paper_portfolio: PaperPortfolio | None,
    *,
    coordinator: ExecutionCoordinator | None = None,
    execution_store: ExecutionStore | None = None,
    runtime: RuntimeState | None = None,
    cycle_id: str | None = None,
) -> bool:
    """Check hard take-profit and execute sell if hit. Returns True if sold."""
    buy_price = data.entry_price
    if not buy_price:
        return False

    _stop_loss_price, take_profit_price = compute_protection_prices(
        buy_price,
        settings,
        data,
    )

    if data.current_price >= take_profit_price:
        logger.info(
            "Take-profit hit for %s: price $%.8f >= TP $%.8f",
            ticker,
            data.current_price,
            take_profit_price,
        )
        sold = execute_sell(
            ticker,
            data.current_price,
            buy_price,
            f"Take-profit triggered: price ${data.current_price:.8f} "
            f">= TP ${take_profit_price:.8f}",
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
        return sold

    return False


def ensure_stop_loss(  # noqa: C901, PLR0911, PLR0912, PLR0913
    ticker: str,
    data: MarketData,
    settings: Settings,
    exchange: CachedExchange,
    *,
    coordinator: ExecutionCoordinator | None = None,
    execution_store: ExecutionStore | None = None,
    runtime: RuntimeState | None = None,
    desired_stop_price: float | None = None,
) -> None:
    """Place a stop-loss order if none exists (LIVE mode only)."""
    buy_price = data.entry_price
    if not buy_price:
        return

    if coordinator is not None:
        if execution_store is None:
            msg = "coordinator stop placement requires execution_store"
            raise ValueError(msg)
        position = execution_store.get_position(ticker)
        if position is None:
            return
        stop_loss_price, _take_profit_price = compute_protection_prices(
            buy_price,
            settings,
            data,
        )
        if desired_stop_price is not None:
            stop_loss_price = max(stop_loss_price, desired_stop_price)
        if stop_loss_price >= data.current_price:
            logger.warning(
                "Protective stop rejected for %s: %.8f is not below current price %.8f.",
                ticker,
                stop_loss_price,
                data.current_price,
            )
            return
        active_protections = execution_store.list_protection(ticker, active_only=True)
        covered_quantity = sum(
            (Decimal(str(item["quantity"])) for item in active_protections),
            start=Decimal(0),
        )
        if active_protections and covered_quantity == position.quantity:
            verifications = [
                coordinator.verify_protective_stop(str(item["protection_id"]))
                for item in active_protections
            ]
            if all(item.valid for item in verifications):
                current_triggers = [
                    Decimal(str(item["trigger_price"])) for item in active_protections
                ]
                target = Decimal(str(stop_loss_price))
                if any(trigger < target for trigger in current_triggers):
                    guard = (
                        runtime.mutation_section()
                        if runtime is not None
                        else contextlib.nullcontext()
                    )
                    with guard:
                        ratcheted = coordinator.ratchet_protective_stops(
                            ticker,
                            target,
                            Decimal(str(data.current_price)),
                        )
                    if not ratcheted:
                        logger.warning("Protective stop ratchet was not confirmed for %s.", ticker)
                return
        guard = runtime.mutation_section() if runtime is not None else contextlib.nullcontext()
        try:
            with guard:
                coordinator.ensure_protective_stop(
                    ticker,
                    position.quantity,
                    Decimal(str(stop_loss_price)),
                    position.average_cost,
                    Decimal(str(data.current_price)),
                )
        except Exception:  # noqa: BLE001
            logger.warning("Failed to ensure exact protective stop for %s.", ticker, exc_info=True)
        if runtime is not None and not coordinator.ready:
            runtime.mark_not_ready()
        return

    if settings.trading_mode != "LIVE":
        return

    open_orders = exchange.get_open_orders(ticker)
    has_stop_loss = any(o.get("type") in ("stop-loss", "stop-loss-limit") for o in open_orders)
    if has_stop_loss:
        return

    stop_loss_price, take_profit_price = compute_protection_prices(
        buy_price,
        settings,
        data,
    )

    base_currency = ticker.split("/")[0]
    balance = exchange.get_free_balance(base_currency)
    if balance > 0.0001:  # noqa: PLR2004
        logger.info(
            "Placing deferred SL for %s: %.8f @ %.8f",
            ticker,
            balance,
            stop_loss_price,
        )
        try:
            exchange.place_protection_orders(
                ticker,
                balance,
                buy_price,
                stop_loss_price,
                take_profit_price,
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "Failed to place deferred SL for %s.",
                ticker,
            )


def execute_dca(  # noqa: PLR0913
    ticker: str,
    data: MarketData,
    trade_signal: TradeSignal,
    settings: Settings,
    exchange: CachedExchange,
    guardian: RiskGuardian,
    journal: TradeJournal,
    notifier: WebhookNotifier | NullNotifier,
    available_slots: int,
    paper_portfolio: PaperPortfolio | None,
    *,
    coordinator: ExecutionCoordinator | None = None,
) -> None:
    """Execute a DCA buy if the strategy signals BUY on an owned position."""
    if coordinator is not None:
        logger.info("DCA disabled for %s: exact protection resizing is unavailable.", ticker)
        return
    if trade_signal.action != "BUY" or trade_signal.confidence < settings.min_buy_confidence:
        return

    active_buys = journal.get_trade_count(ticker, "BUY")
    max_allowed = 1 + settings.max_dca_per_position
    if active_buys >= max_allowed:
        logger.info(
            "DCA limit reached for %s: %d active buys (max %d).",
            ticker,
            active_buys,
            max_allowed,
        )
        return

    open_orders = exchange.get_open_orders(ticker)
    if open_orders:
        logger.info(
            "DCA: Skipping %s - %d open orders remain.",
            ticker,
            len(open_orders),
        )
        return

    price = data.current_price
    limit_price = price * (1 - settings.limit_order_discount_pct)
    amount = guardian.check_buy(
        ticker,
        limit_price,
        trade_signal,
        available_slots,
        is_dca=True,
    )
    if amount <= 0:
        return

    if settings.trading_mode == "LIVE":
        logger.info(">>> EXECUTING LIVE DCA BUY for %s <<<", ticker)
        exchange.place_maker_buy(ticker, limit_price, amount)

        stop_loss_price, take_profit_price = compute_protection_prices(
            limit_price,
            settings,
            data,
        )
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
                "Protection orders deferred for %s DCA.",
                ticker,
            )
    else:
        logger.info(
            ">>> PAPER DCA: BUY %s @ $%s <<<",
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
        reasoning=f"DCA: {trade_signal.reasoning}",
        fee=limit_price * amount * 0.0025,
    )
    notifier.notify_trade(
        ticker,
        "DCA BUY",
        limit_price,
        amount,
        trade_signal.reasoning,
    )

    avg_cost = journal.get_average_cost_basis(ticker)
    if avg_cost and settings.trading_mode == "LIVE":
        new_stop_loss, _new_tp = compute_protection_prices(
            avg_cost,
            settings,
            data,
        )
        logger.info(
            "DCA SL recalc for %s: avg cost $%.8f -> SL $%.8f",
            ticker,
            avg_cost,
            new_stop_loss,
        )

    dca_stop_loss_price, dca_take_profit_price = compute_protection_prices(
        limit_price,
        settings,
        data,
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

    entry_for_state = avg_cost if avg_cost else limit_price

    existing_state = journal.get_position_state(ticker)
    pos_state = PositionState(
        ticker=ticker,
        entry_price=entry_for_state,
        high_watermark=max(
            entry_for_state,
            existing_state.high_watermark if existing_state else 0.0,
        ),
        stop_mode=stop_mode,
        current_sl=dca_stop_loss_price,
        current_tp=dca_take_profit_price,
        partial_exits=existing_state.partial_exits if existing_state else 0,
        regime_at_entry=regime.value if regime else "RANGING",
    )
    journal.upsert_position_state(pos_state)


def reconcile_stale_orders(
    settings: Settings,
    exchange: KrakenExchange,
    journal: TradeJournal,
) -> None:
    """Cancel stale orders and log cancellations to the journal."""
    for ticker in settings.universe_list:
        if not exchange.symbol_exists(ticker):
            continue
        try:
            cancelled = exchange.cancel_stale_orders(
                ticker,
                settings.stale_order_max_age_seconds,
            )
            if cancelled:
                logger.info(
                    "Reconciliation: Cancelled %d stale orders for %s",
                    len(cancelled),
                    ticker,
                )
                for order in cancelled:
                    if order["side"] == "buy":
                        journal.log_trade(
                            account="Kraken",
                            instrument=ticker,
                            side="CANCEL",
                            price=order["price"],
                            quantity=order["amount"],
                            value=order["price"] * order["amount"],
                            currency=settings.quote_currency,
                            reasoning="Stale limit order cancelled.",
                        )
                        journal.delete_position_state(ticker)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Reconciliation: Error for %s: %s",
                ticker,
                exc,
            )


def _process_single_fill(  # noqa: PLR0913
    fill: dict,
    ticker: str,
    journal: TradeJournal,
    notifier: WebhookNotifier | NullNotifier,
    quote_currency: str,
    tax_rate: float,
) -> None:
    """Process a single Kraken fill and record it if not already tracked."""
    fill_id = fill.get("id", "")
    if not fill_id or journal.is_fill_recorded(fill_id):
        return

    side = fill.get("side", "").upper()
    fill_price = float(fill.get("price", 0))
    fill_qty = float(fill.get("amount", 0))
    fill_cost = float(fill.get("cost", 0))
    fee_info = fill.get("fee", {})
    fill_fee = float(fee_info.get("cost", 0)) if fee_info else 0.0

    if side not in ("BUY", "SELL") or fill_price <= 0:
        return

    # v9.7: partial-fill residues (cents) are accounting noise, not trades.
    if fill_cost < DUST_FILL_USD:
        logger.info(
            "Reconciliation: Skipping dust fill for %s ($%.4f) [%s]",
            ticker,
            fill_cost,
            fill_id,
        )
        return

    # v9.7: window widened 15 -> 180 min. Maker orders live up to 2h before
    # stale-order cleanup; fills landing after 15 min were re-journaled on
    # top of the placement row, double-counting the position.
    if journal.has_recent_trade(
        ticker,
        side,
        fill_qty,
        window_minutes=FILL_DEDUP_WINDOW_MINUTES,
        fill_source="bot",
    ):
        return

    profit = 0.0
    tax = 0.0
    if side == "SELL":
        profit, tax = journal.compute_pnl_with_fees(
            ticker,
            fill_price,
            fill_qty,
            fill_fee,
            tax_rate=tax_rate,
        )

    reasoning = (
        f"Exchange fill: {side} detected via reconciliation. "
        f"Price={fill_price}, Fee={fill_fee:.6f} [{fill_id}]"
    )

    journal.log_trade(
        account="Kraken",
        instrument=ticker,
        side=side,
        price=fill_price,
        quantity=fill_qty,
        value=fill_cost,
        currency=quote_currency,
        reasoning=reasoning,
        profit=profit,
        tax=tax,
        fee=fill_fee,
        fill_source="exchange",
        exchange_fill_id=fill_id,
    )

    logger.info(
        "Reconciliation: Recorded %s fill for %s: %.8f @ %s (fee: %.6f, P&L: %.4f) [%s]",
        side,
        ticker,
        fill_qty,
        fmt_price(fill_price),
        fill_fee,
        profit,
        fill_id,
    )

    if side == "SELL":
        journal.delete_position_state(ticker)
        notifier.notify_trade(
            ticker,
            "SELL",
            fill_price,
            fill_qty,
            f"Exchange SL/TP fill detected [{fill_id}]",
        )


def reconcile_fills(  # noqa: PLR0913
    tickers: list[str],
    exchange: KrakenExchange,
    journal: TradeJournal,
    notifier: WebhookNotifier | NullNotifier,
    quote_currency: str,
    tax_rate: float,
) -> None:
    """Sync Kraken fills with our journal to catch SL/TP executions."""
    for ticker in tickers:
        try:
            since_ms = journal.get_last_sync_timestamp(ticker)
            if since_ms is not None:
                since_ms = max(since_ms - 60_000, 0)

            fills = exchange.fetch_my_trades(ticker, since_ms=since_ms)
            for fill in fills:
                _process_single_fill(
                    fill,
                    ticker,
                    journal,
                    notifier,
                    quote_currency,
                    tax_rate,
                )
        except Exception as exc:  # noqa: BLE001, PERF203
            logger.warning("Fill reconciliation error for %s: %s", ticker, exc)
