"""Durable exact-identity coordination between an exchange and ExecutionStore."""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Iterable, Mapping
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Any, TypeGuard

import ccxt  # type: ignore[import-untyped]

from exchange import KRAKEN_FILL_PAGE_SIZE, MAX_PAGINATION_PAGES, KrakenExchange
from execution_store import (
    ExecutionStore,
    FillResult,
    OrderPurpose,
    OrderState,
    Side,
    StoredOrder,
    canonical_decimal,
    deterministic_client_id,
)

_FILLS_CURSOR = "fills:account"
_DEFAULT_FILL_WINDOW_GRANULARITY_MS = 1_000


class SubmissionOutcome(StrEnum):
    """Stable classifications returned by mutation coordination."""

    BLOCKED = "BLOCKED"
    SUBMITTED = "SUBMITTED"
    ALREADY_KNOWN = "ALREADY_KNOWN"
    REJECTED = "REJECTED"
    UNKNOWN = "UNKNOWN"


@dataclasses.dataclass(frozen=True)
class SubmissionResult:
    """Durable result of one submission attempt."""

    intent_id: str
    order_id: int | None
    client_order_id: str
    exchange_order_id: str | None
    outcome: SubmissionOutcome
    state: OrderState | None
    error: str | None = None


@dataclasses.dataclass(frozen=True)
class ReconciledFill:
    """Result of applying one exact exchange fill."""

    fill_id: str
    exchange_order_id: str
    order_id: int | None
    inserted: bool
    state: OrderState | None


@dataclasses.dataclass(frozen=True)
class FillReconciliationResult:
    """Bounded fixed-window fill reconciliation outcome."""

    since_ms: int
    end_ms: int
    fills: tuple[ReconciledFill, ...]
    unresolved_fill_ids: tuple[str, ...]
    complete: bool
    error: str | None = None


@dataclasses.dataclass(frozen=True)
class RecoveryResult:
    """Result of an exact order recovery lookup."""

    order_id: int
    client_order_id: str
    exchange_order_id: str | None
    state: OrderState
    recovered: bool


@dataclasses.dataclass(frozen=True)
class CancellationResult:
    """Result of cancelling one known stale entry order."""

    order_id: int
    exchange_order_id: str
    state: OrderState
    canceled: bool
    error: str | None = None


@dataclasses.dataclass(frozen=True)
class BalanceComparison:
    """Exact store-versus-exchange inventory comparison."""

    symbol: str
    currency: str
    expected: Decimal
    actual: Decimal
    difference: Decimal
    matches: bool


@dataclasses.dataclass(frozen=True)
class ProtectionVerification:
    """Exact quantity and trigger check for one known protective order."""

    protection_id: str
    exchange_order_id: str | None
    expected_quantity: Decimal
    actual_quantity: Decimal | None
    expected_trigger_price: Decimal
    actual_trigger_price: Decimal | None
    valid: bool
    error: str | None = None


@dataclasses.dataclass(frozen=True)
class ProtectionSubmissionResult:
    """Durable result of ensuring one exact protective stop."""

    protection_id: str
    exchange_order_id: str | None
    outcome: SubmissionOutcome
    state: OrderState
    verification: ProtectionVerification | None = None
    error: str | None = None


@dataclasses.dataclass(frozen=True)
class RefreshedOrder:
    """One exact local order refreshed from exchange state."""

    order_id: int
    exchange_order_id: str
    state: OrderState


@dataclasses.dataclass(frozen=True)
class OpenOrderReconciliationResult:
    """Exact open-order refresh and conditional-child adoption result."""

    refreshed: tuple[RefreshedOrder, ...]
    adopted_protection_ids: tuple[str, ...]
    unknown_exchange_order_ids: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class ProtectedExitResult:
    """Protective handoff result for one emergency market exit."""

    submission: SubmissionResult | None
    canceled_protection_ids: tuple[str, ...]
    restored: tuple[ProtectionSubmissionResult, ...]
    ready: bool
    error: str | None = None


@dataclasses.dataclass(frozen=True)
class RecoverySummary:
    """Structured result of one full exact-data recovery pass."""

    recovered_orders: tuple[RecoveryResult, ...]
    open_orders: OpenOrderReconciliationResult
    fills: FillReconciliationResult
    balances: tuple[BalanceComparison, ...]
    protections: tuple[ProtectionVerification, ...]
    readiness: ReadinessResult


@dataclasses.dataclass(frozen=True)
class ReadinessResult:
    """Readiness derived solely from unresolved reconciliation issues."""

    ready: bool
    issues: tuple[dict[str, Any], ...]


class ExecutionCoordinator:
    """Coordinate persisted intent, one mutation, and exact-data recovery."""

    def __init__(
        self,
        store: ExecutionStore,
        exchange: KrakenExchange,
        *,
        live_entries_enabled: bool = False,
        fill_page_size: int = KRAKEN_FILL_PAGE_SIZE,
        max_fill_pages: int = 20,
    ) -> None:
        """Bind one scoped store to one exchange account and an external entry gate."""
        if not isinstance(live_entries_enabled, bool):
            msg = "live_entries_enabled must be a bool"
            raise TypeError(msg)
        if isinstance(fill_page_size, bool) or fill_page_size != KRAKEN_FILL_PAGE_SIZE:
            msg = f"fill_page_size must be exactly {KRAKEN_FILL_PAGE_SIZE} for Kraken"
            raise ValueError(msg)
        if isinstance(max_fill_pages, bool) or not 1 <= max_fill_pages <= MAX_PAGINATION_PAGES:
            msg = f"max_fill_pages must be between 1 and {MAX_PAGINATION_PAGES}"
            raise ValueError(msg)
        self.store = store
        self.exchange = exchange
        self.live_entries_enabled = live_entries_enabled
        self.fill_page_size = fill_page_size
        self.max_fill_pages = max_fill_pages

    def submit_entry(  # noqa: PLR0913
        self,
        intent_id: str,
        symbol: str,
        quantity: Decimal,
        limit_price: Decimal,
        *,
        stop_loss_price: Decimal | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> SubmissionResult:
        """Submit a gated maker entry, optionally with an atomic conditional stop."""
        return self._submit_buy(
            intent_id,
            symbol,
            quantity,
            limit_price,
            purpose=OrderPurpose.ENTRY,
            stop_loss_price=stop_loss_price,
            payload=payload,
        )

    def submit_dca(  # noqa: PLR0913
        self,
        intent_id: str,
        symbol: str,
        quantity: Decimal,
        limit_price: Decimal,
        *,
        stop_loss_price: Decimal | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> SubmissionResult:
        """Submit a gated maker DCA order with the same exact-order protocol."""
        return self._submit_buy(
            intent_id,
            symbol,
            quantity,
            limit_price,
            purpose=OrderPurpose.DCA,
            stop_loss_price=stop_loss_price,
            payload=payload,
        )

    def submit_exit(  # noqa: PLR0913
        self,
        intent_id: str,
        symbol: str,
        quantity: Decimal,
        limit_price: Decimal | None = None,
        *,
        emergency: bool = False,
        payload: Mapping[str, Any] | None = None,
    ) -> SubmissionResult:
        """Submit an ungated maker exit or a controlled emergency market sell."""
        quantity = _required_decimal("quantity", quantity, positive=True)
        if not emergency and limit_price is None:
            msg = "limit_price is required for a non-emergency exit"
            raise ValueError(msg)
        normalized_price = (
            None
            if limit_price is None
            else _required_decimal("limit_price", limit_price, positive=True)
        )
        return self._submit(
            intent_id,
            symbol,
            Side.SELL,
            OrderPurpose.EXIT,
            quantity,
            normalized_price,
            emergency=emergency,
            stop_loss_price=None,
            payload=payload,
        )

    def _submit_buy(  # noqa: PLR0913
        self,
        intent_id: str,
        symbol: str,
        quantity: Decimal,
        limit_price: Decimal,
        *,
        purpose: OrderPurpose,
        stop_loss_price: Decimal | None,
        payload: Mapping[str, Any] | None,
    ) -> SubmissionResult:
        quantity = _required_decimal("quantity", quantity, positive=True)
        limit_price = _required_decimal("limit_price", limit_price, positive=True)
        normalized_stop = (
            None
            if stop_loss_price is None
            else _required_decimal("stop_loss_price", stop_loss_price, positive=True)
        )
        if not self.live_entries_enabled:
            return SubmissionResult(
                intent_id=intent_id,
                order_id=None,
                client_order_id=deterministic_client_id(self.store.scope, intent_id),
                exchange_order_id=None,
                outcome=SubmissionOutcome.BLOCKED,
                state=None,
                error="live entries are disabled",
            )
        return self._submit(
            intent_id,
            symbol,
            Side.BUY,
            purpose,
            quantity,
            limit_price,
            emergency=False,
            stop_loss_price=normalized_stop,
            payload=payload,
        )

    def _submit(  # noqa: PLR0913
        self,
        intent_id: str,
        symbol: str,
        side: Side,
        purpose: OrderPurpose,
        quantity: Decimal,
        limit_price: Decimal | None,
        *,
        emergency: bool,
        stop_loss_price: Decimal | None,
        payload: Mapping[str, Any] | None,
    ) -> SubmissionResult:
        order_type = "MARKET" if emergency else "LIMIT"
        if emergency:
            quantity, limit_price = self.exchange.normalize_market_sell(
                symbol,
                quantity,
                reference_price=limit_price,
            )
        else:
            if limit_price is None:
                msg = "limit_price is required for maker orders"
                raise ValueError(msg)
            quantity, limit_price, stop_loss_price = self.exchange.normalize_limit_order(
                symbol,
                quantity,
                limit_price,
                stop_loss_price=stop_loss_price,
            )
        intent_payload: dict[str, Any] = {"context": dict(payload or {})}
        if stop_loss_price is not None:
            intent_payload["conditional_stop_price"] = stop_loss_price
        client_order_id = self.store.create_order_intent(
            intent_id,
            symbol,
            side,
            quantity,
            purpose=purpose,
            order_type=order_type,
            limit_price=limit_price,
            payload=intent_payload,
        )
        order = self.store.get_order_by_client_id(client_order_id)
        if order is None:
            order = self.store.record_order(intent_id)
        elif order.state is not OrderState.CREATED:
            return self._known_submission(order)

        order = self.store.transition_order(
            order.order_id,
            OrderState.SUBMITTING,
            expected_state=OrderState.CREATED,
        )
        try:
            response = self._place_order(
                order,
                symbol,
                side,
                quantity,
                limit_price,
                emergency=emergency,
                stop_loss_price=stop_loss_price,
            )
        except Exception as error:  # noqa: BLE001
            return self._submission_error(order, error)
        if not isinstance(response, Mapping):
            return self._unknown_response(order, "submission response is not a mapping")
        return self._submission_response(order, response)

    def _place_order(  # noqa: PLR0913
        self,
        order: StoredOrder,
        symbol: str,
        side: Side,
        quantity: Decimal,
        limit_price: Decimal | None,
        *,
        emergency: bool,
        stop_loss_price: Decimal | None,
    ) -> dict[str, Any]:
        if emergency:
            return self.exchange.place_market_sell(
                symbol,
                quantity,
                client_order_id=order.client_order_id,
                reference_price=limit_price,
            )
        if limit_price is None:
            msg = "limit_price is required for maker orders"
            raise ValueError(msg)
        if side is Side.BUY:
            return self.exchange.place_maker_buy(
                symbol,
                limit_price,
                quantity,
                client_order_id=order.client_order_id,
                stop_loss_price=stop_loss_price,
            )
        return self.exchange.place_maker_sell(
            symbol,
            limit_price,
            quantity,
            client_order_id=order.client_order_id,
        )

    def _submission_error(self, order: StoredOrder, error: Exception) -> SubmissionResult:
        if _is_ambiguous_mutation_error(error):
            state = OrderState.UNKNOWN
            outcome = SubmissionOutcome.UNKNOWN
            self._record_unknown_submission(order, error)
        elif isinstance(error, (ccxt.ExchangeError, ValueError)):
            state = OrderState.REJECTED
            outcome = SubmissionOutcome.REJECTED
        else:
            state = OrderState.UNKNOWN
            outcome = SubmissionOutcome.UNKNOWN
            self._record_unknown_submission(order, error)
        updated = self.store.transition_order(
            order.order_id,
            state,
            expected_state=OrderState.SUBMITTING,
            payload={"error_type": type(error).__name__, "message": str(error)},
        )
        return self._submission_result(updated, outcome, error=str(error))

    def _submission_response(
        self,
        order: StoredOrder,
        response: Mapping[str, Any],
    ) -> SubmissionResult:
        status = str(response.get("status") or "open").lower()
        exchange_order_id = response.get("id")
        if status in {"rejected", "expired"}:
            if isinstance(exchange_order_id, str) and exchange_order_id:
                order = self.store.bind_exchange_order_id(order.order_id, exchange_order_id)
            rejected = self.store.transition_order(order.order_id, OrderState.REJECTED)
            return self._submission_result(rejected, SubmissionOutcome.REJECTED)
        if not isinstance(exchange_order_id, str) or not exchange_order_id:
            return self._unknown_response(order, "submission response has no exact order id")
        response_client_id = _order_client_id(response)
        if response_client_id is not None and response_client_id != order.client_order_id:
            return self._unknown_response(order, "submission response clientOrderId mismatch")

        order = self.store.bind_exchange_order_id(order.order_id, exchange_order_id)
        if status in {"canceled", "cancelled"}:
            canceled = self.store.transition_order(order.order_id, OrderState.CANCELED)
            return self._submission_result(canceled, SubmissionOutcome.REJECTED)
        if status in {"closed", "filled"}:
            return self._unknown_response(order, "closed submission requires exact fill recovery")
        opened = self.store.transition_order(order.order_id, OrderState.OPEN)
        return self._submission_result(opened, SubmissionOutcome.SUBMITTED)

    def _unknown_response(self, order: StoredOrder, message: str) -> SubmissionResult:
        self.store.record_reconciliation_issue(
            "UNKNOWN_SUBMISSION",
            order.client_order_id,
            {"message": message, "order_id": order.order_id},
        )
        unknown = self.store.transition_order(order.order_id, OrderState.UNKNOWN)
        return self._submission_result(unknown, SubmissionOutcome.UNKNOWN, error=message)

    def _known_submission(self, order: StoredOrder) -> SubmissionResult:
        if order.state is OrderState.SUBMITTING:
            message = "prior submission stopped while awaiting an exchange response"
            self.store.record_reconciliation_issue(
                "UNKNOWN_SUBMISSION",
                order.client_order_id,
                {"message": message, "order_id": order.order_id},
            )
            order = self.store.transition_order(order.order_id, OrderState.UNKNOWN)
        outcome = (
            SubmissionOutcome.UNKNOWN
            if order.state is OrderState.UNKNOWN
            else SubmissionOutcome.ALREADY_KNOWN
        )
        return self._submission_result(order, outcome)

    @staticmethod
    def _submission_result(
        order: StoredOrder,
        outcome: SubmissionOutcome,
        *,
        error: str | None = None,
    ) -> SubmissionResult:
        return SubmissionResult(
            intent_id=order.intent_id,
            order_id=order.order_id,
            client_order_id=order.client_order_id,
            exchange_order_id=order.exchange_order_id,
            outcome=outcome,
            state=order.state,
            error=error,
        )

    def _record_unknown_submission(self, order: StoredOrder, error: Exception) -> None:
        self.store.record_reconciliation_issue(
            "UNKNOWN_SUBMISSION",
            order.client_order_id,
            {
                "error_type": type(error).__name__,
                "message": str(error),
                "order_id": order.order_id,
            },
        )

    def reconcile_fills(
        self,
        *,
        since_ms: int | None = None,
        end_ms: int | None = None,
    ) -> FillReconciliationResult:
        """Apply a bounded account-wide fill snapshot in exact chronological order."""
        if (
            since_ms is None
            and self.store.scope.startswith("live:")
            and self.store.get_sync_cursor(_FILLS_CURSOR) is None
        ):
            # A live scope with no cursor would import the whole account
            # history from epoch 0 — pre-v10 fills would either exceed the
            # pagination bound every pass or flood position accounting with
            # unresolvable unknown-order fills. Refuse until the cutover
            # rebuild (src/rebuild_v10.py) seeds the cursor explicitly.
            message = (
                "live scope has no fill cursor; run the rebuild_v10 cutover "
                "to seed it before reconciliation"
            )
            self.store.record_reconciliation_issue(
                "FILL_RECONCILIATION_FAILED",
                _FILLS_CURSOR,
                {"message": message},
            )
            now_ms = int(time.time() * 1000) if end_ms is None else end_ms
            return FillReconciliationResult(
                since_ms=0,
                end_ms=now_ms,
                fills=(),
                unresolved_fill_ids=(),
                complete=False,
                error=message,
            )
        start = self._fill_start(since_ms)
        requested_end = int(time.time() * 1000) if end_ms is None else _timestamp("end_ms", end_ms)
        granularity_ms = self._fill_window_granularity_ms()
        end = (
            requested_end
            if requested_end < granularity_ms
            else requested_end // granularity_ms * granularity_ms
        )
        if end < start:
            msg = "end_ms must be greater than or equal to since_ms"
            raise ValueError(msg)
        try:
            exchange_fills = self.exchange.fetch_account_fills(
                start,
                end,
                page_size=self.fill_page_size,
                max_pages=self.max_fill_pages,
            )
        except Exception as error:  # noqa: BLE001
            self.store.record_reconciliation_issue(
                "FILL_RECONCILIATION_FAILED",
                _FILLS_CURSOR,
                {
                    "end_ms": end,
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "since_ms": start,
                },
            )
            return FillReconciliationResult(
                since_ms=start,
                end_ms=end,
                fills=(),
                unresolved_fill_ids=(),
                complete=False,
                error=str(error),
            )

        results: list[ReconciledFill] = []
        unresolved: list[str] = []
        for fill in sorted(exchange_fills, key=_fill_sort_key):
            result = self._apply_exchange_fill(fill)
            if isinstance(result, ReconciledFill):
                results.append(result)
            else:
                unresolved.append(result)
        complete = not unresolved
        if complete:
            self.store.set_sync_cursor(_FILLS_CURSOR, str(end))
            self._resolve_issue("FILL_RECONCILIATION_FAILED", _FILLS_CURSOR)
        return FillReconciliationResult(
            start,
            end,
            tuple(results),
            tuple(unresolved),
            complete,
        )

    def _fill_start(self, since_ms: int | None) -> int:
        if since_ms is not None:
            return _timestamp("since_ms", since_ms)
        cursor = self.store.get_sync_cursor(_FILLS_CURSOR)
        if cursor is None:
            return 0
        try:
            stored = _timestamp("stored fill cursor", int(cursor))
            return max(stored - self._fill_window_granularity_ms(), 0)
        except ValueError as error:
            self.store.record_reconciliation_issue(
                "INVALID_FILL_CURSOR",
                _FILLS_CURSOR,
                {"cursor": cursor},
            )
            msg = "stored fill cursor is not an integer"
            raise ValueError(msg) from error

    def _fill_window_granularity_ms(self) -> int:
        granularity = getattr(
            self.exchange,
            "fill_window_granularity_ms",
            _DEFAULT_FILL_WINDOW_GRANULARITY_MS,
        )
        if isinstance(granularity, bool) or not isinstance(granularity, int) or granularity <= 0:
            msg = "exchange fill_window_granularity_ms must be a positive integer"
            raise ValueError(msg)
        return granularity

    def _apply_exchange_fill(self, fill: Mapping[str, Any]) -> ReconciledFill | str:  # noqa: C901
        fill_id = fill.get("id")
        exchange_order_id = fill.get("order")
        if not _exact_id(fill_id) or not _exact_id(exchange_order_id):
            issue_key = str(fill_id or exchange_order_id or "missing-identity")
            self.store.record_reconciliation_issue(
                "INVALID_FILL",
                issue_key,
                {"message": "fill and order IDs are required"},
            )
            return issue_key
        order = self.store.get_order_by_exchange_id(exchange_order_id)
        protection = self.store.get_protection_by_exchange_id(exchange_order_id)
        if order is None and protection is None:
            protection = self._adopt_closed_conditional_stop(fill, exchange_order_id)
        if order is None and protection is None:
            try:
                applied = self._store_fill(fill, fill_id, exchange_order_id, None)
            except Exception as error:  # noqa: BLE001
                self.store.record_reconciliation_issue(
                    "INVALID_FILL",
                    fill_id,
                    {"error_type": type(error).__name__, "message": str(error)},
                )
                return fill_id
            self.store.record_reconciliation_issue(
                "HALT_UNKNOWN_ORDER_FILL",
                fill_id,
                {"exchange_order_id": exchange_order_id, "fill_id": fill_id},
            )
            return ReconciledFill(
                fill_id=fill_id,
                exchange_order_id=exchange_order_id,
                order_id=None,
                inserted=applied.inserted,
                state=None,
            )
        if protection is not None:
            fill_symbol = fill.get("symbol")
            fill_side = str(fill.get("side") or "").upper()
            if fill_symbol != protection["symbol"] or fill_side != Side.SELL.value:
                self.store.record_reconciliation_issue(
                    "INVALID_FILL",
                    fill_id,
                    {"message": "fill does not match exact protective order symbol/side"},
                )
                return fill_id
        try:
            applied = self._store_fill(fill, fill_id, exchange_order_id, order)
        except Exception as error:  # noqa: BLE001
            self.store.record_reconciliation_issue(
                "INVALID_FILL",
                fill_id,
                {"error_type": type(error).__name__, "message": str(error)},
            )
            return fill_id
        state = None if order is None else self.store.get_order(order.order_id).state
        self._resolve_issue("INVALID_FILL", fill_id)
        if order is not None and state is not OrderState.UNKNOWN:
            self._resolve_issue("UNKNOWN_SUBMISSION", order.client_order_id)
            self._resolve_issue("CLOSED_ORDER_PENDING_FILLS", exchange_order_id)
            self._resolve_issue("MISSING_KNOWN_ORDER", exchange_order_id)
            if state is OrderState.FILLED:
                self._resolve_issue("EXIT_PENDING_FILL", order.client_order_id)
        if protection is not None:
            self._refresh_protection_after_fill(protection)
        return ReconciledFill(
            fill_id=fill_id,
            exchange_order_id=exchange_order_id,
            order_id=None if order is None else order.order_id,
            inserted=applied.inserted,
            state=state,
        )

    def _refresh_protection_after_fill(self, protection: Mapping[str, Any]) -> None:
        protection_id = str(protection["protection_id"])
        exchange_order_id = str(protection["exchange_order_id"])
        symbol = str(protection["symbol"])
        try:
            exchange_order = self.exchange.fetch_order(exchange_order_id, symbol)
        except Exception:  # noqa: BLE001
            return
        status = str(exchange_order.get("status") or "").lower()
        current = OrderState(protection["state"])
        if status in {"closed", "filled"}:
            if current is not OrderState.FILLED:
                self.store.transition_protection(protection_id, OrderState.FILLED)
            self._resolve_issue("CLOSED_PROTECTION_PENDING_FILL", protection_id)
            self._resolve_issue("MISSING_PROTECTION_ORDER", protection_id)
        elif current in {OrderState.OPEN, OrderState.UNKNOWN}:
            self.store.transition_protection(protection_id, OrderState.PARTIALLY_FILLED)

    def _store_fill(
        self,
        fill: Mapping[str, Any],
        fill_id: str,
        exchange_order_id: str,
        order: StoredOrder | None,
    ) -> FillResult:
        side = Side(str(fill.get("side", "")).upper())
        symbol = fill.get("symbol")
        if not _exact_id(symbol):
            msg = "fill symbol is required"
            raise ValueError(msg)
        info = fill.get("info")
        raw_info = info if isinstance(info, Mapping) else {}
        price = _external_decimal(
            "fill price",
            raw_info.get("price", fill.get("price")),
            positive=True,
        )
        quantity = _external_decimal(
            "fill amount",
            raw_info.get("vol", fill.get("amount")),
            positive=True,
        )
        cost = _external_decimal(
            "fill cost",
            raw_info.get("cost", fill.get("cost", price * quantity)),
            positive=True,
        )
        executed_at_ms = _timestamp("fill timestamp", fill.get("timestamp"))
        fee_data = fill.get("fee")
        fee = Decimal(0)
        fee_currency = None
        if isinstance(fee_data, Mapping):
            fee = _external_decimal(
                "fill fee",
                raw_info.get("fee", fee_data.get("cost", 0)),
                positive=False,
            )
            raw_currency = fee_data.get("currency")
            if raw_currency is not None:
                if not _exact_id(raw_currency):
                    msg = "fill fee currency is invalid"
                    raise ValueError(msg)
                fee_currency = raw_currency
        return self.store.apply_fill(
            fill_id,
            symbol,
            side,
            price,
            quantity,
            executed_at_ms=executed_at_ms,
            cost=cost,
            fee=fee,
            fee_currency=fee_currency,
            order_id=None if order is None else order.order_id,
            protection_id=(
                None
                if order is not None
                else (
                    str(protection["protection_id"])
                    if (protection := self.store.get_protection_by_exchange_id(exchange_order_id))
                    is not None
                    else None
                )
            ),
            payload={
                "exchange_order_id": exchange_order_id,
                "raw_cost": str(raw_info.get("cost", fill.get("cost", cost))),
                "raw_price": str(raw_info.get("price", fill.get("price", price))),
                "raw_quantity": str(raw_info.get("vol", fill.get("amount", quantity))),
            },
        )

    def _adopt_closed_conditional_stop(
        self,
        fill: Mapping[str, Any],
        exchange_order_id: str,
    ) -> dict[str, Any] | None:
        """Adopt a filled conditional child only through its exact known parent."""
        symbol = fill.get("symbol")
        if not _exact_id(symbol) or str(fill.get("side") or "").lower() != "sell":
            return None
        try:
            exchange_order = self.exchange.fetch_order(exchange_order_id, symbol)
        except Exception:  # noqa: BLE001
            return None
        protection_id = self._adopt_conditional_stop(
            exchange_order,
            symbol,
            verify_open=False,
        )
        return None if protection_id is None else self.store.get_protection(protection_id)

    def recover_unknown_orders(self) -> tuple[RecoveryResult, ...]:
        """Recover uncertain submissions by exact exchange or client identifiers."""
        recovered: list[RecoveryResult] = []
        for stored_order in self.store.list_orders(active_only=True):
            if stored_order.state is OrderState.CREATED:
                self.store.transition_order(
                    stored_order.order_id,
                    OrderState.FAILED,
                    expected_state=OrderState.CREATED,
                    payload={"reason": "recovery proved mutation was never admitted"},
                )
                continue
            if stored_order.state not in {OrderState.SUBMITTING, OrderState.UNKNOWN}:
                continue
            order_for_recovery = stored_order
            if stored_order.state is OrderState.SUBMITTING:
                self._record_interrupted_submission(stored_order)
                order_for_recovery = self.store.transition_order(
                    stored_order.order_id,
                    OrderState.UNKNOWN,
                )
            recovered.append(self._recover_order(order_for_recovery))
        return tuple(recovered)

    def _record_interrupted_submission(self, order: StoredOrder) -> None:
        self.store.record_reconciliation_issue(
            "UNKNOWN_SUBMISSION",
            order.client_order_id,
            {"message": "recovery found an interrupted submission", "order_id": order.order_id},
        )

    def _recover_order(self, order: StoredOrder) -> RecoveryResult:
        context = self.store.get_order_context(order.order_id)
        try:
            exchange_order = self._find_exchange_order(order, context["symbol"])
        except Exception as error:  # noqa: BLE001
            self.store.record_reconciliation_issue(
                "ORDER_RECOVERY_FAILED",
                order.client_order_id,
                {"error_type": type(error).__name__, "message": str(error)},
            )
            return RecoveryResult(
                order_id=order.order_id,
                client_order_id=order.client_order_id,
                exchange_order_id=order.exchange_order_id,
                state=order.state,
                recovered=False,
            )
        if exchange_order is None:
            return RecoveryResult(
                order_id=order.order_id,
                client_order_id=order.client_order_id,
                exchange_order_id=order.exchange_order_id,
                state=order.state,
                recovered=False,
            )
        try:
            return self._finish_recovery(order, exchange_order)
        except Exception as error:  # noqa: BLE001
            self.store.record_reconciliation_issue(
                "ORDER_RECOVERY_FAILED",
                order.client_order_id,
                {"error_type": type(error).__name__, "message": str(error)},
            )
            return RecoveryResult(
                order_id=order.order_id,
                client_order_id=order.client_order_id,
                exchange_order_id=order.exchange_order_id,
                state=order.state,
                recovered=False,
            )

    def _find_exchange_order(
        self,
        order: StoredOrder,
        symbol: str,
    ) -> Mapping[str, Any] | None:
        if order.exchange_order_id is not None:
            try:
                return self.exchange.fetch_order(
                    order.exchange_order_id,
                    symbol,
                    client_order_id=order.client_order_id,
                )
            except ccxt.OrderNotFound:
                return None
        open_orders = self.exchange.fetch_open_orders(
            symbol,
            client_order_id=order.client_order_id,
        )
        closed_orders = self.exchange.fetch_closed_orders(
            symbol,
            client_order_id=order.client_order_id,
        )
        matches = [
            item
            for item in (*open_orders, *closed_orders)
            if _order_client_id(item) == order.client_order_id
        ]
        unique = {item.get("id"): item for item in matches if _exact_id(item.get("id"))}
        if len(unique) > 1:
            msg = f"multiple exchange orders have clientOrderId {order.client_order_id!r}"
            raise ValueError(msg)
        return next(iter(unique.values()), None)

    def _finish_recovery(
        self,
        order: StoredOrder,
        exchange_order: Mapping[str, Any],
    ) -> RecoveryResult:
        exchange_order_id = exchange_order.get("id")
        if not _exact_id(exchange_order_id):
            msg = "recovered order has no exact exchange id"
            raise ValueError(msg)
        response_client_id = _order_client_id(exchange_order)
        if response_client_id != order.client_order_id:
            msg = "recovered order clientOrderId mismatch"
            raise ValueError(msg)
        order = self.store.bind_exchange_order_id(order.order_id, exchange_order_id)
        status = str(exchange_order.get("status") or "").lower()
        target = _recoverable_order_state(status)
        if target is None:
            self.store.record_reconciliation_issue(
                "UNKNOWN_SUBMISSION",
                order.client_order_id,
                {"message": f"recovered order status {status!r} requires fills"},
            )
            return RecoveryResult(
                order_id=order.order_id,
                client_order_id=order.client_order_id,
                exchange_order_id=order.exchange_order_id,
                state=order.state,
                recovered=False,
            )
        order = self.store.transition_order(order.order_id, target)
        self._resolve_issue("UNKNOWN_SUBMISSION", order.client_order_id)
        self._resolve_issue("UNKNOWN_CANCELLATION", order.client_order_id)
        self._resolve_issue("CANCELLATION_REJECTED", order.client_order_id)
        self._resolve_issue("ORDER_RECOVERY_FAILED", order.client_order_id)
        self._resolve_issue("STALE_ORDER_LOOKUP_FAILED", order.client_order_id)
        return RecoveryResult(
            order_id=order.order_id,
            client_order_id=order.client_order_id,
            exchange_order_id=order.exchange_order_id,
            state=order.state,
            recovered=True,
        )

    def reconcile_open_orders(
        self,
        symbols: str | Iterable[str],
    ) -> OpenOrderReconciliationResult:
        """Refresh known orders and adopt only exact Kraken conditional stop children."""
        exact_symbols = _exact_symbols(symbols)
        refreshed: list[RefreshedOrder] = []
        adopted: list[str] = []
        unknown: list[str] = []
        observed_exchange_ids: set[str] = set()
        for symbol in exact_symbols:
            try:
                exchange_orders = self.exchange.fetch_open_orders(symbol)
            except Exception as error:  # noqa: BLE001
                self.store.record_reconciliation_issue(
                    "OPEN_ORDER_RECONCILIATION_FAILED",
                    symbol,
                    {"error_type": type(error).__name__, "message": str(error)},
                )
                continue
            self._resolve_issue("OPEN_ORDER_RECONCILIATION_FAILED", symbol)
            for index, exchange_order in enumerate(exchange_orders):
                exchange_order_id = exchange_order.get("id")
                if not _exact_id(exchange_order_id):
                    exact_key = f"{symbol}:missing:{index}"
                    self.store.record_reconciliation_issue(
                        "INVALID_OPEN_ORDER",
                        exact_key,
                        {"message": "open order has no exact exchange id"},
                    )
                    unknown.append(exact_key)
                    continue
                observed_exchange_ids.add(exchange_order_id)
                known = self._known_open_order(exchange_order, symbol)
                if known is not None:
                    refreshed.append(known)
                    self._resolve_issue("HALT_UNKNOWN_OPEN_ORDER", exchange_order_id)
                    continue
                known_protection = self._known_open_protection(exchange_order, symbol)
                if known_protection is not None:
                    self._resolve_issue("HALT_UNKNOWN_OPEN_ORDER", exchange_order_id)
                    continue
                protection_id = self._adopt_conditional_stop(exchange_order, symbol)
                if protection_id is not None:
                    adopted.append(protection_id)
                    self._resolve_issue("HALT_UNKNOWN_OPEN_ORDER", exchange_order_id)
                    continue
                self.store.record_reconciliation_issue(
                    "HALT_UNKNOWN_OPEN_ORDER",
                    exchange_order_id,
                    {"exchange_order_id": exchange_order_id, "symbol": symbol},
                )
                unknown.append(exchange_order_id)
        refreshed.extend(
            self._refresh_missing_known_orders(exact_symbols, observed_exchange_ids),
        )
        self._refresh_missing_protections(exact_symbols, observed_exchange_ids)
        return OpenOrderReconciliationResult(
            refreshed=tuple(refreshed),
            adopted_protection_ids=tuple(adopted),
            unknown_exchange_order_ids=tuple(unknown),
        )

    def _known_open_order(
        self,
        exchange_order: Mapping[str, Any],
        symbol: str,
    ) -> RefreshedOrder | None:
        exchange_order_id = exchange_order["id"]
        order = self.store.get_order_by_exchange_id(exchange_order_id)
        response_client_id = _order_client_id(exchange_order)
        if order is None and response_client_id is not None:
            order = self.store.get_order_by_client_id(response_client_id)
            if order is not None and order.exchange_order_id is None:
                order = self.store.bind_exchange_order_id(order.order_id, exchange_order_id)
        if order is None:
            return None
        if order.exchange_order_id != exchange_order_id:
            self.store.record_reconciliation_issue(
                "KNOWN_ORDER_MISMATCH",
                exchange_order_id,
                {
                    "expected_exchange_order_id": order.exchange_order_id,
                    "message": "clientOrderId is bound to a different exchange order",
                },
            )
            return RefreshedOrder(order.order_id, exchange_order_id, order.state)
        context = self.store.get_order_context(order.order_id)
        if context["symbol"] != symbol:
            self.store.record_reconciliation_issue(
                "KNOWN_ORDER_MISMATCH",
                exchange_order_id,
                {"expected_symbol": context["symbol"], "observed_symbol": symbol},
            )
            return RefreshedOrder(order.order_id, exchange_order_id, order.state)
        if order.state in {
            OrderState.CANCELED,
            OrderState.FILLED,
            OrderState.REJECTED,
            OrderState.FAILED,
        }:
            self.store.record_reconciliation_issue(
                "KNOWN_ORDER_MISMATCH",
                exchange_order_id,
                {"message": f"terminal local order is exchange-open: {order.state.value}"},
            )
            return RefreshedOrder(order.order_id, exchange_order_id, order.state)
        if response_client_id is not None and response_client_id != order.client_order_id:
            self.store.record_reconciliation_issue(
                "KNOWN_ORDER_MISMATCH",
                exchange_order_id,
                {"message": "clientOrderId mismatch"},
            )
            return RefreshedOrder(order.order_id, exchange_order_id, order.state)
        if order.state in {
            OrderState.CREATED,
            OrderState.SUBMITTING,
            OrderState.CANCEL_PENDING,
            OrderState.UNKNOWN,
        }:
            order = self.store.transition_order(order.order_id, OrderState.OPEN)
        self._resolve_issue("UNKNOWN_SUBMISSION", order.client_order_id)
        self._resolve_issue("UNKNOWN_CANCELLATION", order.client_order_id)
        self._resolve_issue("KNOWN_ORDER_MISMATCH", exchange_order_id)
        return RefreshedOrder(order.order_id, exchange_order_id, order.state)

    def _known_open_protection(
        self,
        exchange_order: Mapping[str, Any],
        symbol: str,
    ) -> dict[str, Any] | None:
        exchange_order_id = exchange_order["id"]
        protection = self.store.get_protection_by_exchange_id(exchange_order_id)
        if protection is None:
            client_order_id = _order_client_id(exchange_order)
            if client_order_id is not None:
                protection = self.store.get_protection(client_order_id)
                if protection is not None and protection["exchange_order_id"] is None:
                    self.store.transition_protection(
                        client_order_id,
                        OrderState.OPEN,
                        exchange_order_id=exchange_order_id,
                    )
                    protection = self.store.get_protection(client_order_id)
        if protection is None:
            return None
        if protection["exchange_order_id"] != exchange_order_id:
            self.store.record_reconciliation_issue(
                "PROTECTION_MISMATCH",
                str(protection["protection_id"]),
                {
                    "expected_exchange_order_id": protection["exchange_order_id"],
                    "message": "clientOrderId is bound to a different protection order",
                },
            )
            return protection
        if protection["symbol"] != symbol:
            self.store.record_reconciliation_issue(
                "PROTECTION_MISMATCH",
                str(protection["protection_id"]),
                {"message": "protective order symbol mismatch"},
            )
            return protection
        protection_state = OrderState(protection["state"])
        if protection_state in {
            OrderState.CANCELED,
            OrderState.FILLED,
            OrderState.REJECTED,
            OrderState.FAILED,
        }:
            self.store.record_reconciliation_issue(
                "PROTECTION_MISMATCH",
                str(protection["protection_id"]),
                {
                    "message": "terminal local protection is exchange-open: "
                    f"{protection_state.value}",
                },
            )
            return protection
        if protection_state in {OrderState.CANCEL_PENDING, OrderState.UNKNOWN}:
            protection_id = str(protection["protection_id"])
            self.store.transition_protection(
                protection_id,
                OrderState.OPEN,
                exchange_order_id=exchange_order_id,
            )
            protection = self.store.get_protection(protection_id)
            if protection is None:
                msg = "protection disappeared during exact refresh"
                raise RuntimeError(msg)
        self.verify_protective_stop(str(protection["protection_id"]))
        return protection

    def _adopt_conditional_stop(  # noqa: C901, PLR0911
        self,
        exchange_order: Mapping[str, Any],
        symbol: str,
        *,
        verify_open: bool = True,
    ) -> str | None:
        info = exchange_order.get("info")
        if not isinstance(info, Mapping):
            return None
        parent_exchange_order_id = info.get("refid")
        if not _exact_id(parent_exchange_order_id):
            return None
        parent = self.store.get_order_by_exchange_id(parent_exchange_order_id)
        if parent is None:
            return None
        context = self.store.get_order_context(parent.order_id)
        if context["purpose"] not in {OrderPurpose.ENTRY.value, OrderPurpose.DCA.value}:
            return None
        if context["symbol"] != symbol:
            return None
        if str(exchange_order.get("side") or "").lower() != "sell":
            return None
        if not _is_stop_order(exchange_order):
            return None
        exchange_order_id = exchange_order["id"]
        try:
            quantity = _external_decimal(
                "conditional stop amount",
                exchange_order.get("amount"),
                positive=True,
            )
            trigger = _protection_trigger(exchange_order)
        except (TypeError, ValueError):
            return None
        position = self.store.get_position(symbol)
        if position is None:
            return None
        if quantity > position.quantity:
            return None
        protection_id = deterministic_client_id(
            self.store.scope,
            f"conditional:{exchange_order_id}",
        )
        self.store.record_protection(
            protection_id,
            symbol,
            "STOP_LOSS",
            quantity,
            trigger,
            epoch_id=position.epoch_id,
            exchange_order_id=exchange_order_id,
            state=OrderState.OPEN if verify_open else OrderState.UNKNOWN,
            payload={
                "client_order_id": _order_client_id(exchange_order),
                "parent_exchange_order_id": parent_exchange_order_id,
                "source": "KRAKEN_CONDITIONAL_CHILD",
            },
        )
        if not verify_open:
            return protection_id
        verification = self.verify_protective_stop(protection_id)
        return protection_id if verification.valid else None

    def _refresh_missing_known_orders(
        self,
        symbols: tuple[str, ...],
        observed_exchange_ids: set[str],
    ) -> list[RefreshedOrder]:
        refreshed: list[RefreshedOrder] = []
        for order in self.store.list_orders(active_only=True):
            exchange_order_id = order.exchange_order_id
            if exchange_order_id is None or exchange_order_id in observed_exchange_ids:
                continue
            context = self.store.get_order_context(order.order_id)
            if context["symbol"] not in symbols:
                continue
            refreshed.append(self._refresh_exact_order(order, context["symbol"]))
        return refreshed

    def _refresh_exact_order(self, order: StoredOrder, symbol: str) -> RefreshedOrder:
        exchange_order_id = order.exchange_order_id
        if exchange_order_id is None:
            msg = "exact order refresh requires an exchange order id"
            raise ValueError(msg)
        try:
            exchange_order = self.exchange.fetch_order(
                exchange_order_id,
                symbol,
                client_order_id=order.client_order_id,
            )
        except Exception as error:  # noqa: BLE001
            self.store.record_reconciliation_issue(
                "MISSING_KNOWN_ORDER",
                exchange_order_id,
                {"error_type": type(error).__name__, "message": str(error)},
            )
            if order.state is not OrderState.UNKNOWN:
                order = self.store.transition_order(order.order_id, OrderState.UNKNOWN)
            return RefreshedOrder(order.order_id, exchange_order_id, order.state)
        status = str(exchange_order.get("status") or "").lower()
        target = _recoverable_order_state(status)
        if target is None:
            self.store.record_reconciliation_issue(
                "CLOSED_ORDER_PENDING_FILLS",
                exchange_order_id,
                {"status": status},
            )
            if order.state is not OrderState.UNKNOWN:
                order = self.store.transition_order(order.order_id, OrderState.UNKNOWN)
        elif target is OrderState.OPEN and order.state is OrderState.PARTIALLY_FILLED:
            self._resolve_issue("MISSING_KNOWN_ORDER", exchange_order_id)
        elif target is OrderState.REJECTED and order.state is OrderState.PARTIALLY_FILLED:
            order = self.store.transition_order(order.order_id, OrderState.UNKNOWN)
            self.store.record_reconciliation_issue(
                "KNOWN_ORDER_MISMATCH",
                exchange_order_id,
                {"message": "partially filled order was reported rejected"},
            )
        elif target is not order.state:
            order = self.store.transition_order(order.order_id, target)
            self._resolve_issue("MISSING_KNOWN_ORDER", exchange_order_id)
        return RefreshedOrder(order.order_id, exchange_order_id, order.state)

    def _refresh_missing_protections(
        self,
        symbols: tuple[str, ...],
        observed_exchange_ids: set[str],
    ) -> None:
        for symbol in symbols:
            for protection in self.store.list_protection(symbol, active_only=True):
                exchange_order_id = protection["exchange_order_id"]
                if not _exact_id(exchange_order_id):
                    continue
                if exchange_order_id in observed_exchange_ids:
                    continue
                try:
                    exchange_order = self.exchange.fetch_order(exchange_order_id, symbol)
                except Exception as error:  # noqa: BLE001
                    self.store.record_reconciliation_issue(
                        "MISSING_PROTECTION_ORDER",
                        str(protection["protection_id"]),
                        {"error_type": type(error).__name__, "message": str(error)},
                    )
                    self.store.transition_protection(
                        str(protection["protection_id"]),
                        OrderState.UNKNOWN,
                    )
                    continue
                status = str(exchange_order.get("status") or "").lower()
                target = _recoverable_order_state(status)
                protection_id = str(protection["protection_id"])
                if target is OrderState.OPEN:
                    self.verify_protective_stop(protection_id)
                    self._resolve_issue("MISSING_PROTECTION_ORDER", protection_id)
                elif target is OrderState.CANCELED:
                    self.store.transition_protection(protection_id, OrderState.CANCELED)
                    self._resolve_issue("PROTECTION_CANCEL_PENDING", protection_id)
                    self._resolve_issue("PROTECTION_CANCEL_FAILED", protection_id)
                else:
                    self.store.transition_protection(protection_id, OrderState.UNKNOWN)
                    self.store.record_reconciliation_issue(
                        "CLOSED_PROTECTION_PENDING_FILL",
                        protection_id,
                        {"status": status},
                    )

    def ensure_protective_stop(  # noqa: C901, PLR0911, PLR0913
        self,
        symbol: str,
        quantity: Decimal,
        trigger_price: Decimal,
        entry_price: Decimal,
        current_price: Decimal,
        *,
        attempt: int = 0,
    ) -> ProtectionSubmissionResult:
        """Persist, submit, and read back one exact stop protecting current inventory."""
        quantity = _required_decimal("quantity", quantity, positive=True)
        trigger_price = _required_decimal("trigger_price", trigger_price, positive=True)
        entry_price = _required_decimal("entry_price", entry_price, positive=True)
        current_price = _required_decimal("current_price", current_price, positive=True)
        if trigger_price >= current_price:
            msg = "trigger_price must be below current_price"
            raise ValueError(msg)
        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 0:
            msg = "attempt must be a non-negative integer"
            raise ValueError(msg)
        position = self.store.get_position(symbol)
        if position is None:
            msg = f"cannot protect {symbol} without a stored position"
            raise ValueError(msg)
        quantity, trigger_price = self.exchange.normalize_stop_loss(
            symbol,
            quantity,
            trigger_price,
        )
        if quantity != position.quantity:
            msg = f"protection quantity {quantity} does not equal position {position.quantity}"
            raise ValueError(msg)

        existing = self._matching_active_protection(
            symbol,
            position.epoch_id,
            quantity,
            trigger_price,
        )
        if existing is not None:
            if OrderState(existing["state"]) is OrderState.CREATED:
                protection_id = str(existing["protection_id"])
                self.store.transition_protection(protection_id, OrderState.SUBMITTING)
                try:
                    response = self.exchange.place_stop_loss(
                        symbol,
                        quantity,
                        trigger_price,
                        client_order_id=protection_id,
                    )
                except Exception as error:  # noqa: BLE001
                    return self._protection_submission_error(protection_id, error)
                if not isinstance(response, Mapping):
                    response = {}
                return self._protection_submission_response(protection_id, response)
            return self._existing_protection_result(existing)
        active_protections = self.store.list_protection(symbol, active_only=True)
        if active_protections:
            conflict = active_protections[0]
            return ProtectionSubmissionResult(
                protection_id=str(conflict["protection_id"]),
                exchange_order_id=conflict["exchange_order_id"],
                outcome=SubmissionOutcome.UNKNOWN,
                state=OrderState(conflict["state"]),
                error="conflicting active protection must be reconciled first",
            )
        protection_id = deterministic_client_id(
            self.store.scope,
            "stop:"
            f"{symbol}:{position.epoch_id}:{canonical_decimal(quantity)}:"
            f"{canonical_decimal(trigger_price)}",
            attempt=attempt,
        )
        stored = self.store.get_protection(protection_id)
        if stored is not None:
            return self._existing_protection_result(stored)
        self.store.record_protection(
            protection_id,
            symbol,
            "STOP_LOSS",
            quantity,
            trigger_price,
            epoch_id=position.epoch_id,
            state=OrderState.CREATED,
            payload={
                "attempt": attempt,
                "client_order_id": protection_id,
                "current_price": current_price,
                "entry_price": entry_price,
                "source": "COORDINATOR",
            },
        )
        self.store.transition_protection(protection_id, OrderState.SUBMITTING)
        try:
            response = self.exchange.place_stop_loss(
                symbol,
                quantity,
                trigger_price,
                client_order_id=protection_id,
            )
        except Exception as error:  # noqa: BLE001
            return self._protection_submission_error(protection_id, error)
        if not isinstance(response, Mapping):
            response = {}
        return self._protection_submission_response(protection_id, response)

    def _matching_active_protection(
        self,
        symbol: str,
        epoch_id: str,
        quantity: Decimal,
        trigger_price: Decimal,
    ) -> dict[str, Any] | None:
        for protection in self.store.list_protection(symbol, active_only=True):
            if (
                protection["epoch_id"] == epoch_id
                and Decimal(str(protection["quantity"])) == quantity
                and Decimal(str(protection["trigger_price"])) == trigger_price
            ):
                return protection
            self.store.record_reconciliation_issue(
                "PROTECTION_CONFLICT",
                str(protection["protection_id"]),
                {
                    "expected_quantity": str(quantity),
                    "expected_trigger_price": str(trigger_price),
                    "symbol": symbol,
                },
            )
        return None

    def ratchet_protective_stops(
        self,
        symbol: str,
        trigger_price: Decimal,
        current_price: Decimal,
    ) -> bool:
        """Raise every exact stop in place; never lower or cancel protection."""
        trigger_price = _required_decimal("trigger_price", trigger_price, positive=True)
        current_price = _required_decimal("current_price", current_price, positive=True)
        position = self.store.get_position(symbol)
        if position is None:
            return False
        normalized_quantity, normalized_trigger = self.exchange.normalize_stop_loss(
            symbol,
            position.quantity,
            trigger_price,
        )
        if normalized_quantity != position.quantity or normalized_trigger >= current_price:
            return False
        protections = self.store.list_protection(symbol, active_only=True)
        if not protections:
            return False
        for protection in protections:
            if protection["epoch_id"] != position.epoch_id:
                return False
            prior_trigger = Decimal(str(protection["trigger_price"]))
            if normalized_trigger <= prior_trigger:
                continue
            if not self._amend_protective_stop(protection, normalized_trigger):
                return False
        return True

    def _amend_protective_stop(
        self,
        protection: Mapping[str, Any],
        trigger_price: Decimal,
    ) -> bool:
        protection_id = str(protection["protection_id"])
        exchange_order_id = protection.get("exchange_order_id")
        if not _exact_id(exchange_order_id):
            return False
        quantity = Decimal(str(protection["quantity"]))
        amend_id = None
        try:
            response = self.exchange.amend_stop_loss(
                exchange_order_id,
                str(protection["symbol"]),
                quantity,
                trigger_price,
            )
            if isinstance(response, Mapping) and _exact_id(response.get("id")):
                amend_id = str(response["id"])
        except Exception as error:  # noqa: BLE001
            if not _is_ambiguous_mutation_error(error):
                return False
        try:
            order = self.exchange.fetch_order(exchange_order_id, str(protection["symbol"]))
            actual_remaining = _remaining_order_quantity(order)
            actual_trigger = _protection_trigger(order)
        except Exception as error:  # noqa: BLE001
            self.store.record_reconciliation_issue(
                "UNKNOWN_PROTECTION_AMENDMENT",
                protection_id,
                {"error_type": type(error).__name__, "message": str(error)},
            )
            return False
        expected_remaining = self.store.get_protection_remaining_quantity(protection_id)
        if (
            order.get("id") != exchange_order_id
            or actual_remaining != expected_remaining
            or actual_trigger != trigger_price
        ):
            self.store.record_reconciliation_issue(
                "UNKNOWN_PROTECTION_AMENDMENT",
                protection_id,
                {
                    "actual_remaining": str(actual_remaining),
                    "actual_trigger": str(actual_trigger),
                    "expected_remaining": str(expected_remaining),
                    "expected_trigger": str(trigger_price),
                },
            )
            return False
        self.store.amend_protection_expectation(
            protection_id,
            quantity,
            trigger_price,
            exchange_amend_id=amend_id,
        )
        self._resolve_issue("UNKNOWN_PROTECTION_AMENDMENT", protection_id)
        return self.verify_protective_stop(protection_id).valid

    def _existing_protection_result(
        self,
        protection: Mapping[str, Any],
    ) -> ProtectionSubmissionResult:
        protection_id = str(protection["protection_id"])
        state = OrderState(protection["state"])
        exchange_order_id = protection["exchange_order_id"]
        if state in {OrderState.SUBMITTING, OrderState.CANCEL_PENDING, OrderState.UNKNOWN}:
            self.store.record_reconciliation_issue(
                "UNKNOWN_PROTECTION_SUBMISSION",
                protection_id,
                {"state": state.value},
            )
            return ProtectionSubmissionResult(
                protection_id,
                exchange_order_id,
                SubmissionOutcome.UNKNOWN,
                state,
                error="protection submission is unresolved",
            )
        verification = None
        if (
            state in {OrderState.OPEN, OrderState.PARTIALLY_FILLED}
            and exchange_order_id is not None
        ):
            verification = self.verify_protective_stop(protection_id)
        outcome = (
            SubmissionOutcome.ALREADY_KNOWN
            if state in {OrderState.OPEN, OrderState.PARTIALLY_FILLED}
            else SubmissionOutcome.REJECTED
        )
        return ProtectionSubmissionResult(
            protection_id,
            exchange_order_id,
            outcome,
            state,
            verification,
        )

    def _protection_submission_error(
        self,
        protection_id: str,
        error: Exception,
    ) -> ProtectionSubmissionResult:
        if _is_ambiguous_mutation_error(error):
            state = OrderState.UNKNOWN
            outcome = SubmissionOutcome.UNKNOWN
            self.store.record_reconciliation_issue(
                "UNKNOWN_PROTECTION_SUBMISSION",
                protection_id,
                {"error_type": type(error).__name__, "message": str(error)},
            )
        elif isinstance(error, (ccxt.ExchangeError, ValueError)):
            state = OrderState.REJECTED
            outcome = SubmissionOutcome.REJECTED
        else:
            state = OrderState.UNKNOWN
            outcome = SubmissionOutcome.UNKNOWN
            self.store.record_reconciliation_issue(
                "UNKNOWN_PROTECTION_SUBMISSION",
                protection_id,
                {"error_type": type(error).__name__, "message": str(error)},
            )
        self.store.transition_protection(protection_id, state)
        return ProtectionSubmissionResult(
            protection_id,
            None,
            outcome,
            state,
            error=str(error),
        )

    def _protection_submission_response(
        self,
        protection_id: str,
        response: Mapping[str, Any],
    ) -> ProtectionSubmissionResult:
        status = str(response.get("status") or "open").lower()
        exchange_order_id = response.get("id")
        if status in {"rejected", "expired"}:
            self.store.transition_protection(protection_id, OrderState.REJECTED)
            return ProtectionSubmissionResult(
                protection_id,
                exchange_order_id if _exact_id(exchange_order_id) else None,
                SubmissionOutcome.REJECTED,
                OrderState.REJECTED,
            )
        if not _exact_id(exchange_order_id):
            message = "protection response has no exact exchange order id"
            self.store.transition_protection(protection_id, OrderState.UNKNOWN)
            self.store.record_reconciliation_issue(
                "UNKNOWN_PROTECTION_SUBMISSION",
                protection_id,
                {"message": message},
            )
            return ProtectionSubmissionResult(
                protection_id,
                None,
                SubmissionOutcome.UNKNOWN,
                OrderState.UNKNOWN,
                error=message,
            )
        response_client_id = _order_client_id(response)
        if response_client_id is not None and response_client_id != protection_id:
            message = "protection response clientOrderId mismatch"
            self.store.transition_protection(protection_id, OrderState.UNKNOWN)
            self.store.record_reconciliation_issue(
                "UNKNOWN_PROTECTION_SUBMISSION",
                protection_id,
                {"message": message},
            )
            return ProtectionSubmissionResult(
                protection_id,
                None,
                SubmissionOutcome.UNKNOWN,
                OrderState.UNKNOWN,
                error=message,
            )
        self.store.transition_protection(
            protection_id,
            OrderState.OPEN,
            exchange_order_id=exchange_order_id,
        )
        verification = self.verify_protective_stop(protection_id)
        outcome = SubmissionOutcome.SUBMITTED if verification.valid else SubmissionOutcome.UNKNOWN
        return ProtectionSubmissionResult(
            protection_id,
            exchange_order_id,
            outcome,
            OrderState.OPEN,
            verification,
            None if verification.valid else verification.error,
        )

    def submit_protected_exit(
        self,
        intent_id: str,
        symbol: str,
        quantity: Decimal,
        current_price: Decimal,
        *,
        payload: Mapping[str, Any] | None = None,
    ) -> ProtectedExitResult:
        """Hand off exact stop coverage to one emergency market sell without a gap on failure."""
        quantity = _required_decimal("quantity", quantity, positive=True)
        current_price = _required_decimal("current_price", current_price, positive=True)
        position = self.store.get_position(symbol)
        if position is None:
            msg = f"cannot exit {symbol} without a stored position"
            raise ValueError(msg)
        quantity, current_price = self.exchange.normalize_market_sell(
            symbol,
            quantity,
            reference_price=current_price,
        )
        if current_price is None:
            msg = "protected exit requires a normalized reference price"
            raise ValueError(msg)
        if quantity != position.quantity:
            msg = f"protected exit quantity {quantity} does not equal position {position.quantity}"
            raise ValueError(msg)
        pending_exit = self._active_exit(symbol)
        if pending_exit is not None:
            return ProtectedExitResult(
                submission=self._submission_result(
                    pending_exit,
                    SubmissionOutcome.ALREADY_KNOWN,
                    error="prior exit remains pending exact fills",
                ),
                canceled_protection_ids=(),
                restored=(),
                ready=False,
                error="prior exit remains pending exact fills",
            )
        protections = self.store.list_protection(symbol, active_only=True)
        canceled: list[dict[str, Any]] = []
        for protection in protections:
            error = self._cancel_protection_for_handoff(protection, symbol)
            if error is not None:
                restored_after_cancel_failure = self._restore_protections(
                    canceled,
                    current_price,
                    position.average_cost,
                )
                self.store.record_reconciliation_issue(
                    "PROTECTED_EXIT_HANDOFF_FAILED",
                    intent_id,
                    {"message": error},
                )
                return ProtectedExitResult(
                    submission=None,
                    canceled_protection_ids=tuple(str(item["protection_id"]) for item in canceled),
                    restored=restored_after_cancel_failure,
                    ready=self.ready,
                    error=error,
                )
            canceled.append(protection)
        submission = self.submit_exit(
            intent_id,
            symbol,
            quantity,
            current_price,
            emergency=True,
            payload=payload,
        )
        restored: tuple[ProtectionSubmissionResult, ...] = ()
        if submission.outcome in {SubmissionOutcome.REJECTED, SubmissionOutcome.UNKNOWN}:
            restored = self._restore_protections(canceled, current_price, position.average_cost)
            if canceled and (
                not restored
                or any(
                    item.outcome
                    not in {SubmissionOutcome.SUBMITTED, SubmissionOutcome.ALREADY_KNOWN}
                    for item in restored
                )
            ):
                self.store.record_reconciliation_issue(
                    "PROTECTED_EXIT_HANDOFF_FAILED",
                    intent_id,
                    {"message": "exit failed and prior protection was not fully restored"},
                )
        else:
            self.store.record_reconciliation_issue(
                "EXIT_PENDING_FILL",
                submission.client_order_id,
                {
                    "exchange_order_id": submission.exchange_order_id,
                    "symbol": symbol,
                },
            )
            self._resolve_exit_handoff_failures(symbol)
        return ProtectedExitResult(
            submission=submission,
            canceled_protection_ids=tuple(str(item["protection_id"]) for item in canceled),
            restored=restored,
            ready=self.ready,
        )

    def _active_exit(self, symbol: str) -> StoredOrder | None:
        for order in self.store.list_orders(active_only=True):
            context = self.store.get_order_context(order.order_id)
            if context["symbol"] == symbol and context["purpose"] == OrderPurpose.EXIT.value:
                return order
        return None

    def _cancel_protection_for_handoff(
        self,
        protection: Mapping[str, Any],
        symbol: str,
    ) -> str | None:
        protection_id = str(protection["protection_id"])
        state = OrderState(protection["state"])
        exchange_order_id = protection["exchange_order_id"]
        if state not in {OrderState.OPEN, OrderState.PARTIALLY_FILLED}:
            return f"protection {protection_id} is {state.value}"
        if not _exact_id(exchange_order_id):
            return f"protection {protection_id} has no exact exchange order id"
        self.store.record_reconciliation_issue(
            "PROTECTION_CANCEL_PENDING",
            protection_id,
            {"exchange_order_id": exchange_order_id},
        )
        self.store.transition_protection(protection_id, OrderState.CANCEL_PENDING)
        client_order_id = _protection_client_order_id(protection)
        try:
            self.exchange.cancel_order(
                exchange_order_id,
                symbol,
                client_order_id=client_order_id,
            )
            exchange_order = self.exchange.fetch_order(exchange_order_id, symbol)
        except Exception as error:  # noqa: BLE001
            target = (
                OrderState.UNKNOWN
                if _is_ambiguous_mutation_error(error) or isinstance(error, ccxt.OrderNotFound)
                else state
            )
            self.store.transition_protection(protection_id, target)
            self.store.record_reconciliation_issue(
                "PROTECTION_CANCEL_FAILED",
                protection_id,
                {"error_type": type(error).__name__, "message": str(error)},
            )
            self._resolve_issue("PROTECTION_CANCEL_PENDING", protection_id)
            return str(error)
        status = str(exchange_order.get("status") or "").lower()
        if status not in {"canceled", "cancelled", "expired"}:
            self.store.transition_protection(protection_id, OrderState.UNKNOWN)
            message = f"protection cancel read-back returned {status!r}"
            self.store.record_reconciliation_issue(
                "PROTECTION_CANCEL_FAILED",
                protection_id,
                {"message": message},
            )
            self._resolve_issue("PROTECTION_CANCEL_PENDING", protection_id)
            return message
        self.store.transition_protection(protection_id, OrderState.CANCELED)
        self._resolve_issue("PROTECTION_CANCEL_PENDING", protection_id)
        self._resolve_issue("PROTECTION_CANCEL_FAILED", protection_id)
        return None

    def _restore_protections(
        self,
        protections: list[dict[str, Any]],
        current_price: Decimal,
        entry_price: Decimal,
    ) -> tuple[ProtectionSubmissionResult, ...]:
        restored: list[ProtectionSubmissionResult] = []
        for protection in protections:
            payload = protection.get("payload")
            prior_attempt = payload.get("attempt", 0) if isinstance(payload, Mapping) else 0
            try:
                result = self.ensure_protective_stop(
                    str(protection["symbol"]),
                    Decimal(str(protection["quantity"])),
                    Decimal(str(protection["trigger_price"])),
                    entry_price,
                    current_price,
                    attempt=int(prior_attempt) + 1,
                )
            except Exception as error:  # noqa: BLE001
                protection_id = str(protection["protection_id"])
                self.store.record_reconciliation_issue(
                    "PROTECTION_REARM_FAILED",
                    protection_id,
                    {"error_type": type(error).__name__, "message": str(error)},
                )
                result = ProtectionSubmissionResult(
                    protection_id=protection_id,
                    exchange_order_id=protection["exchange_order_id"],
                    outcome=SubmissionOutcome.REJECTED,
                    state=OrderState(protection["state"]),
                    error=str(error),
                )
            restored.append(result)
        return tuple(restored)

    def cancel_stale_entries(
        self,
        max_age_seconds: int,
        *,
        now_ms: int | None = None,
    ) -> tuple[CancellationResult, ...]:
        """Cancel only bound, known, stale ENTRY or DCA orders one at a time."""
        if isinstance(max_age_seconds, bool) or max_age_seconds <= 0:
            msg = "max_age_seconds must be a positive integer"
            raise ValueError(msg)
        now = int(time.time() * 1000) if now_ms is None else _timestamp("now_ms", now_ms)
        results: list[CancellationResult] = []
        for order in self.store.list_orders(active_only=True):
            if order.exchange_order_id is None:
                continue
            if order.state not in {OrderState.OPEN, OrderState.PARTIALLY_FILLED}:
                continue
            context = self.store.get_order_context(order.order_id)
            if context["purpose"] not in {OrderPurpose.ENTRY.value, OrderPurpose.DCA.value}:
                continue
            result = self._cancel_if_stale(order, context["symbol"], now, max_age_seconds)
            if result is not None:
                results.append(result)
        return tuple(results)

    def _cancel_if_stale(
        self,
        order: StoredOrder,
        symbol: str,
        now_ms: int,
        max_age_seconds: int,
    ) -> CancellationResult | None:
        exchange_order_id = order.exchange_order_id
        if exchange_order_id is None:
            msg = "stale cancellation requires a bound exchange order id"
            raise ValueError(msg)
        try:
            exchange_order = self.exchange.fetch_order(
                exchange_order_id,
                symbol,
                client_order_id=order.client_order_id,
            )
        except Exception as error:  # noqa: BLE001
            self.store.record_reconciliation_issue(
                "STALE_ORDER_LOOKUP_FAILED",
                order.client_order_id,
                {"error_type": type(error).__name__, "message": str(error)},
            )
            order = self.store.transition_order(order.order_id, OrderState.UNKNOWN)
            return CancellationResult(
                order_id=order.order_id,
                exchange_order_id=exchange_order_id,
                state=order.state,
                canceled=False,
                error=str(error),
            )
        opened_at_ms = _timestamp("order timestamp", exchange_order.get("timestamp"))
        if now_ms - opened_at_ms <= max_age_seconds * 1000:
            return None
        previous_state = order.state
        order = self.store.transition_order(order.order_id, OrderState.CANCEL_PENDING)
        try:
            self.exchange.cancel_order(
                exchange_order_id,
                symbol,
                client_order_id=order.client_order_id,
            )
        except Exception as error:  # noqa: BLE001
            if _is_ambiguous_mutation_error(error) or isinstance(error, ccxt.OrderNotFound):
                self.store.record_reconciliation_issue(
                    "UNKNOWN_CANCELLATION",
                    order.client_order_id,
                    {"error_type": type(error).__name__, "message": str(error)},
                )
                order = self.store.transition_order(order.order_id, OrderState.UNKNOWN)
            else:
                order = self.store.transition_order(order.order_id, previous_state)
                self.store.record_reconciliation_issue(
                    "CANCELLATION_REJECTED",
                    order.client_order_id,
                    {"error_type": type(error).__name__, "message": str(error)},
                )
            return CancellationResult(
                order_id=order.order_id,
                exchange_order_id=order.exchange_order_id or "",
                state=order.state,
                canceled=False,
                error=str(error),
            )
        order = self.store.transition_order(order.order_id, OrderState.CANCELED)
        self._resolve_issue("UNKNOWN_CANCELLATION", order.client_order_id)
        self._resolve_issue("CANCELLATION_REJECTED", order.client_order_id)
        self._resolve_issue("STALE_ORDER_LOOKUP_FAILED", order.client_order_id)
        return CancellationResult(
            order_id=order.order_id,
            exchange_order_id=order.exchange_order_id or "",
            state=order.state,
            canceled=True,
        )

    def compare_balance(
        self,
        symbol: str,
        *,
        tolerance: Decimal = Decimal(0),
    ) -> BalanceComparison:
        """Compare one store position with the exact total base balance."""
        tolerance = _required_decimal("tolerance", tolerance, positive=False)
        position = self.store.get_position(symbol)
        expected = Decimal(0) if position is None else position.quantity
        currency, separator, _quote = symbol.partition("/")
        if not separator or not currency:
            msg = "symbol must use BASE/QUOTE form"
            raise ValueError(msg)
        actual = _external_decimal(
            "exchange balance",
            self.exchange.get_total_balance(currency),
            positive=False,
        )
        difference = actual - expected
        matches = abs(difference) <= tolerance
        if matches:
            self._resolve_issue("BALANCE_MISMATCH", symbol)
        else:
            self.store.record_reconciliation_issue(
                "BALANCE_MISMATCH",
                symbol,
                {
                    "actual": str(actual),
                    "difference": str(difference),
                    "expected": str(expected),
                    "tolerance": str(tolerance),
                },
            )
        return BalanceComparison(symbol, currency, expected, actual, difference, matches)

    def compare_balances(
        self,
        symbols: str | Iterable[str],
        *,
        tolerance: Decimal = Decimal(0),
    ) -> tuple[BalanceComparison, ...]:
        """Compare one or more symbols without inferring cross-symbol inventory."""
        exact_symbols = (symbols,) if isinstance(symbols, str) else tuple(symbols)
        return tuple(self.compare_balance(symbol, tolerance=tolerance) for symbol in exact_symbols)

    def recover(
        self,
        symbols: str | Iterable[str],
        tolerance: Decimal,
    ) -> RecoverySummary:
        """Run one exact recovery pass and return all evidence with final readiness."""
        exact_symbols = _exact_symbols(symbols)
        tolerance = _required_decimal("tolerance", tolerance, positive=False)
        recovered_orders = self.recover_unknown_orders()
        fills = self.reconcile_fills()
        open_orders = self.reconcile_open_orders(exact_symbols)
        balances = self.compare_balances(exact_symbols, tolerance=tolerance)
        protections: list[ProtectionVerification] = []
        for symbol in exact_symbols:
            active_protections = self.store.list_protection(symbol, active_only=True)
            protections.extend(
                self.verify_protective_stop(str(protection["protection_id"]))
                for protection in active_protections
            )
            self._verify_protection_coverage(symbol, active_protections)
        return RecoverySummary(
            recovered_orders=recovered_orders,
            open_orders=open_orders,
            fills=fills,
            balances=balances,
            protections=tuple(protections),
            readiness=self.readiness(),
        )

    def _verify_protection_coverage(
        self,
        symbol: str,
        active_protections: list[dict[str, Any]],
    ) -> None:
        """Require exact aggregate stop coverage for every open position."""
        position = self.store.get_position(symbol)
        if position is None:
            if active_protections:
                self.store.record_reconciliation_issue(
                    "PROTECTION_WITHOUT_POSITION",
                    symbol,
                    {"protection_ids": [item["protection_id"] for item in active_protections]},
                )
            else:
                self._resolve_issue("PROTECTION_WITHOUT_POSITION", symbol)
                self._resolve_issue("MISSING_PROTECTION", symbol)
                self._resolve_issue("PROTECTION_COVERAGE_MISMATCH", symbol)
            return
        if not active_protections:
            self.store.record_reconciliation_issue(
                "MISSING_PROTECTION",
                symbol,
                {"expected_quantity": str(position.quantity)},
            )
            return
        self._resolve_issue("MISSING_PROTECTION", symbol)
        current_epoch = [
            item for item in active_protections if item["epoch_id"] == position.epoch_id
        ]
        stale_ids = [
            str(item["protection_id"])
            for item in active_protections
            if item["epoch_id"] != position.epoch_id
        ]
        if stale_ids:
            self.store.record_reconciliation_issue(
                "PROTECTION_EPOCH_MISMATCH",
                symbol,
                {"epoch_id": position.epoch_id, "protection_ids": stale_ids},
            )
        else:
            self._resolve_issue("PROTECTION_EPOCH_MISMATCH", symbol)
        covered_quantity = sum(
            (
                self.store.get_protection_remaining_quantity(str(item["protection_id"]))
                for item in current_epoch
            ),
            start=Decimal(0),
        )
        if covered_quantity != position.quantity:
            self.store.record_reconciliation_issue(
                "PROTECTION_COVERAGE_MISMATCH",
                symbol,
                {
                    "covered_quantity": str(covered_quantity),
                    "position_quantity": str(position.quantity),
                },
            )
            return
        self._resolve_issue("PROTECTION_COVERAGE_MISMATCH", symbol)
        self._resolve_issue("PROTECTION_WITHOUT_POSITION", symbol)

    def verify_protective_stop(
        self,
        protection_id: str,
        *,
        symbol: str | None = None,
        exchange_order_id: str | None = None,
        expected_quantity: Decimal | None = None,
        expected_trigger_price: Decimal | None = None,
    ) -> ProtectionVerification:
        """Verify stop coverage from one exact exchange order, never by fuzzy matching."""
        stored = self.store.get_protection(protection_id)
        symbol, exchange_order_id, expected_quantity, expected_trigger_price = (
            self._protection_expectation(
                stored,
                symbol,
                exchange_order_id,
                expected_quantity,
                expected_trigger_price,
            )
        )
        if exchange_order_id is None:
            return self._protection_failure(
                protection_id,
                None,
                expected_quantity,
                expected_trigger_price,
                None,
                None,
                "protection has no exact exchange order id",
            )
        try:
            order = self.exchange.fetch_order(exchange_order_id, symbol)
            actual_quantity = _remaining_order_quantity(order)
            actual_trigger = _protection_trigger(order)
        except Exception as error:  # noqa: BLE001
            return self._protection_failure(
                protection_id,
                exchange_order_id,
                expected_quantity,
                expected_trigger_price,
                None,
                None,
                str(error),
            )
        status = str(order.get("status") or "open").lower()
        valid = (
            order.get("id") == exchange_order_id
            and str(order.get("side") or "").lower() == "sell"
            and _is_stop_order(order)
            and actual_quantity == expected_quantity
            and actual_trigger == expected_trigger_price
            and status in {"open", "pending", "partially_filled"}
        )
        if not valid:
            return self._protection_failure(
                protection_id,
                exchange_order_id,
                expected_quantity,
                expected_trigger_price,
                actual_quantity,
                actual_trigger,
                "protective stop quantity, trigger, side, or state mismatch",
            )
        self._resolve_issue("PROTECTION_MISMATCH", protection_id)
        self._resolve_issue("UNKNOWN_PROTECTION_SUBMISSION", protection_id)
        self._resolve_issue("PROTECTION_CONFLICT", protection_id)
        self._resolve_issue("PROTECTION_CANCEL_PENDING", protection_id)
        self._resolve_issue("PROTECTION_CANCEL_FAILED", protection_id)
        if symbol is not None:
            self._resolve_rearm_failures(symbol)
        if stored is not None and OrderState(stored["state"]) in {
            OrderState.CREATED,
            OrderState.SUBMITTING,
            OrderState.UNKNOWN,
        }:
            self.store.transition_protection(protection_id, OrderState.OPEN)
        return ProtectionVerification(
            protection_id=protection_id,
            exchange_order_id=exchange_order_id,
            expected_quantity=expected_quantity,
            actual_quantity=actual_quantity,
            expected_trigger_price=expected_trigger_price,
            actual_trigger_price=actual_trigger,
            valid=True,
        )

    def _protection_expectation(
        self,
        stored: Mapping[str, Any] | None,
        symbol: str | None,
        exchange_order_id: str | None,
        expected_quantity: Decimal | None,
        expected_trigger_price: Decimal | None,
    ) -> tuple[str, str | None, Decimal, Decimal]:
        if stored is not None:
            symbol = str(stored["symbol"])
            exchange_order_id = stored["exchange_order_id"]
            expected_quantity = self.store.get_protection_remaining_quantity(
                str(stored["protection_id"]),
            )
            expected_trigger_price = Decimal(str(stored["trigger_price"]))
        if not _exact_id(symbol):
            msg = "symbol is required for an unrecorded protection check"
            raise ValueError(msg)
        if expected_quantity is None or expected_trigger_price is None:
            msg = "expected quantity and trigger are required"
            raise ValueError(msg)
        quantity = _required_decimal("expected_quantity", expected_quantity, positive=True)
        trigger = _required_decimal(
            "expected_trigger_price",
            expected_trigger_price,
            positive=True,
        )
        return symbol, exchange_order_id, quantity, trigger

    def _protection_failure(  # noqa: PLR0913
        self,
        protection_id: str,
        exchange_order_id: str | None,
        expected_quantity: Decimal,
        expected_trigger_price: Decimal,
        actual_quantity: Decimal | None,
        actual_trigger_price: Decimal | None,
        error: str,
    ) -> ProtectionVerification:
        self.store.record_reconciliation_issue(
            "PROTECTION_MISMATCH",
            protection_id,
            {
                "actual_quantity": None if actual_quantity is None else str(actual_quantity),
                "actual_trigger_price": (
                    None if actual_trigger_price is None else str(actual_trigger_price)
                ),
                "error": error,
                "exchange_order_id": exchange_order_id,
                "expected_quantity": str(expected_quantity),
                "expected_trigger_price": str(expected_trigger_price),
            },
        )
        return ProtectionVerification(
            protection_id=protection_id,
            exchange_order_id=exchange_order_id,
            expected_quantity=expected_quantity,
            actual_quantity=actual_quantity,
            expected_trigger_price=expected_trigger_price,
            actual_trigger_price=actual_trigger_price,
            valid=False,
            error=error,
        )

    @property
    def ready(self) -> bool:
        """Return whether no unresolved exact-data issue blocks execution."""
        return not self.store.list_reconciliation_issues()

    @property
    def protection_remediation_only(self) -> bool:
        """Allow protection and exit remediation when only protection issues are open.

        The remediable kinds all describe an exchange-side protection or
        protected-exit gap on otherwise exact data. Blocking stop re-arming on
        these (as full readiness does) would leave a live position without an
        exchange stop indefinitely — the one failure mode worse than trading
        with an open issue.
        """
        issues = self.store.list_reconciliation_issues()
        remediable = {
            "MISSING_PROTECTION",
            "PROTECTION_REARM_FAILED",
            "PROTECTED_EXIT_HANDOFF_FAILED",
        }
        return bool(issues) and {str(issue["kind"]) for issue in issues} <= remediable

    def is_ready(self) -> bool:
        """Report readiness in method form for integration call sites."""
        return self.ready

    def readiness(self) -> ReadinessResult:
        """Return readiness with the unresolved issue evidence."""
        issues = tuple(self.store.list_reconciliation_issues())
        return ReadinessResult(not issues, issues)

    def _resolve_issue(self, kind: str, exact_key: str) -> None:
        for issue in self.store.list_reconciliation_issues():
            if issue["kind"] == kind and issue["exact_key"] == exact_key:
                self.store.resolve_reconciliation_issue(
                    int(issue["issue_id"]),
                    {"action": "exact exchange data reconciled"},
                )

    def _resolve_rearm_failures(self, symbol: str) -> None:
        """Resolve rearm failures for a symbol once its protection verifies exact.

        Re-arming submits a fresh deterministic protection id, so the failure
        recorded against the canceled protection's id can only be closed by
        symbol-level evidence: an exchange-verified stop now covers the position.
        """
        for issue in self.store.list_reconciliation_issues():
            if str(issue["kind"]) != "PROTECTION_REARM_FAILED":
                continue
            stale = self.store.get_protection(str(issue["exact_key"]))
            if stale is not None and str(stale["symbol"]) == symbol:
                self.store.resolve_reconciliation_issue(
                    int(issue["issue_id"]),
                    {"action": f"protective stop for {symbol} verified on exchange"},
                )

    def _resolve_exit_handoff_failures(self, symbol: str) -> None:
        """Resolve handoff failures for a symbol once a protected exit succeeds.

        Handoff issues are keyed by cycle-scoped intent ids, so a next-cycle
        retry can never resolve a prior cycle's record by exact key alone.
        """
        suffix = f":exit:{symbol}"
        for issue in self.store.list_reconciliation_issues():
            if str(issue["kind"]) == "PROTECTED_EXIT_HANDOFF_FAILED" and str(
                issue["exact_key"],
            ).endswith(suffix):
                self.store.resolve_reconciliation_issue(
                    int(issue["issue_id"]),
                    {"action": f"protected exit for {symbol} submitted exactly"},
                )


def _required_decimal(name: str, value: Decimal, *, positive: bool) -> Decimal:
    if not isinstance(value, Decimal):
        msg = f"{name} must be Decimal"
        raise TypeError(msg)
    if not value.is_finite():
        msg = f"{name} must be finite"
        raise ValueError(msg)
    if positive and value <= 0:
        msg = f"{name} must be positive"
        raise ValueError(msg)
    if not positive and value < 0:
        msg = f"{name} must not be negative"
        raise ValueError(msg)
    return value


def _external_decimal(name: str, value: object, *, positive: bool) -> Decimal:
    if isinstance(value, bool):
        msg = f"{name} must be numeric"
        raise TypeError(msg)
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        msg = f"{name} must be a finite decimal"
        raise ValueError(msg) from error
    if not parsed.is_finite():
        msg = f"{name} must be a finite decimal"
        raise ValueError(msg)
    if positive and parsed <= 0:
        msg = f"{name} must be positive"
        raise ValueError(msg)
    if not positive and parsed < 0:
        msg = f"{name} must not be negative"
        raise ValueError(msg)
    return parsed


def _remaining_order_quantity(order: Mapping[str, Any]) -> Decimal:
    """Read exact open quantity, falling back to amount minus filled."""
    if order.get("remaining") is not None:
        return _external_decimal("protection remaining", order["remaining"], positive=True)
    amount = _external_decimal("protection amount", order.get("amount"), positive=True)
    filled = _external_decimal("protection filled", order.get("filled", 0), positive=False)
    remaining = amount - filled
    return _required_decimal("protection remaining", remaining, positive=True)


def _timestamp(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{name} must be an integer"
        raise TypeError(msg)
    if value < 0:
        msg = f"{name} must not be negative"
        raise ValueError(msg)
    return value


def _exact_id(value: object) -> TypeGuard[str]:
    return isinstance(value, str) and bool(value) and value.strip() == value


def _order_client_id(order: Mapping[str, Any]) -> str | None:
    value = order.get("clientOrderId")
    if value is not None:
        return str(value)
    info = order.get("info")
    if isinstance(info, Mapping) and info.get("cl_ord_id") is not None:
        return str(info["cl_ord_id"])
    return None


def _is_ambiguous_mutation_error(error: Exception) -> bool:
    return isinstance(error, (ccxt.NetworkError, TimeoutError, ConnectionError, OSError))


def _fill_sort_key(fill: Mapping[str, Any]) -> tuple[int, str]:
    timestamp = fill.get("timestamp")
    fill_id = fill.get("id")
    return (
        timestamp if isinstance(timestamp, int) and not isinstance(timestamp, bool) else -1,
        fill_id if isinstance(fill_id, str) else "",
    )


def _recoverable_order_state(status: str) -> OrderState | None:
    if status in {"open", "pending", "partially_filled"}:
        return OrderState.OPEN
    if status in {"canceled", "cancelled", "expired"}:
        return OrderState.CANCELED
    if status == "rejected":
        return OrderState.REJECTED
    return None


def _protection_trigger(order: Mapping[str, Any]) -> Decimal:
    for key in ("stopLossPrice", "triggerPrice", "stopPrice"):
        value = order.get(key)
        if value is not None:
            return _external_decimal("protection trigger", value, positive=True)
    msg = "protective order has no exact trigger field"
    raise ValueError(msg)


def _is_stop_order(order: Mapping[str, Any]) -> bool:
    if order.get("stopLossPrice") is not None:
        return True
    if str(order.get("type") or "").lower() in {"stop-loss", "stop-loss-limit"}:
        return True
    info = order.get("info")
    if not isinstance(info, Mapping):
        return False
    description = info.get("descr")
    if not isinstance(description, Mapping):
        return False
    return description.get("ordertype") in {"stop-loss", "stop-loss-limit"}


def _exact_symbols(symbols: str | Iterable[str]) -> tuple[str, ...]:
    values = (symbols,) if isinstance(symbols, str) else tuple(symbols)
    if not values:
        msg = "at least one symbol is required"
        raise ValueError(msg)
    for symbol in values:
        if not _exact_id(symbol) or "/" not in symbol:
            msg = f"invalid exact symbol {symbol!r}"
            raise ValueError(msg)
    if len(set(values)) != len(values):
        msg = "symbols must not contain duplicates"
        raise ValueError(msg)
    return values


def _protection_client_order_id(protection: Mapping[str, Any]) -> str | None:
    payload = protection.get("payload")
    if not isinstance(payload, Mapping):
        return None
    client_order_id = payload.get("client_order_id")
    return client_order_id if _exact_id(client_order_id) else None


__all__ = [
    "BalanceComparison",
    "CancellationResult",
    "ExecutionCoordinator",
    "FillReconciliationResult",
    "OpenOrderReconciliationResult",
    "ProtectedExitResult",
    "ProtectionSubmissionResult",
    "ProtectionVerification",
    "ReadinessResult",
    "ReconciledFill",
    "RecoveryResult",
    "RecoverySummary",
    "RefreshedOrder",
    "SubmissionOutcome",
    "SubmissionResult",
]
