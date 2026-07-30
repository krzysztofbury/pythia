"""Scope-bound v10 execution persistence with exact, Decimal-safe accounting."""

from __future__ import annotations

import contextlib
import dataclasses
import datetime as dt
import hashlib
import json
import sqlite3
import threading
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping


SCHEMA_VERSION = 10
STORE_FORMAT = "oracle-pythia.execution"
APPLICATION_ID = 0x4F505931


class ExecutionStoreError(RuntimeError):
    """Base error for v10 execution persistence."""


class UnscopedDatabaseError(ExecutionStoreError):
    """Raised instead of guessing how a legacy or foreign database should map."""


class ScopeMismatchError(ExecutionStoreError):
    """Raised when a v10 database belongs to a different execution scope."""


class SchemaMismatchError(ExecutionStoreError):
    """Raised when an existing database is not exactly the supported v10 schema."""


class PayloadConflictError(ExecutionStoreError):
    """Raised when an exact identifier is reused for different immutable data."""


class InvalidOrderTransitionError(ExecutionStoreError):
    """Raised when an order state transition is not in the checked state graph."""


class FillApplicationError(ExecutionStoreError):
    """Raised when a fill cannot produce valid long-only position accounting."""


class Side(StrEnum):
    """Supported spot execution sides."""

    BUY = "BUY"
    SELL = "SELL"


class OrderPurpose(StrEnum):
    """Why an order exists; cleanup and recovery depend on this distinction."""

    ENTRY = "ENTRY"
    DCA = "DCA"
    EXIT = "EXIT"
    PROTECTION = "PROTECTION"


class OrderState(StrEnum):
    """Persisted order lifecycle states."""

    CREATED = "CREATED"
    SUBMITTING = "SUBMITTING"
    OPEN = "OPEN"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCEL_PENDING = "CANCEL_PENDING"
    CANCELED = "CANCELED"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


_ORDER_TRANSITIONS: dict[OrderState, frozenset[OrderState]] = {
    OrderState.CREATED: frozenset(
        {
            OrderState.SUBMITTING,
            OrderState.OPEN,
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
            OrderState.CANCELED,
            OrderState.REJECTED,
            OrderState.FAILED,
        },
    ),
    OrderState.SUBMITTING: frozenset(
        {
            OrderState.OPEN,
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
            OrderState.CANCELED,
            OrderState.REJECTED,
            OrderState.FAILED,
            OrderState.UNKNOWN,
        },
    ),
    OrderState.OPEN: frozenset(
        {
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
            OrderState.CANCEL_PENDING,
            OrderState.CANCELED,
            OrderState.REJECTED,
            OrderState.UNKNOWN,
        },
    ),
    OrderState.PARTIALLY_FILLED: frozenset(
        {
            OrderState.FILLED,
            OrderState.CANCEL_PENDING,
            OrderState.CANCELED,
            OrderState.UNKNOWN,
        },
    ),
    OrderState.CANCEL_PENDING: frozenset(
        {
            OrderState.OPEN,
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
            OrderState.CANCELED,
            OrderState.UNKNOWN,
        },
    ),
    # Exact fills can legitimately arrive after a cancel acknowledgement.
    OrderState.CANCELED: frozenset(
        {OrderState.PARTIALLY_FILLED, OrderState.FILLED},
    ),
    OrderState.UNKNOWN: frozenset(
        {
            OrderState.OPEN,
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
            OrderState.CANCEL_PENDING,
            OrderState.CANCELED,
            OrderState.REJECTED,
            OrderState.FAILED,
        },
    ),
    OrderState.FILLED: frozenset(),
    OrderState.REJECTED: frozenset(),
    OrderState.FAILED: frozenset(),
}


@dataclasses.dataclass(frozen=True)
class StoredOrder:
    """Current persisted order state."""

    order_id: int
    intent_id: str
    client_order_id: str
    exchange_order_id: str | None
    state: OrderState
    requested_quantity: Decimal
    filled_quantity: Decimal
    average_fill_price: Decimal | None


@dataclasses.dataclass(frozen=True)
class PositionSnapshot:
    """Current open weighted-average position for one symbol."""

    epoch_id: str
    symbol: str
    sequence: int
    quantity: Decimal
    cost_basis: Decimal
    average_cost: Decimal
    realized_pnl: Decimal
    opened_at_ms: int


@dataclasses.dataclass(frozen=True)
class FillResult:
    """Result of an idempotent fill application."""

    inserted: bool
    fill_id: str
    epoch_id: str
    position: PositionSnapshot | None


@dataclasses.dataclass
class _EpochCalculation:
    epoch_id: str
    sequence: int
    opening_fill_id: str
    opened_at_ms: int
    closing_fill_id: str | None = None
    closed_at_ms: int | None = None
    quantity: Decimal = Decimal(0)
    cost_basis: Decimal = Decimal(0)
    realized_pnl: Decimal = Decimal(0)


@dataclasses.dataclass(frozen=True)
class _EventCalculation:
    fill_id: str
    epoch_id: str
    quantity_before: Decimal
    quantity_after: Decimal
    average_cost_before: Decimal
    average_cost_after: Decimal
    realized_pnl: Decimal
    executed_at_ms: int


_ORDER_STATES_SQL = ", ".join(f"'{state.value}'" for state in OrderState)

_SCHEMA_STATEMENTS = (
    "CREATE TABLE schema_migrations ("
    "version INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, applied_at TEXT NOT NULL"
    ") STRICT",
    "CREATE TABLE store_metadata ("
    "singleton INTEGER PRIMARY KEY CHECK(singleton = 1), "
    "format TEXT NOT NULL, schema_version INTEGER NOT NULL, scope TEXT NOT NULL, "
    "created_at TEXT NOT NULL"
    ") STRICT",
    "CREATE TABLE order_intents ("
    "intent_id TEXT PRIMARY KEY, client_order_id TEXT NOT NULL UNIQUE, "
    "symbol TEXT NOT NULL, side TEXT NOT NULL CHECK(side IN ('BUY', 'SELL')), "
    "purpose TEXT NOT NULL CHECK(purpose IN ('ENTRY', 'DCA', 'EXIT', 'PROTECTION')), "
    "order_type TEXT NOT NULL, quantity TEXT NOT NULL, limit_price TEXT, "
    "attempt INTEGER NOT NULL CHECK(attempt >= 0), payload_json TEXT NOT NULL, "
    "payload_hash TEXT NOT NULL, created_at TEXT NOT NULL"
    ") STRICT",
    "CREATE TABLE orders ("
    "order_id INTEGER PRIMARY KEY, intent_id TEXT NOT NULL UNIQUE "
    "REFERENCES order_intents(intent_id) ON DELETE RESTRICT, "
    "client_order_id TEXT NOT NULL UNIQUE, exchange_order_id TEXT UNIQUE, "
    f"state TEXT NOT NULL CHECK(state IN ({_ORDER_STATES_SQL})), "
    "requested_quantity TEXT NOT NULL, filled_quantity TEXT NOT NULL, "
    "average_fill_price TEXT, payload_json TEXT NOT NULL, payload_hash TEXT NOT NULL, "
    "created_at TEXT NOT NULL, updated_at TEXT NOT NULL"
    ") STRICT",
    "CREATE TABLE order_state_events ("
    "event_id INTEGER PRIMARY KEY, order_id INTEGER NOT NULL "
    "REFERENCES orders(order_id) ON DELETE CASCADE, old_state TEXT, "
    f"new_state TEXT NOT NULL CHECK(new_state IN ({_ORDER_STATES_SQL})), "
    "payload_json TEXT NOT NULL, occurred_at TEXT NOT NULL"
    ") STRICT",
    "CREATE TABLE fills ("
    "fill_id TEXT PRIMARY KEY, order_id INTEGER REFERENCES orders(order_id) ON DELETE RESTRICT, "
    "protection_id TEXT REFERENCES protection_orders(protection_id) ON DELETE RESTRICT, "
    "symbol TEXT NOT NULL, side TEXT NOT NULL CHECK(side IN ('BUY', 'SELL')), "
    "price TEXT NOT NULL, quantity TEXT NOT NULL, cost TEXT NOT NULL, "
    "fee TEXT NOT NULL, fee_currency TEXT, "
    "executed_at_ms INTEGER NOT NULL CHECK(executed_at_ms >= 0), "
    "payload_json TEXT NOT NULL, payload_hash TEXT NOT NULL, received_at TEXT NOT NULL"
    ") STRICT",
    "CREATE INDEX fills_symbol_order ON fills(symbol, executed_at_ms, fill_id)",
    "CREATE TABLE position_epochs ("
    "epoch_id TEXT PRIMARY KEY, symbol TEXT NOT NULL, sequence INTEGER NOT NULL, "
    "opening_fill_id TEXT NOT NULL REFERENCES fills(fill_id) ON DELETE RESTRICT, "
    "closing_fill_id TEXT REFERENCES fills(fill_id) ON DELETE RESTRICT, "
    "opened_at_ms INTEGER NOT NULL, closed_at_ms INTEGER, "
    "status TEXT NOT NULL CHECK(status IN ('OPEN', 'CLOSED')), "
    "quantity TEXT NOT NULL, cost_basis TEXT NOT NULL, average_cost TEXT NOT NULL, "
    "realized_pnl TEXT NOT NULL, UNIQUE(symbol, sequence)"
    ") STRICT",
    "CREATE TABLE position_events ("
    "fill_id TEXT PRIMARY KEY REFERENCES fills(fill_id) ON DELETE CASCADE, "
    "epoch_id TEXT NOT NULL REFERENCES position_epochs(epoch_id) "
    "ON DELETE CASCADE ON UPDATE CASCADE, event_index INTEGER NOT NULL, "
    "quantity_before TEXT NOT NULL, quantity_after TEXT NOT NULL, "
    "average_cost_before TEXT NOT NULL, average_cost_after TEXT NOT NULL, "
    "realized_pnl TEXT NOT NULL, executed_at_ms INTEGER NOT NULL, "
    "UNIQUE(epoch_id, event_index)"
    ") STRICT",
    "CREATE TABLE protection_orders ("
    "protection_id TEXT PRIMARY KEY, epoch_id TEXT REFERENCES position_epochs(epoch_id) "
    "ON DELETE SET NULL ON UPDATE CASCADE, symbol TEXT NOT NULL, "
    "kind TEXT NOT NULL CHECK(kind IN ('STOP_LOSS', 'TAKE_PROFIT')), "
    f"state TEXT NOT NULL CHECK(state IN ({_ORDER_STATES_SQL})), "
    "quantity TEXT NOT NULL, trigger_price TEXT NOT NULL, exchange_order_id TEXT UNIQUE, "
    "payload_json TEXT NOT NULL, payload_hash TEXT NOT NULL, "
    "created_at TEXT NOT NULL, updated_at TEXT NOT NULL"
    ") STRICT",
    "CREATE TABLE protection_amendments ("
    "amendment_id INTEGER PRIMARY KEY, protection_id TEXT NOT NULL "
    "REFERENCES protection_orders(protection_id) ON DELETE CASCADE, "
    "prior_quantity TEXT NOT NULL, new_quantity TEXT NOT NULL, "
    "prior_trigger_price TEXT NOT NULL, new_trigger_price TEXT NOT NULL, "
    "exchange_amend_id TEXT, amended_at TEXT NOT NULL"
    ") STRICT",
    "CREATE TABLE sync_cursors ("
    "stream TEXT PRIMARY KEY, cursor TEXT NOT NULL, updated_at TEXT NOT NULL"
    ") STRICT",
    "CREATE TABLE reconciliation_issues ("
    "issue_id INTEGER PRIMARY KEY, kind TEXT NOT NULL, exact_key TEXT NOT NULL, "
    "status TEXT NOT NULL CHECK(status IN ('OPEN', 'RESOLVED')), details_json TEXT NOT NULL, "
    "detected_at TEXT NOT NULL, last_seen_at TEXT NOT NULL, resolved_at TEXT, "
    "resolution_json TEXT"
    ") STRICT",
    "CREATE UNIQUE INDEX reconciliation_open_exact "
    "ON reconciliation_issues(kind, exact_key) WHERE status = 'OPEN'",
)

_REQUIRED_TABLES = frozenset(
    {
        "fills",
        "order_intents",
        "order_state_events",
        "orders",
        "position_epochs",
        "position_events",
        "protection_amendments",
        "protection_orders",
        "reconciliation_issues",
        "schema_migrations",
        "store_metadata",
        "sync_cursors",
    },
)


def canonical_decimal(value: Decimal | int | str) -> str:
    """Return a finite Decimal as a canonical, non-exponent string."""
    if isinstance(value, (bool, float)):
        message = "decimal values must be Decimal, int, or string; floats are not accepted"
        raise TypeError(message)
    try:
        decimal_value = value if isinstance(value, Decimal) else Decimal(value)
    except (InvalidOperation, ValueError) as error:
        message = f"invalid decimal value: {value!r}"
        raise ValueError(message) from error
    if not decimal_value.is_finite():
        message = "decimal value must be finite"
        raise ValueError(message)
    if decimal_value == 0:
        return "0"
    canonical = format(decimal_value, "f")
    if "." in canonical:
        canonical = canonical.rstrip("0").rstrip(".")
    return canonical


def deterministic_client_id(scope: str, intent_id: str, *, attempt: int = 0) -> str:
    """Build a stable Kraken short-UUID client ID from exact intent values."""
    _validate_identifier("scope", scope)
    _validate_identifier("intent_id", intent_id)
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 0:
        message = "attempt must be a non-negative integer"
        raise ValueError(message)
    digest = _digest_parts(scope, intent_id, str(attempt))
    return digest[:32]


class ExecutionStore:
    """One exact execution scope backed by a fresh v10 SQLite database."""

    def __init__(
        self,
        database_path: str | Path,
        scope: str,
        *,
        busy_timeout_ms: int = 5_000,
    ) -> None:
        """Open an exact scope, creating only an empty database as fresh v10."""
        _validate_identifier("scope", scope)
        if busy_timeout_ms <= 0:
            message = "busy_timeout_ms must be positive"
            raise ValueError(message)
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.scope = scope
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self.database_path,
            isolation_level=None,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        try:
            self._configure(busy_timeout_ms)
            self._initialize_or_validate()
        except Exception:
            self._connection.close()
            raise

    @property
    def connection(self) -> sqlite3.Connection:
        """Expose the scoped connection for diagnostics and integrity tooling."""
        return self._connection

    def close(self) -> None:
        """Close the store connection."""
        with self._lock:
            self._connection.close()

    def __enter__(self) -> Self:
        """Return this open store."""
        return self

    def __exit__(self, *_exc: object) -> None:
        """Close the store on context exit."""
        self.close()

    def create_order_intent(  # noqa: PLR0913
        self,
        intent_id: str,
        symbol: str,
        side: Side | str,
        quantity: Decimal | int | str,
        *,
        purpose: OrderPurpose | str = OrderPurpose.ENTRY,
        order_type: str = "LIMIT",
        limit_price: Decimal | int | str | None = None,
        attempt: int = 0,
        payload: Mapping[str, Any] | None = None,
    ) -> str:
        """Persist an immutable intent and return its deterministic client ID."""
        _validate_identifier("intent_id", intent_id)
        _validate_identifier("symbol", symbol)
        _validate_identifier("order_type", order_type)
        parsed_side = Side(side)
        parsed_purpose = OrderPurpose(purpose)
        quantity_text = _positive_decimal("quantity", quantity)
        price_text = None if limit_price is None else _positive_decimal("limit_price", limit_price)
        client_order_id = deterministic_client_id(self.scope, intent_id, attempt=attempt)
        payload_json = _canonical_json(payload or {})
        identity = _canonical_json(
            {
                "attempt": attempt,
                "client_order_id": client_order_id,
                "limit_price": price_text,
                "order_type": order_type,
                "purpose": parsed_purpose.value,
                "payload": json.loads(payload_json),
                "quantity": quantity_text,
                "side": parsed_side.value,
                "symbol": symbol,
            },
        )
        payload_hash = _sha256(identity)
        now = _utc_now()
        with self._transaction():
            existing = self._connection.execute(
                "SELECT client_order_id, payload_hash FROM order_intents WHERE intent_id = ?",
                (intent_id,),
            ).fetchone()
            if existing is not None:
                if existing["payload_hash"] != payload_hash:
                    message = f"intent_id {intent_id!r} has different payload"
                    raise PayloadConflictError(message)
                return str(existing["client_order_id"])
            self._connection.execute(
                "INSERT INTO order_intents("
                "intent_id, client_order_id, symbol, side, purpose, order_type, quantity, "
                "limit_price, "
                "attempt, payload_json, payload_hash, created_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    intent_id,
                    client_order_id,
                    symbol,
                    parsed_side.value,
                    parsed_purpose.value,
                    order_type,
                    quantity_text,
                    price_text,
                    attempt,
                    payload_json,
                    payload_hash,
                    now,
                ),
            )
        return client_order_id

    def record_order(
        self,
        intent_id: str,
        *,
        exchange_order_id: str | None = None,
        state: OrderState | str = OrderState.CREATED,
        payload: Mapping[str, Any] | None = None,
    ) -> StoredOrder:
        """Create the single exchange-order record associated with an intent."""
        parsed_state = OrderState(state)
        if parsed_state in {OrderState.PARTIALLY_FILLED, OrderState.FILLED}:
            message = "a new order cannot be created in a filled state without fills"
            raise ValueError(message)
        if exchange_order_id is not None:
            _validate_identifier("exchange_order_id", exchange_order_id)
        payload_json = _canonical_json(payload or {})
        identity_hash = _sha256(
            _canonical_json(
                {
                    "exchange_order_id": exchange_order_id,
                    "payload": json.loads(payload_json),
                },
            ),
        )
        now = _utc_now()
        with self._transaction():
            existing = self._order_for_intent(intent_id)
            if existing is not None:
                if existing["payload_hash"] != identity_hash:
                    message = f"intent_id {intent_id!r} already has an order"
                    raise PayloadConflictError(message)
                return self._stored_order(existing)
            intent = self._connection.execute(
                "SELECT client_order_id, quantity FROM order_intents WHERE intent_id = ?",
                (intent_id,),
            ).fetchone()
            if intent is None:
                message = f"unknown intent_id {intent_id!r}"
                raise KeyError(message)
            cursor = self._connection.execute(
                "INSERT INTO orders("
                "intent_id, client_order_id, exchange_order_id, state, requested_quantity, "
                "filled_quantity, average_fill_price, payload_json, payload_hash, "
                "created_at, updated_at"
                ") VALUES (?, ?, ?, ?, ?, '0', NULL, ?, ?, ?, ?)",
                (
                    intent_id,
                    intent["client_order_id"],
                    exchange_order_id,
                    parsed_state.value,
                    intent["quantity"],
                    payload_json,
                    identity_hash,
                    now,
                    now,
                ),
            )
            if cursor.lastrowid is None:
                message = "SQLite did not return an order row ID"
                raise SchemaMismatchError(message)
            order_id = cursor.lastrowid
            self._insert_order_event(order_id, None, parsed_state, payload_json, now)
            row = self._order_by_id(order_id)
        return self._stored_order(row)

    def bind_exchange_order_id(self, order_id: int, exchange_order_id: str) -> StoredOrder:
        """Bind the exact exchange ID once after a placement response arrives."""
        _validate_identifier("exchange_order_id", exchange_order_id)
        with self._transaction():
            row = self._order_by_id(order_id)
            current = row["exchange_order_id"]
            if current is not None and current != exchange_order_id:
                message = f"order {order_id} already has exchange_order_id {current!r}"
                raise PayloadConflictError(message)
            self._connection.execute(
                "UPDATE orders SET exchange_order_id = ?, updated_at = ? WHERE order_id = ?",
                (exchange_order_id, _utc_now(), order_id),
            )
            updated = self._order_by_id(order_id)
        return self._stored_order(updated)

    def transition_order(
        self,
        order_id: int,
        new_state: OrderState | str,
        *,
        expected_state: OrderState | str | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> StoredOrder:
        """Apply a checked state transition, optionally as compare-and-set."""
        target = OrderState(new_state)
        expected = None if expected_state is None else OrderState(expected_state)
        payload_json = _canonical_json(payload or {})
        with self._transaction():
            row = self._order_by_id(order_id)
            current = OrderState(row["state"])
            if expected is not None and current != expected:
                message = f"order {order_id} is {current.value}, expected {expected.value}"
                raise InvalidOrderTransitionError(message)
            if current != target:
                self._checked_order_update(order_id, current, target, payload_json)
            updated = self._order_by_id(order_id)
        return self._stored_order(updated)

    def get_order(self, order_id: int) -> StoredOrder:
        """Return one order by its internal exact identifier."""
        with self._lock:
            return self._stored_order(self._order_by_id(order_id))

    def get_order_by_client_id(self, client_order_id: str) -> StoredOrder | None:
        """Find an order by the deterministic identity sent to the exchange."""
        _validate_identifier("client_order_id", client_order_id)
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM orders WHERE client_order_id = ?",
                (client_order_id,),
            ).fetchone()
        return None if row is None else self._stored_order(row)

    def get_order_by_exchange_id(self, exchange_order_id: str) -> StoredOrder | None:
        """Find an order by the exact exchange transaction identifier."""
        _validate_identifier("exchange_order_id", exchange_order_id)
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM orders WHERE exchange_order_id = ?",
                (exchange_order_id,),
            ).fetchone()
        return None if row is None else self._stored_order(row)

    def get_order_context(self, order_id: int) -> dict[str, str]:
        """Return immutable intent metadata needed by recovery and protection."""
        with self._lock:
            row = self._connection.execute(
                "SELECT intents.symbol, intents.side, intents.purpose, intents.order_type "
                "FROM orders JOIN order_intents AS intents USING(intent_id) "
                "WHERE orders.order_id = ?",
                (order_id,),
            ).fetchone()
        if row is None:
            message = f"unknown order_id {order_id}"
            raise KeyError(message)
        context_keys = ("symbol", "side", "purpose", "order_type")
        return {key: str(row[key]) for key in context_keys}

    def list_orders(self, *, active_only: bool = False) -> list[StoredOrder]:
        """List orders deterministically, optionally excluding terminal states."""
        query = "SELECT * FROM orders"
        params: tuple[object, ...] = ()
        if active_only:
            terminal = (
                OrderState.CANCELED.value,
                OrderState.FILLED.value,
                OrderState.REJECTED.value,
                OrderState.FAILED.value,
            )
            placeholders = ",".join("?" for _ in terminal)
            query += f" WHERE state NOT IN ({placeholders})"
            params = terminal
        query += " ORDER BY order_id"
        with self._lock:
            rows = self._connection.execute(query, params).fetchall()
        return [self._stored_order(row) for row in rows]

    def list_order_contexts(self, *, active_only: bool = False) -> list[dict[str, Any]]:
        """List orders with immutable symbol, side, and purpose in exact order."""
        query = (
            "SELECT orders.*, intents.symbol, intents.side, intents.purpose, "
            "intents.order_type FROM orders JOIN order_intents AS intents USING(intent_id)"
        )
        params: tuple[object, ...] = ()
        if active_only:
            terminal = (
                OrderState.CANCELED.value,
                OrderState.FILLED.value,
                OrderState.REJECTED.value,
                OrderState.FAILED.value,
            )
            placeholders = ",".join("?" for _ in terminal)
            query += f" WHERE orders.state NOT IN ({placeholders})"
            params = terminal
        query += " ORDER BY orders.order_id"
        with self._lock:
            rows = self._connection.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def apply_fill(  # noqa: PLR0913
        self,
        fill_id: str,
        symbol: str,
        side: Side | str,
        price: Decimal | int | str,
        quantity: Decimal | int | str,
        *,
        executed_at_ms: int,
        cost: Decimal | int | str | None = None,
        fee: Decimal | int | str = Decimal(0),
        fee_currency: str | None = None,
        order_id: int | None = None,
        protection_id: str | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> FillResult:
        """Insert an exact fill and atomically rebuild all accounting for its symbol."""
        _validate_identifier("fill_id", fill_id)
        _validate_identifier("symbol", symbol)
        if isinstance(executed_at_ms, bool) or not isinstance(executed_at_ms, int):
            message = "executed_at_ms must be an integer"
            raise TypeError(message)
        if executed_at_ms < 0:
            message = "executed_at_ms must not be negative"
            raise ValueError(message)
        parsed_side = Side(side)
        price_text = _positive_decimal("price", price)
        quantity_text = _positive_decimal("quantity", quantity)
        cost_text = (
            canonical_decimal(Decimal(price_text) * Decimal(quantity_text))
            if cost is None
            else _positive_decimal("cost", cost)
        )
        fee_text = _nonnegative_decimal("fee", fee)
        if fee_currency is not None:
            _validate_identifier("fee_currency", fee_currency)
        if protection_id is not None:
            _validate_identifier("protection_id", protection_id)
        if order_id is not None and protection_id is not None:
            message = "a fill cannot belong to both an order and a protection"
            raise ValueError(message)
        payload_json = _canonical_json(payload or {})
        payload_hash = self._fill_hash(
            order_id,
            protection_id,
            symbol,
            parsed_side,
            price_text,
            quantity_text,
            cost_text,
            fee_text,
            fee_currency,
            executed_at_ms,
            payload_json,
        )
        with self._transaction():
            duplicate = self._connection.execute(
                "SELECT payload_hash FROM fills WHERE fill_id = ?",
                (fill_id,),
            ).fetchone()
            if duplicate is not None:
                if duplicate["payload_hash"] != payload_hash:
                    message = f"fill_id {fill_id!r} has different payload"
                    raise PayloadConflictError(message)
                return self._fill_result(fill_id, inserted=False)
            self._validate_fill_owner(order_id, protection_id, symbol, parsed_side)
            self._connection.execute(
                "INSERT INTO fills("
                "fill_id, order_id, protection_id, symbol, side, price, quantity, cost, "
                "fee, fee_currency, "
                "executed_at_ms, payload_json, payload_hash, received_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    fill_id,
                    order_id,
                    protection_id,
                    symbol,
                    parsed_side.value,
                    price_text,
                    quantity_text,
                    cost_text,
                    fee_text,
                    fee_currency,
                    executed_at_ms,
                    payload_json,
                    payload_hash,
                    _utc_now(),
                ),
            )
            self._rebuild_position(symbol)
            if order_id is not None:
                self._rebuild_order_from_fills(order_id)
            return self._fill_result(fill_id, inserted=True)

    def get_position(self, symbol: str) -> PositionSnapshot | None:
        """Return the currently open epoch for a symbol, if any."""
        _validate_identifier("symbol", symbol)
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM position_epochs WHERE symbol = ? AND status = 'OPEN'",
                (symbol,),
            ).fetchone()
        return None if row is None else self._position_snapshot(row)

    def list_position_epochs(self, symbol: str) -> list[dict[str, Any]]:
        """Return all flat-to-flat epochs for a symbol in deterministic order."""
        _validate_identifier("symbol", symbol)
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM position_epochs WHERE symbol = ? ORDER BY sequence",
                (symbol,),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_open_positions(self) -> list[PositionSnapshot]:
        """Return every currently open position in symbol order."""
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM position_epochs WHERE status = 'OPEN' ORDER BY symbol",
            ).fetchall()
        return [self._position_snapshot(row) for row in rows]

    def get_daily_realized_pnl(self, *, now_ms: int | None = None) -> Decimal:
        """Return fee-inclusive realized P&L for the current UTC day."""
        now = (
            dt.datetime.now(dt.UTC)
            if now_ms is None
            else dt.datetime.fromtimestamp(
                now_ms / 1000,
                tz=dt.UTC,
            )
        )
        start = dt.datetime(now.year, now.month, now.day, tzinfo=dt.UTC)
        start_ms = int(start.timestamp() * 1000)
        end_ms = start_ms + 86_400_000
        with self._lock:
            rows = self._connection.execute(
                "SELECT realized_pnl FROM position_events "
                "WHERE executed_at_ms >= ? AND executed_at_ms < ?",
                (start_ms, end_ms),
            ).fetchall()
        return sum((Decimal(row[0]) for row in rows), start=Decimal(0))

    def record_protection(  # noqa: PLR0913
        self,
        protection_id: str,
        symbol: str,
        kind: str,
        quantity: Decimal | int | str,
        trigger_price: Decimal | int | str,
        *,
        epoch_id: str | None = None,
        exchange_order_id: str | None = None,
        state: OrderState | str = OrderState.CREATED,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        """Persist immutable protection-order identity and its initial state."""
        _validate_identifier("protection_id", protection_id)
        _validate_identifier("symbol", symbol)
        if kind not in {"STOP_LOSS", "TAKE_PROFIT"}:
            message = f"unsupported protection kind {kind!r}"
            raise ValueError(message)
        quantity_text = _positive_decimal("quantity", quantity)
        trigger_text = _positive_decimal("trigger_price", trigger_price)
        parsed_state = OrderState(state)
        payload_json = _canonical_json(payload or {})
        identity_hash = _sha256(
            _canonical_json(
                {
                    "epoch_id": epoch_id,
                    "exchange_order_id": exchange_order_id,
                    "kind": kind,
                    "payload": json.loads(payload_json),
                    "quantity": quantity_text,
                    "symbol": symbol,
                    "trigger_price": trigger_text,
                },
            ),
        )
        now = _utc_now()
        with self._transaction():
            existing = self._connection.execute(
                "SELECT payload_hash FROM protection_orders WHERE protection_id = ?",
                (protection_id,),
            ).fetchone()
            if existing is not None:
                if existing["payload_hash"] != identity_hash:
                    message = f"protection_id {protection_id!r} has different payload"
                    raise PayloadConflictError(message)
                return
            self._connection.execute(
                "INSERT INTO protection_orders("
                "protection_id, epoch_id, symbol, kind, state, quantity, trigger_price, "
                "exchange_order_id, payload_json, payload_hash, created_at, updated_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    protection_id,
                    epoch_id,
                    symbol,
                    kind,
                    parsed_state.value,
                    quantity_text,
                    trigger_text,
                    exchange_order_id,
                    payload_json,
                    identity_hash,
                    now,
                    now,
                ),
            )

    def list_protection(self, symbol: str, *, active_only: bool = False) -> list[dict[str, Any]]:
        """List known protective orders for exact recovery and coverage checks."""
        _validate_identifier("symbol", symbol)
        query = "SELECT * FROM protection_orders WHERE symbol = ?"
        params: list[object] = [symbol]
        if active_only:
            query += (
                " AND state IN ('CREATED', 'SUBMITTING', 'OPEN', 'PARTIALLY_FILLED', "
                "'CANCEL_PENDING', 'UNKNOWN')"
            )
        query += " ORDER BY protection_id"
        with self._lock:
            rows = self._connection.execute(query, params).fetchall()
        return [self._protection_record(row) for row in rows]

    def get_protection(self, protection_id: str) -> dict[str, Any] | None:
        """Return one protection record by its exact internal identifier."""
        _validate_identifier("protection_id", protection_id)
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM protection_orders WHERE protection_id = ?",
                (protection_id,),
            ).fetchone()
        return None if row is None else self._protection_record(row)

    def get_protection_by_exchange_id(self, exchange_order_id: str) -> dict[str, Any] | None:
        """Return one protection record by its exact exchange order identifier."""
        _validate_identifier("exchange_order_id", exchange_order_id)
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM protection_orders WHERE exchange_order_id = ?",
                (exchange_order_id,),
            ).fetchone()
        return None if row is None else self._protection_record(row)

    def get_protection_remaining_quantity(self, protection_id: str) -> Decimal:
        """Return original protection quantity minus exact linked stop fills."""
        _validate_identifier("protection_id", protection_id)
        with self._lock:
            row = self._connection.execute(
                "SELECT quantity FROM protection_orders WHERE protection_id = ?",
                (protection_id,),
            ).fetchone()
            if row is None:
                message = f"unknown protection {protection_id!r}"
                raise KeyError(message)
            filled = self._connection.execute(
                "SELECT quantity FROM fills WHERE protection_id = ?",
                (protection_id,),
            ).fetchall()
        remaining = Decimal(str(row["quantity"])) - sum(
            (Decimal(str(fill["quantity"])) for fill in filled),
            start=Decimal(0),
        )
        if remaining < 0:
            message = f"fills exceed protection quantity for {protection_id!r}"
            raise FillApplicationError(message)
        return remaining

    def amend_protection_expectation(
        self,
        protection_id: str,
        quantity: Decimal | int | str,
        trigger_price: Decimal | int | str,
        *,
        exchange_amend_id: str | None = None,
    ) -> None:
        """Record a verified in-place amendment and update current expectations."""
        _validate_identifier("protection_id", protection_id)
        quantity_text = _positive_decimal("quantity", quantity)
        trigger_text = _positive_decimal("trigger_price", trigger_price)
        if exchange_amend_id is not None:
            _validate_identifier("exchange_amend_id", exchange_amend_id)
        with self._transaction():
            row = self._connection.execute(
                "SELECT state, quantity, trigger_price FROM protection_orders "
                "WHERE protection_id = ?",
                (protection_id,),
            ).fetchone()
            if row is None:
                message = f"unknown protection {protection_id!r}"
                raise KeyError(message)
            if OrderState(row["state"]) not in {OrderState.OPEN, OrderState.PARTIALLY_FILLED}:
                message = f"protection {protection_id!r} is not amendable"
                raise InvalidOrderTransitionError(message)
            if Decimal(trigger_text) < Decimal(str(row["trigger_price"])):
                message = "protective stop amendments must not lower the trigger"
                raise ValueError(message)
            now = _utc_now()
            self._connection.execute(
                "INSERT INTO protection_amendments("
                "protection_id, prior_quantity, new_quantity, prior_trigger_price, "
                "new_trigger_price, exchange_amend_id, amended_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    protection_id,
                    row["quantity"],
                    quantity_text,
                    row["trigger_price"],
                    trigger_text,
                    exchange_amend_id,
                    now,
                ),
            )
            self._connection.execute(
                "UPDATE protection_orders SET quantity = ?, trigger_price = ?, updated_at = ? "
                "WHERE protection_id = ?",
                (quantity_text, trigger_text, now, protection_id),
            )

    @staticmethod
    def _protection_record(row: sqlite3.Row) -> dict[str, Any]:
        record = dict(row)
        record["payload"] = json.loads(str(record["payload_json"]))
        return record

    def transition_protection(
        self,
        protection_id: str,
        state: OrderState | str,
        *,
        exchange_order_id: str | None = None,
    ) -> None:
        """Update a known protection order after exchange confirmation."""
        _validate_identifier("protection_id", protection_id)
        parsed_state = OrderState(state)
        if exchange_order_id is not None:
            _validate_identifier("exchange_order_id", exchange_order_id)
        with self._transaction():
            row = self._connection.execute(
                "SELECT state, exchange_order_id FROM protection_orders WHERE protection_id = ?",
                (protection_id,),
            ).fetchone()
            if row is None:
                message = f"unknown protection_id {protection_id!r}"
                raise KeyError(message)
            current = OrderState(row["state"])
            if parsed_state != current and parsed_state not in _ORDER_TRANSITIONS[current]:
                message = (
                    f"protection {protection_id!r} cannot transition "
                    f"{current.value} -> {parsed_state.value}"
                )
                raise InvalidOrderTransitionError(message)
            current_exchange_id = row["exchange_order_id"]
            if (
                current_exchange_id is not None
                and exchange_order_id is not None
                and current_exchange_id != exchange_order_id
            ):
                message = f"protection {protection_id!r} already has a different exchange ID"
                raise PayloadConflictError(message)
            self._connection.execute(
                "UPDATE protection_orders SET state = ?, "
                "exchange_order_id = COALESCE(?, exchange_order_id), updated_at = ? "
                "WHERE protection_id = ?",
                (parsed_state.value, exchange_order_id, _utc_now(), protection_id),
            )

    def set_sync_cursor(self, stream: str, cursor: str) -> None:
        """Set an opaque cursor for one exact reconciliation stream."""
        _validate_identifier("stream", stream)
        _validate_identifier("cursor", cursor)
        with self._transaction():
            self._connection.execute(
                "INSERT INTO sync_cursors(stream, cursor, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(stream) DO UPDATE SET cursor=excluded.cursor, "
                "updated_at=excluded.updated_at",
                (stream, cursor, _utc_now()),
            )

    def get_sync_cursor(self, stream: str) -> str | None:
        """Return an opaque cursor without interpretation or fuzzy matching."""
        _validate_identifier("stream", stream)
        with self._lock:
            row = self._connection.execute(
                "SELECT cursor FROM sync_cursors WHERE stream = ?",
                (stream,),
            ).fetchone()
        return None if row is None else str(row["cursor"])

    def record_reconciliation_issue(
        self,
        kind: str,
        exact_key: str,
        details: Mapping[str, Any],
    ) -> int:
        """Record or exactly re-observe one unresolved reconciliation issue."""
        _validate_identifier("kind", kind)
        _validate_identifier("exact_key", exact_key)
        details_json = _canonical_json(details)
        now = _utc_now()
        with self._transaction():
            existing = self._connection.execute(
                "SELECT issue_id FROM reconciliation_issues "
                "WHERE kind = ? AND exact_key = ? AND status = 'OPEN'",
                (kind, exact_key),
            ).fetchone()
            if existing is not None:
                issue_id = int(existing["issue_id"])
                self._connection.execute(
                    "UPDATE reconciliation_issues SET details_json = ?, last_seen_at = ? "
                    "WHERE issue_id = ?",
                    (details_json, now, issue_id),
                )
                return issue_id
            cursor = self._connection.execute(
                "INSERT INTO reconciliation_issues("
                "kind, exact_key, status, details_json, detected_at, last_seen_at"
                ") VALUES (?, ?, 'OPEN', ?, ?, ?)",
                (kind, exact_key, details_json, now, now),
            )
            if cursor.lastrowid is None:
                message = "SQLite did not return a reconciliation issue row ID"
                raise SchemaMismatchError(message)
            return cursor.lastrowid

    def resolve_reconciliation_issue(
        self,
        issue_id: int,
        resolution: Mapping[str, Any],
    ) -> None:
        """Resolve one issue by exact numeric ID."""
        resolution_json = _canonical_json(resolution)
        with self._transaction():
            cursor = self._connection.execute(
                "UPDATE reconciliation_issues SET status = 'RESOLVED', resolved_at = ?, "
                "resolution_json = ? WHERE issue_id = ? AND status = 'OPEN'",
                (_utc_now(), resolution_json, issue_id),
            )
            if cursor.rowcount != 1:
                message = f"unknown open reconciliation issue {issue_id}"
                raise KeyError(message)

    def list_reconciliation_issues(self, *, status: str = "OPEN") -> list[dict[str, Any]]:
        """List issues in insertion order with decoded JSON details."""
        if status not in {"OPEN", "RESOLVED"}:
            message = "status must be OPEN or RESOLVED"
            raise ValueError(message)
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM reconciliation_issues WHERE status = ? ORDER BY issue_id",
                (status,),
            ).fetchall()
        issues = [dict(row) for row in rows]
        for issue in issues:
            issue["details"] = json.loads(issue.pop("details_json"))
            resolution_json = issue.pop("resolution_json")
            issue["resolution"] = None if resolution_json is None else json.loads(resolution_json)
        return issues

    def _configure(self, busy_timeout_ms: int) -> None:
        self._connection.execute(f"PRAGMA busy_timeout = {busy_timeout_ms}")
        self._connection.execute("PRAGMA foreign_keys = ON")
        mode = self._connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]
        if str(mode).lower() != "wal":
            message = f"SQLite refused WAL mode: {mode!r}"
            raise ExecutionStoreError(message)
        self._connection.execute("PRAGMA synchronous = FULL")

    def _initialize_or_validate(self) -> None:
        tables = {
            str(row[0])
            for row in self._connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'",
            )
        }
        if not tables:
            self._initialize_schema()
            return
        if "store_metadata" not in tables:
            message = "existing database has no v10 scope metadata; legacy migration is forbidden"
            raise UnscopedDatabaseError(message)
        row = self._connection.execute("SELECT * FROM store_metadata").fetchone()
        metadata_count = self._connection.execute(
            "SELECT COUNT(*) FROM store_metadata",
        ).fetchone()[0]
        if row is None or metadata_count != 1:
            message = "store_metadata must contain exactly one row"
            raise SchemaMismatchError(message)
        if row["format"] != STORE_FORMAT or row["schema_version"] != SCHEMA_VERSION:
            message = (
                f"expected {STORE_FORMAT} v{SCHEMA_VERSION}, "
                f"got {row['format']!r} v{row['schema_version']}"
            )
            raise SchemaMismatchError(message)
        if row["scope"] != self.scope:
            message = f"database scope is {row['scope']!r}, requested {self.scope!r}"
            raise ScopeMismatchError(message)
        if not _REQUIRED_TABLES.issubset(tables):
            missing = sorted(_REQUIRED_TABLES - tables)
            message = f"v10 database is missing tables: {missing}"
            raise SchemaMismatchError(message)
        migration = self._connection.execute(
            "SELECT name FROM schema_migrations WHERE version = ?",
            (SCHEMA_VERSION,),
        ).fetchone()
        if migration is None or migration["name"] != "fresh_execution_store":
            message = "v10 schema_migrations marker is missing or changed"
            raise SchemaMismatchError(message)
        if self._connection.execute("PRAGMA application_id").fetchone()[0] != APPLICATION_ID:
            message = "SQLite application_id does not identify an execution store"
            raise SchemaMismatchError(message)

    def _initialize_schema(self) -> None:
        now = _utc_now()
        with self._transaction():
            for statement in _SCHEMA_STATEMENTS:
                self._connection.execute(statement)
            self._connection.execute(f"PRAGMA application_id = {APPLICATION_ID}")
            self._connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            self._connection.execute(
                "INSERT INTO store_metadata(singleton, format, schema_version, scope, created_at) "
                "VALUES (1, ?, ?, ?, ?)",
                (STORE_FORMAT, SCHEMA_VERSION, self.scope, now),
            )
            self._connection.execute(
                "INSERT INTO schema_migrations(version, name, applied_at) VALUES (?, ?, ?)",
                (SCHEMA_VERSION, "fresh_execution_store", now),
            )

    @contextlib.contextmanager
    def _transaction(self) -> Iterator[None]:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield
            except Exception:
                self._connection.rollback()
                raise
            else:
                self._connection.commit()

    def _order_for_intent(self, intent_id: str) -> sqlite3.Row | None:
        return self._connection.execute(
            "SELECT * FROM orders WHERE intent_id = ?",
            (intent_id,),
        ).fetchone()

    def _order_by_id(self, order_id: int) -> sqlite3.Row:
        row = self._connection.execute(
            "SELECT * FROM orders WHERE order_id = ?",
            (order_id,),
        ).fetchone()
        if row is None:
            message = f"unknown order_id {order_id}"
            raise KeyError(message)
        return row

    @staticmethod
    def _stored_order(row: sqlite3.Row) -> StoredOrder:
        average = row["average_fill_price"]
        return StoredOrder(
            order_id=int(row["order_id"]),
            intent_id=str(row["intent_id"]),
            client_order_id=str(row["client_order_id"]),
            exchange_order_id=row["exchange_order_id"],
            state=OrderState(row["state"]),
            requested_quantity=Decimal(row["requested_quantity"]),
            filled_quantity=Decimal(row["filled_quantity"]),
            average_fill_price=None if average is None else Decimal(average),
        )

    def _insert_order_event(
        self,
        order_id: int,
        old_state: OrderState | None,
        new_state: OrderState,
        payload_json: str,
        occurred_at: str,
    ) -> None:
        self._connection.execute(
            "INSERT INTO order_state_events("
            "order_id, old_state, new_state, payload_json, occurred_at"
            ") VALUES (?, ?, ?, ?, ?)",
            (
                order_id,
                None if old_state is None else old_state.value,
                new_state.value,
                payload_json,
                occurred_at,
            ),
        )

    def _checked_order_update(
        self,
        order_id: int,
        current: OrderState,
        target: OrderState,
        payload_json: str,
    ) -> None:
        if target not in _ORDER_TRANSITIONS[current]:
            message = f"order {order_id} cannot transition {current.value} -> {target.value}"
            raise InvalidOrderTransitionError(message)
        now = _utc_now()
        self._connection.execute(
            "UPDATE orders SET state = ?, updated_at = ? WHERE order_id = ?",
            (target.value, now, order_id),
        )
        self._insert_order_event(order_id, current, target, payload_json, now)

    def _validate_fill_owner(
        self,
        order_id: int | None,
        protection_id: str | None,
        symbol: str,
        side: Side,
    ) -> None:
        if order_id is None and protection_id is None:
            return
        if order_id is not None:
            row = self._connection.execute(
                "SELECT intents.symbol, intents.side FROM orders "
                "JOIN order_intents AS intents USING(intent_id) WHERE orders.order_id = ?",
                (order_id,),
            ).fetchone()
            owner = f"order {order_id}"
        else:
            row = self._connection.execute(
                "SELECT symbol, 'SELL' AS side FROM protection_orders WHERE protection_id = ?",
                (protection_id,),
            ).fetchone()
            owner = f"protection {protection_id}"
        if row is None:
            message = f"unknown {owner}"
            raise KeyError(message)
        if row["symbol"] != symbol or row["side"] != side.value:
            message = f"fill symbol/side {symbol} {side.value} does not match {owner}"
            raise FillApplicationError(message)

    def _fill_hash(  # noqa: PLR0913
        self,
        order_id: int | None,
        protection_id: str | None,
        symbol: str,
        side: Side,
        price: str,
        quantity: str,
        cost: str,
        fee: str,
        fee_currency: str | None,
        executed_at_ms: int,
        payload_json: str,
    ) -> str:
        return _sha256(
            _canonical_json(
                {
                    "cost": cost,
                    "executed_at_ms": executed_at_ms,
                    "fee": fee,
                    "fee_currency": fee_currency,
                    "order_id": order_id,
                    "protection_id": protection_id,
                    "payload": json.loads(payload_json),
                    "price": price,
                    "quantity": quantity,
                    "side": side.value,
                    "symbol": symbol,
                },
            ),
        )

    def _fill_result(self, fill_id: str, *, inserted: bool) -> FillResult:
        event = self._connection.execute(
            "SELECT events.epoch_id, fills.symbol FROM position_events AS events "
            "JOIN fills USING(fill_id) WHERE events.fill_id = ?",
            (fill_id,),
        ).fetchone()
        if event is None:
            message = f"fill {fill_id!r} has no position event"
            raise SchemaMismatchError(message)
        return FillResult(
            inserted=inserted,
            fill_id=fill_id,
            epoch_id=str(event["epoch_id"]),
            position=self.get_position(str(event["symbol"])),
        )

    def _rebuild_position(self, symbol: str) -> None:
        rows = self._connection.execute(
            "SELECT fill_id, side, price, quantity, cost, fee, fee_currency, executed_at_ms "
            "FROM fills "
            "WHERE symbol = ? ORDER BY executed_at_ms, fill_id COLLATE BINARY",
            (symbol,),
        ).fetchall()
        epochs, events = self._calculate_position(symbol, rows)
        self._connection.execute(
            "DELETE FROM position_events WHERE epoch_id IN "
            "(SELECT epoch_id FROM position_epochs WHERE symbol = ?)",
            (symbol,),
        )
        existing_rows = self._connection.execute(
            "SELECT epoch_id, sequence FROM position_epochs WHERE symbol = ? ORDER BY sequence",
            (symbol,),
        ).fetchall()
        existing_by_sequence = {int(row["sequence"]): str(row["epoch_id"]) for row in existing_rows}
        for epoch in epochs:
            old_epoch_id = existing_by_sequence.get(epoch.sequence)
            if old_epoch_id is not None and old_epoch_id != epoch.epoch_id:
                self._connection.execute(
                    "UPDATE position_epochs SET epoch_id = ? WHERE epoch_id = ?",
                    (epoch.epoch_id, old_epoch_id),
                )
            self._upsert_epoch(symbol, epoch)
        desired_ids = {epoch.epoch_id for epoch in epochs}
        for row in existing_rows:
            current_id = existing_by_sequence[int(row["sequence"])]
            mapped = next(
                (epoch.epoch_id for epoch in epochs if epoch.sequence == int(row["sequence"])),
                current_id,
            )
            if mapped not in desired_ids:
                self._connection.execute(
                    "DELETE FROM position_epochs WHERE epoch_id = ?",
                    (mapped,),
                )
        for event in events:
            self._insert_position_event(event)

    def _calculate_position(
        self,
        symbol: str,
        rows: list[sqlite3.Row],
    ) -> tuple[list[_EpochCalculation], list[_EventCalculation]]:
        epochs: list[_EpochCalculation] = []
        events: list[_EventCalculation] = []
        epoch: _EpochCalculation | None = None
        base_currency, quote_currency = symbol.split("/", maxsplit=1)
        for row in rows:
            side = Side(row["side"])
            fill_quantity = Decimal(row["quantity"])
            fill_cost = Decimal(row["cost"])
            fee = Decimal(row["fee"])
            fee_currency = row["fee_currency"] or quote_currency
            if fee > 0 and fee_currency not in {base_currency, quote_currency}:
                message = (
                    f"fill {row['fill_id']!r} fee currency {fee_currency!r} does not match {symbol}"
                )
                raise FillApplicationError(message)
            if epoch is None:
                if side is Side.SELL:
                    message = f"fill {row['fill_id']!r} would make {symbol} inventory negative"
                    raise FillApplicationError(message)
                epoch = self._new_epoch(symbol, len(epochs) + 1, row)
            event = self._calculate_event(
                epoch,
                row,
                side,
                fill_quantity,
                fill_cost,
                fee,
                str(fee_currency),
                base_currency,
                quote_currency,
            )
            events.append(event)
            if epoch.quantity == 0:
                epoch.closing_fill_id = str(row["fill_id"])
                epoch.closed_at_ms = int(row["executed_at_ms"])
                epochs.append(epoch)
                epoch = None
        if epoch is not None:
            epochs.append(epoch)
        return epochs, events

    def _new_epoch(
        self,
        symbol: str,
        sequence: int,
        row: sqlite3.Row,
    ) -> _EpochCalculation:
        fill_id = str(row["fill_id"])
        epoch_id = f"pe_{_digest_parts(self.scope, symbol, fill_id)[:29]}"
        return _EpochCalculation(
            epoch_id=epoch_id,
            sequence=sequence,
            opening_fill_id=fill_id,
            opened_at_ms=int(row["executed_at_ms"]),
        )

    @staticmethod
    def _calculate_event(  # noqa: PLR0913
        epoch: _EpochCalculation,
        row: sqlite3.Row,
        side: Side,
        fill_quantity: Decimal,
        fill_cost: Decimal,
        fee: Decimal,
        fee_currency: str,
        base_currency: str,
        quote_currency: str,
    ) -> _EventCalculation:
        quantity_before = epoch.quantity
        average_before = Decimal(0) if quantity_before == 0 else epoch.cost_basis / quantity_before
        realized = Decimal(0)
        if side is Side.BUY:
            acquired_quantity = (
                fill_quantity - fee if fee_currency == base_currency else fill_quantity
            )
            if acquired_quantity <= 0:
                message = f"fill {row['fill_id']!r} base fee consumes the acquired quantity"
                raise FillApplicationError(message)
            epoch.quantity += acquired_quantity
            epoch.cost_basis += fill_cost
            if fee_currency == quote_currency:
                epoch.cost_basis += fee
        else:
            inventory_reduction = (
                fill_quantity + fee if fee_currency == base_currency else fill_quantity
            )
            if inventory_reduction > epoch.quantity:
                message = (
                    f"fill {row['fill_id']!r} sells {inventory_reduction} "
                    f"with only {epoch.quantity} held"
                )
                raise FillApplicationError(message)
            net_proceeds = fill_cost
            if fee_currency == quote_currency:
                net_proceeds -= fee
            realized = net_proceeds - (average_before * inventory_reduction)
            epoch.realized_pnl += realized
            epoch.quantity -= inventory_reduction
            epoch.cost_basis -= average_before * inventory_reduction
            if epoch.quantity == 0:
                epoch.cost_basis = Decimal(0)
        average_after = Decimal(0) if epoch.quantity == 0 else epoch.cost_basis / epoch.quantity
        return _EventCalculation(
            fill_id=str(row["fill_id"]),
            epoch_id=epoch.epoch_id,
            quantity_before=quantity_before,
            quantity_after=epoch.quantity,
            average_cost_before=average_before,
            average_cost_after=average_after,
            realized_pnl=realized,
            executed_at_ms=int(row["executed_at_ms"]),
        )

    def _upsert_epoch(self, symbol: str, epoch: _EpochCalculation) -> None:
        average = Decimal(0) if epoch.quantity == 0 else epoch.cost_basis / epoch.quantity
        status = "CLOSED" if epoch.closed_at_ms is not None else "OPEN"
        self._connection.execute(
            "INSERT INTO position_epochs("
            "epoch_id, symbol, sequence, opening_fill_id, closing_fill_id, opened_at_ms, "
            "closed_at_ms, status, quantity, cost_basis, average_cost, realized_pnl"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(epoch_id) DO UPDATE SET "
            "sequence=excluded.sequence, opening_fill_id=excluded.opening_fill_id, "
            "closing_fill_id=excluded.closing_fill_id, opened_at_ms=excluded.opened_at_ms, "
            "closed_at_ms=excluded.closed_at_ms, status=excluded.status, "
            "quantity=excluded.quantity, "
            "cost_basis=excluded.cost_basis, average_cost=excluded.average_cost, "
            "realized_pnl=excluded.realized_pnl",
            (
                epoch.epoch_id,
                symbol,
                epoch.sequence,
                epoch.opening_fill_id,
                epoch.closing_fill_id,
                epoch.opened_at_ms,
                epoch.closed_at_ms,
                status,
                canonical_decimal(epoch.quantity),
                canonical_decimal(epoch.cost_basis),
                canonical_decimal(average),
                canonical_decimal(epoch.realized_pnl),
            ),
        )

    def _insert_position_event(self, event: _EventCalculation) -> None:
        next_index = self._connection.execute(
            "SELECT COALESCE(MAX(event_index), 0) + 1 FROM position_events WHERE epoch_id = ?",
            (event.epoch_id,),
        ).fetchone()[0]
        self._connection.execute(
            "INSERT INTO position_events("
            "fill_id, epoch_id, event_index, quantity_before, quantity_after, "
            "average_cost_before, average_cost_after, realized_pnl, executed_at_ms"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event.fill_id,
                event.epoch_id,
                next_index,
                canonical_decimal(event.quantity_before),
                canonical_decimal(event.quantity_after),
                canonical_decimal(event.average_cost_before),
                canonical_decimal(event.average_cost_after),
                canonical_decimal(event.realized_pnl),
                event.executed_at_ms,
            ),
        )

    def _rebuild_order_from_fills(self, order_id: int) -> None:
        order = self._order_by_id(order_id)
        fills = self._connection.execute(
            "SELECT cost, quantity FROM fills WHERE order_id = ?",
            (order_id,),
        ).fetchall()
        filled_quantity = sum((Decimal(row["quantity"]) for row in fills), start=Decimal(0))
        requested_quantity = Decimal(order["requested_quantity"])
        if filled_quantity > requested_quantity:
            message = f"fills for order {order_id} exceed requested quantity {requested_quantity}"
            raise FillApplicationError(message)
        notional = sum(
            (Decimal(row["cost"]) for row in fills),
            start=Decimal(0),
        )
        average = notional / filled_quantity
        target = (
            OrderState.FILLED
            if filled_quantity == requested_quantity
            else OrderState.PARTIALLY_FILLED
        )
        current = OrderState(order["state"])
        if current != target and target not in _ORDER_TRANSITIONS[current]:
            message = f"fill cannot transition order {order_id} {current.value} -> {target.value}"
            raise InvalidOrderTransitionError(message)
        now = _utc_now()
        self._connection.execute(
            "UPDATE orders SET state = ?, filled_quantity = ?, average_fill_price = ?, "
            "updated_at = ? WHERE order_id = ?",
            (
                target.value,
                canonical_decimal(filled_quantity),
                canonical_decimal(average),
                now,
                order_id,
            ),
        )
        if current != target:
            self._insert_order_event(order_id, current, target, "{}", now)

    @staticmethod
    def _position_snapshot(row: sqlite3.Row) -> PositionSnapshot:
        return PositionSnapshot(
            epoch_id=str(row["epoch_id"]),
            symbol=str(row["symbol"]),
            sequence=int(row["sequence"]),
            quantity=Decimal(row["quantity"]),
            cost_basis=Decimal(row["cost_basis"]),
            average_cost=Decimal(row["average_cost"]),
            realized_pnl=Decimal(row["realized_pnl"]),
            opened_at_ms=int(row["opened_at_ms"]),
        )


def _positive_decimal(name: str, value: Decimal | int | str) -> str:
    canonical = canonical_decimal(value)
    if Decimal(canonical) <= 0:
        message = f"{name} must be positive"
        raise ValueError(message)
    return canonical


def _nonnegative_decimal(name: str, value: Decimal | int | str) -> str:
    canonical = canonical_decimal(value)
    if Decimal(canonical) < 0:
        message = f"{name} must not be negative"
        raise ValueError(message)
    return canonical


def _canonical_json(value: object) -> str:
    def decimal_default(item: object) -> str:
        if isinstance(item, Decimal):
            return canonical_decimal(item)
        message = f"value of type {type(item).__name__} is not JSON serializable"
        raise TypeError(message)

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=decimal_default,
    )


def _validate_identifier(name: str, value: str) -> None:
    if not isinstance(value, str):
        message = f"{name} must be a string"
        raise TypeError(message)
    if not value or value.strip() != value:
        message = f"{name} must be non-empty and have no surrounding whitespace"
        raise ValueError(message)


def _digest_parts(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        encoded = part.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return dt.datetime.now(tz=dt.UTC).isoformat(timespec="microseconds")
