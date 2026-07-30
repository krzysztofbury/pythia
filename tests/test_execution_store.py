"""Tests for the isolated v10 execution store."""

from __future__ import annotations

import re
import sqlite3
from decimal import Decimal
from typing import TYPE_CHECKING

import pytest

from execution_store import (
    SCHEMA_VERSION,
    ExecutionStore,
    FillApplicationError,
    InvalidOrderTransitionError,
    OrderPurpose,
    OrderState,
    PayloadConflictError,
    ScopeMismatchError,
    UnscopedDatabaseError,
    canonical_decimal,
    deterministic_client_id,
)

if TYPE_CHECKING:
    from pathlib import Path

BUSY_TIMEOUT_MS = 5_000


def test_fresh_database_is_scope_bound_and_configured(tmp_path: Path) -> None:
    """A new store records scope and enables all required SQLite safeguards."""
    path = tmp_path / "execution.db"
    store = ExecutionStore(path, "live:kraken:account-a")

    metadata = store.connection.execute("SELECT * FROM store_metadata").fetchone()
    migration = store.connection.execute("SELECT * FROM schema_migrations").fetchone()
    assert metadata["scope"] == "live:kraken:account-a"
    assert metadata["schema_version"] == SCHEMA_VERSION
    assert migration["version"] == SCHEMA_VERSION
    assert store.connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert store.connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert store.connection.execute("PRAGMA busy_timeout").fetchone()[0] == BUSY_TIMEOUT_MS
    store.close()

    with pytest.raises(ScopeMismatchError):
        ExecutionStore(path, "paper:kraken:account-a")


def test_existing_unscoped_database_is_never_adopted(tmp_path: Path) -> None:
    """Legacy rows remain untouched when opening an unscoped database fails."""
    path = tmp_path / "legacy.db"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE trades(id INTEGER PRIMARY KEY, fill_source TEXT)")
    connection.execute("INSERT INTO trades(fill_source) VALUES ('bot')")
    connection.commit()
    connection.close()

    with pytest.raises(UnscopedDatabaseError, match="legacy migration is forbidden"):
        ExecutionStore(path, "live:kraken:account-a")

    connection = sqlite3.connect(path)
    assert connection.execute("SELECT COUNT(*) FROM trades").fetchone()[0] == 1
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE name = 'store_metadata'",
        ).fetchone()[0]
        == 0
    )
    connection.close()


def test_decimal_canonicalization_and_client_ids_are_strict() -> None:
    """Decimals and deterministic IDs have stable exact representations."""
    assert canonical_decimal(Decimal("100.5000")) == "100.5"
    assert canonical_decimal(Decimal("1E+3")) == "1000"
    assert canonical_decimal(Decimal("-0.000")) == "0"
    with pytest.raises(TypeError):
        canonical_decimal(0.1)  # type: ignore[arg-type]

    first = deterministic_client_id("scope-a", "intent-1", attempt=2)
    assert first == deterministic_client_id("scope-a", "intent-1", attempt=2)
    assert first != deterministic_client_id("scope-b", "intent-1", attempt=2)
    assert first != deterministic_client_id("scope-a", "intent-1", attempt=3)
    assert re.fullmatch(r"[0-9a-f]{32}", first)


def test_order_transitions_and_fill_derived_state(tmp_path: Path) -> None:
    """Only checked transitions occur and exact fills derive order progress."""
    store = ExecutionStore(tmp_path / "execution.db", "paper:kraken:a")
    client_id = store.create_order_intent(
        "intent-1",
        "BTC/USD",
        "BUY",
        Decimal("2.000"),
        purpose=OrderPurpose.ENTRY,
        limit_price=Decimal("100.00"),
    )
    assert (
        store.connection.execute(
            "SELECT quantity FROM order_intents WHERE intent_id = 'intent-1'",
        ).fetchone()[0]
        == "2"
    )

    order = store.record_order("intent-1", state=OrderState.SUBMITTING)
    assert order.client_order_id == client_id
    order = store.bind_exchange_order_id(order.order_id, "exchange-order-1")
    assert order.exchange_order_id == "exchange-order-1"
    order = store.transition_order(
        order.order_id,
        OrderState.OPEN,
        expected_state=OrderState.SUBMITTING,
    )
    assert order.state is OrderState.OPEN

    store.apply_fill(
        "fill-1",
        "BTC/USD",
        "BUY",
        Decimal(100),
        Decimal("0.5"),
        executed_at_ms=20,
        order_id=order.order_id,
    )
    order = store.get_order(order.order_id)
    assert order.state is OrderState.PARTIALLY_FILLED
    assert order.filled_quantity == Decimal("0.5")

    with pytest.raises(InvalidOrderTransitionError):
        store.transition_order(order.order_id, OrderState.CREATED)

    store.apply_fill(
        "fill-2",
        "BTC/USD",
        "BUY",
        Decimal(110),
        Decimal("1.5"),
        executed_at_ms=30,
        order_id=order.order_id,
    )
    order = store.get_order(order.order_id)
    assert order.state is OrderState.FILLED
    assert order.average_fill_price == Decimal("107.5")


def test_fill_idempotency_conflict_and_out_of_order_rebuild(tmp_path: Path) -> None:
    """Historical insertion rebuilds positions while fill IDs remain exact."""
    store = ExecutionStore(tmp_path / "execution.db", "paper:kraken:a")
    late = store.apply_fill(
        "fill-late",
        "ETH/USD",
        "BUY",
        Decimal(120),
        Decimal(1),
        executed_at_ms=20,
        payload={"trade": "late"},
    )
    assert late.inserted is True

    early = store.apply_fill(
        "fill-early",
        "ETH/USD",
        "BUY",
        Decimal(100),
        Decimal(1),
        executed_at_ms=10,
        payload={"trade": "early"},
    )
    assert early.position is not None
    assert early.position.quantity == Decimal(2)
    assert early.position.average_cost == Decimal(110)
    assert early.epoch_id != late.epoch_id

    duplicate = store.apply_fill(
        "fill-early",
        "ETH/USD",
        "BUY",
        Decimal("100.0"),
        Decimal("1.00"),
        executed_at_ms=10,
        payload={"trade": "early"},
    )
    assert duplicate.inserted is False
    expected_fill_count = 2
    stored_fill_count = store.connection.execute("SELECT COUNT(*) FROM fills").fetchone()[0]
    assert stored_fill_count == expected_fill_count

    with pytest.raises(PayloadConflictError):
        store.apply_fill(
            "fill-early",
            "ETH/USD",
            "BUY",
            Decimal(101),
            Decimal(1),
            executed_at_ms=10,
            payload={"trade": "early"},
        )


def test_partial_fills_flat_epochs_and_oversell_rollback(tmp_path: Path) -> None:
    """Weighted average accounting spans partial fills and flat epochs atomically."""
    store = ExecutionStore(tmp_path / "execution.db", "paper:kraken:a")
    store.apply_fill("b1", "SOL/USD", "BUY", "100", "1", executed_at_ms=10)
    store.apply_fill("b2", "SOL/USD", "BUY", "120", "1", executed_at_ms=20)
    store.apply_fill("s1", "SOL/USD", "SELL", "130", "0.5", executed_at_ms=30)
    position = store.get_position("SOL/USD")
    assert position is not None
    assert position.quantity == Decimal("1.5")
    assert position.average_cost == Decimal(110)
    assert position.realized_pnl == Decimal(10)

    store.apply_fill("s2", "SOL/USD", "SELL", "90", "1.5", executed_at_ms=40)
    assert store.get_position("SOL/USD") is None
    store.apply_fill("b3", "SOL/USD", "BUY", "80", "2", executed_at_ms=50)
    epochs = store.list_position_epochs("SOL/USD")
    assert [epoch["status"] for epoch in epochs] == ["CLOSED", "OPEN"]
    assert [epoch["sequence"] for epoch in epochs] == [1, 2]

    with pytest.raises(FillApplicationError, match="sells"):
        store.apply_fill("bad-sell", "SOL/USD", "SELL", "90", "3", executed_at_ms=60)
    assert (
        store.connection.execute(
            "SELECT COUNT(*) FROM fills WHERE fill_id = 'bad-sell'",
        ).fetchone()[0]
        == 0
    )
    position = store.get_position("SOL/USD")
    assert position is not None
    assert position.quantity == Decimal(2)


def test_quote_fees_are_included_in_cost_basis_and_realized_pnl(tmp_path: Path) -> None:
    """Accounting uses actual fees rather than optimistic gross prices."""
    store = ExecutionStore(tmp_path / "execution.db", "paper:kraken:a")
    store.apply_fill(
        "buy",
        "BTC/USD",
        "BUY",
        "100",
        "1",
        fee="1",
        fee_currency="USD",
        executed_at_ms=10,
    )
    position = store.get_position("BTC/USD")
    assert position is not None
    assert position.cost_basis == Decimal(101)
    assert position.average_cost == Decimal(101)

    store.apply_fill(
        "sell",
        "BTC/USD",
        "SELL",
        "110",
        "1",
        fee="1",
        fee_currency="USD",
        executed_at_ms=20,
    )
    assert store.get_position("BTC/USD") is None
    epoch = store.list_position_epochs("BTC/USD")[0]
    assert Decimal(epoch["realized_pnl"]) == Decimal(8)


def test_authoritative_fill_cost_drives_accounting(tmp_path: Path) -> None:
    """Kraken's exact trade cost wins over a rounded price-times-volume product."""
    store = ExecutionStore(tmp_path / "execution.db", "paper:kraken:a")

    store.apply_fill(
        "buy",
        "BTC/USD",
        "BUY",
        "100.004",
        "1",
        cost="100.01",
        executed_at_ms=10,
    )

    position = store.get_position("BTC/USD")
    assert position is not None
    assert position.cost_basis == Decimal("100.01")


def test_protection_cursor_and_reconciliation_issue_storage(tmp_path: Path) -> None:
    """Protection, cursor, and issue records use explicit exact keys."""
    store = ExecutionStore(tmp_path / "execution.db", "live:kraken:a")
    result = store.apply_fill("buy-1", "BTC/USD", "BUY", "100", "1", executed_at_ms=1)
    store.record_protection(
        "stop-1",
        "BTC/USD",
        "STOP_LOSS",
        "1.0",
        "90.00",
        epoch_id=result.epoch_id,
    )
    stored = store.connection.execute(
        "SELECT quantity, trigger_price FROM protection_orders WHERE protection_id = 'stop-1'",
    ).fetchone()
    assert tuple(stored) == ("1", "90")

    assert store.get_sync_cursor("fills:BTC/USD") is None
    store.set_sync_cursor("fills:BTC/USD", "exact-cursor-1")
    assert store.get_sync_cursor("fills:BTC/USD") == "exact-cursor-1"

    issue_id = store.record_reconciliation_issue(
        "UNKNOWN_FILL",
        "exchange-fill-7",
        {"fill_id": "exchange-fill-7"},
    )
    assert issue_id == store.record_reconciliation_issue(
        "UNKNOWN_FILL",
        "exchange-fill-7",
        {"fill_id": "exchange-fill-7", "observed": 2},
    )
    observed_count = 2
    assert store.list_reconciliation_issues()[0]["details"]["observed"] == observed_count
    store.resolve_reconciliation_issue(issue_id, {"action": "accepted"})
    assert store.list_reconciliation_issues() == []
    assert store.list_reconciliation_issues(status="RESOLVED")[0]["resolution"] == {
        "action": "accepted",
    }
