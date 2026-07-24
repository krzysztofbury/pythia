"""Deterministic exact-order coordinator integration tests."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any

import ccxt

from execution_coordinator import ExecutionCoordinator, SubmissionOutcome
from execution_store import ExecutionStore, OrderState
from paper_exchange import MarketRules, PaperExchange

if TYPE_CHECKING:
    from pathlib import Path

ONE_SECOND_MS = 1_000


class FakeExchange:
    """Small deterministic exchange with controllable response loss and fill history."""

    def __init__(self) -> None:
        """Initialize empty deterministic exchange state."""
        self.orders: dict[str, dict[str, Any]] = {}
        self.fills: list[dict[str, Any]] = []
        self.balances: dict[str, Decimal] = {}
        self.mutation_calls: list[tuple[str, str]] = []
        self.cancelled: list[str] = []
        self.lose_next_response = False
        self.reject_next = False
        self.cancel_next = False
        self.amount_increment = Decimal("0.1")
        self.price_increment = Decimal("0.01")
        self._next_order = 1

    def normalize_limit_order(
        self,
        symbol: str,
        amount: Decimal,
        price: Decimal,
        *,
        stop_loss_price: Decimal | None = None,
    ) -> tuple[Decimal, Decimal, Decimal | None]:
        """Return fake venue-normalized limit inputs without mutation."""
        _ = symbol
        quantity = amount.quantize(self.amount_increment)
        limit = price.quantize(self.price_increment)
        stop = None if stop_loss_price is None else stop_loss_price.quantize(self.price_increment)
        return quantity, limit, stop

    def normalize_market_sell(
        self,
        symbol: str,
        amount: Decimal,
        *,
        reference_price: Decimal | None = None,
    ) -> tuple[Decimal, Decimal | None]:
        """Return fake venue-normalized market inputs without mutation."""
        _ = symbol
        quantity = amount.quantize(self.amount_increment)
        price = None if reference_price is None else reference_price.quantize(self.price_increment)
        return quantity, price

    def normalize_stop_loss(
        self,
        symbol: str,
        amount: Decimal,
        trigger_price: Decimal,
    ) -> tuple[Decimal, Decimal]:
        """Return fake venue-normalized stop inputs without mutation."""
        _ = symbol
        return amount.quantize(self.amount_increment), trigger_price.quantize(self.price_increment)

    def _place(  # noqa: PLR0913
        self,
        kind: str,
        symbol: str,
        amount: Decimal,
        client_order_id: str,
        *,
        side: str,
        price: Decimal | None,
    ) -> dict[str, Any]:
        self.mutation_calls.append((kind, client_order_id))
        if self.reject_next:
            self.reject_next = False
            message = "definite rejection"
            raise ccxt.InvalidOrder(message)
        order_id = f"exchange-{self._next_order}"
        self._next_order += 1
        order = {
            "id": order_id,
            "clientOrderId": client_order_id,
            "symbol": symbol,
            "side": side,
            "price": price,
            "amount": amount,
            "status": "open",
            "timestamp": 1_000,
        }
        self.orders[order_id] = order
        if self.cancel_next:
            self.cancel_next = False
            order["status"] = "canceled"
        if self.lose_next_response:
            self.lose_next_response = False
            message = "response lost after acceptance"
            raise ccxt.RequestTimeout(message)
        return dict(order)

    def place_maker_buy(
        self,
        symbol: str,
        price: Decimal,
        amount: Decimal,
        *,
        client_order_id: str,
        stop_loss_price: Decimal | None = None,
    ) -> dict[str, Any]:
        """Create one fake maker buy."""
        _ = stop_loss_price
        return self._place(
            "maker-buy",
            symbol,
            amount,
            client_order_id,
            side="buy",
            price=price,
        )

    def place_maker_sell(
        self,
        symbol: str,
        price: Decimal,
        amount: Decimal,
        *,
        client_order_id: str,
    ) -> dict[str, Any]:
        """Create one fake maker sell."""
        return self._place(
            "maker-sell",
            symbol,
            amount,
            client_order_id,
            side="sell",
            price=price,
        )

    def place_market_sell(
        self,
        symbol: str,
        amount: Decimal,
        *,
        client_order_id: str,
        reference_price: Decimal | None = None,
    ) -> dict[str, Any]:
        """Create one fake emergency sell."""
        return self._place(
            "market-sell",
            symbol,
            amount,
            client_order_id,
            side="sell",
            price=reference_price,
        )

    def place_stop_loss(
        self,
        symbol: str,
        amount: Decimal,
        trigger_price: Decimal,
        *,
        client_order_id: str,
    ) -> dict[str, Any]:
        """Create one exact fake protective stop."""
        order = self._place(
            "stop-loss",
            symbol,
            amount,
            client_order_id,
            side="sell",
            price=None,
        )
        order["type"] = "stop-loss"
        order["stopLossPrice"] = trigger_price
        self.orders[order["id"]].update(order)
        return order

    def amend_stop_loss(
        self,
        order_id: str,
        symbol: str,
        amount: Decimal,
        trigger_price: Decimal,
    ) -> dict[str, Any]:
        """Amend one fake retained stop in place."""
        order = self.fetch_order(order_id, symbol)
        self.mutation_calls.append(("amend-stop", order_id))
        self.orders[order_id]["amount"] = amount
        self.orders[order_id]["stopLossPrice"] = trigger_price
        return dict(order, id=f"amend-{order_id}")

    def fetch_account_fills(
        self,
        since_ms: int,
        end_ms: int,
        *,
        page_size: int,
        max_pages: int,
    ) -> list[dict[str, Any]]:
        """Return the fixed fake fill history."""
        self.last_fill_page_config = (page_size, max_pages)
        return [dict(fill) for fill in self.fills if since_ms <= fill["timestamp"] <= end_ms]

    def fetch_order(
        self,
        order_id: str,
        symbol: str | None = None,
        *,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        """Fetch one retained fake order exactly."""
        order = self.orders.get(order_id)
        if order is None or (symbol is not None and order["symbol"] != symbol):
            raise ccxt.OrderNotFound(order_id)
        if client_order_id is not None and order["clientOrderId"] != client_order_id:
            raise ccxt.OrderNotFound(client_order_id)
        return dict(order)

    def fetch_open_orders(
        self,
        symbol: str | None = None,
        *,
        client_order_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return exact matching open orders."""
        return [
            dict(order)
            for order in self.orders.values()
            if order["status"] == "open"
            and (symbol is None or order["symbol"] == symbol)
            and (client_order_id is None or order["clientOrderId"] == client_order_id)
        ]

    def fetch_closed_orders(
        self,
        symbol: str | None = None,
        *,
        client_order_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return exact matching terminal orders."""
        return [
            dict(order)
            for order in self.orders.values()
            if order["status"] != "open"
            and (symbol is None or order["symbol"] == symbol)
            and (client_order_id is None or order["clientOrderId"] == client_order_id)
        ]

    def cancel_order(
        self,
        order_id: str,
        symbol: str | None = None,
        *,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        """Cancel one exact fake order."""
        order = self.fetch_order(order_id, symbol, client_order_id=client_order_id)
        self.mutation_calls.append(("cancel", client_order_id or ""))
        self.orders[order_id]["status"] = "canceled"
        self.cancelled.append(order_id)
        order["status"] = "canceled"
        return order

    def get_total_balance(self, currency: str) -> Decimal:
        """Return one exact fake total balance."""
        return self.balances.get(currency, Decimal(0))


def _coordinator(tmp_path: Path) -> tuple[ExecutionCoordinator, ExecutionStore, FakeExchange]:
    store = ExecutionStore(tmp_path / "execution.db", "live:kraken:test")
    exchange = FakeExchange()
    coordinator = ExecutionCoordinator(
        store,
        exchange,  # type: ignore[arg-type]
        live_entries_enabled=True,
        max_fill_pages=3,
    )
    return coordinator, store, exchange


def _fill(
    fill_id: str,
    order_id: str,
    amount: str,
    timestamp: int,
    *,
    side: str = "buy",
) -> dict[str, Any]:
    return {
        "id": fill_id,
        "order": order_id,
        "timestamp": timestamp,
        "symbol": "BTC/USD",
        "side": side,
        "price": "100",
        "amount": amount,
        "fee": {"cost": "0.1", "currency": "USD"},
    }


def test_response_loss_is_unknown_never_retried_and_recovers_exactly(tmp_path: Path) -> None:
    """An accepted request with a lost response is found by exact clientOrderId."""
    coordinator, store, exchange = _coordinator(tmp_path)
    exchange.lose_next_response = True

    first = coordinator.submit_entry("entry-1", "BTC/USD", Decimal(1), Decimal(100))
    second = coordinator.submit_entry("entry-1", "BTC/USD", Decimal(1), Decimal(100))

    assert first.outcome is SubmissionOutcome.UNKNOWN
    assert first.state is OrderState.UNKNOWN
    assert second.outcome is SubmissionOutcome.UNKNOWN
    assert len(exchange.mutation_calls) == 1
    assert coordinator.ready is False

    recovery = coordinator.recover_unknown_orders()
    assert recovery[0].recovered is True
    assert recovery[0].state is OrderState.OPEN
    assert store.get_order(first.order_id or 0).exchange_order_id == "exchange-1"
    assert coordinator.ready is True


def test_definite_rejection_is_terminal_without_retry_or_issue(tmp_path: Path) -> None:
    """An exchange InvalidOrder is a definite rejection rather than UNKNOWN."""
    coordinator, store, exchange = _coordinator(tmp_path)
    exchange.reject_next = True

    result = coordinator.submit_dca("dca-1", "BTC/USD", Decimal(1), Decimal(100))

    assert result.outcome is SubmissionOutcome.REJECTED
    assert result.state is OrderState.REJECTED
    assert len(exchange.mutation_calls) == 1
    assert store.list_reconciliation_issues() == []


def test_partial_and_duplicate_fills_are_exact_and_idempotent(tmp_path: Path) -> None:
    """Partial fills derive state and replaying an exact fill does not double count."""
    coordinator, store, exchange = _coordinator(tmp_path)
    submission = coordinator.submit_entry("entry-2", "BTC/USD", Decimal(2), Decimal(100))
    assert submission.exchange_order_id is not None
    exchange.fills = [_fill("fill-1", submission.exchange_order_id, "1", 10)]

    partial = coordinator.reconcile_fills(since_ms=0, end_ms=20)
    duplicate = coordinator.reconcile_fills(since_ms=0, end_ms=20)

    assert partial.fills[0].inserted is True
    assert partial.fills[0].state is OrderState.PARTIALLY_FILLED
    assert duplicate.fills[0].inserted is False
    position = store.get_position("BTC/USD")
    assert position is not None
    assert position.quantity == Decimal(1)

    exchange.fills.append(_fill("fill-2", submission.exchange_order_id, "1", 30))
    complete = coordinator.reconcile_fills(since_ms=0, end_ms=40)
    assert complete.fills[-1].state is OrderState.FILLED
    assert store.get_order(submission.order_id or 0).filled_quantity == Decimal(2)


def test_unknown_fill_applies_inventory_advances_cursor_and_halts(tmp_path: Path) -> None:
    """An exact external fill updates inventory while its provenance keeps readiness false."""
    coordinator, store, exchange = _coordinator(tmp_path)
    exchange.fills = [_fill("external-fill", "external-order", "1", 10)]

    result = coordinator.reconcile_fills(since_ms=0, end_ms=20)

    assert result.complete is True
    assert result.unresolved_fill_ids == ()
    assert store.get_sync_cursor("fills:account") == "20"
    position = store.get_position("BTC/USD")
    assert position is not None
    assert position.quantity == Decimal(1)
    assert store.list_reconciliation_issues()[0]["kind"] == "HALT_UNKNOWN_ORDER_FILL"
    assert coordinator.is_ready() is False


def test_fill_cursor_uses_overlapping_whole_second_windows(tmp_path: Path) -> None:
    """A fill after Kraken's floored end second is recovered by the next overlap."""
    coordinator, store, exchange = _coordinator(tmp_path)
    exchange.fills = [_fill("boundary-fill", "external-order", "1", 1_500)]

    first = coordinator.reconcile_fills(since_ms=0, end_ms=1_999)
    second = coordinator.reconcile_fills(end_ms=2_999)

    assert first.end_ms == ONE_SECOND_MS
    assert first.fills == ()
    assert second.since_ms == 0
    assert [fill.fill_id for fill in second.fills] == ["boundary-fill"]
    assert store.get_sync_cursor("fills:account") == "2000"


def test_submission_persists_venue_normalized_values(tmp_path: Path) -> None:
    """Durable order identity exactly matches the values sent to the venue."""
    coordinator, store, exchange = _coordinator(tmp_path)

    result = coordinator.submit_entry(
        "entry-rounded",
        "BTC/USD",
        Decimal("1.24"),
        Decimal("100.006"),
        stop_loss_price=Decimal("90.006"),
    )

    assert result.order_id is not None
    stored = store.connection.execute(
        "SELECT quantity, limit_price, payload_json FROM order_intents WHERE intent_id = ?",
        (result.intent_id,),
    ).fetchone()
    assert stored["quantity"] == "1.2"
    assert stored["limit_price"] == "100.01"
    assert exchange.orders[result.exchange_order_id]["amount"] == Decimal("1.2")


def test_external_stop_fill_closes_inventory_but_halts_readiness(tmp_path: Path) -> None:
    """An unknown exact stop fill is accounted before its provenance is investigated."""
    coordinator, store, exchange = _coordinator(tmp_path)
    submission = coordinator.submit_entry("entry-stop", "BTC/USD", Decimal(1), Decimal(100))
    assert submission.exchange_order_id is not None
    exchange.fills = [_fill("entry-fill", submission.exchange_order_id, "1", 10)]
    coordinator.reconcile_fills(since_ms=0, end_ms=10)
    exchange.fills.append(
        _fill("external-stop-fill", "external-stop", "1", 20, side="sell"),
    )

    result = coordinator.reconcile_fills(since_ms=10, end_ms=20)

    assert result.complete is True
    assert store.get_position("BTC/USD") is None
    assert store.get_sync_cursor("fills:account") == "20"
    assert any(
        issue["kind"] == "HALT_UNKNOWN_ORDER_FILL" and issue["exact_key"] == "external-stop-fill"
        for issue in store.list_reconciliation_issues()
    )
    assert coordinator.ready is False


def test_malformed_fill_still_blocks_cursor(tmp_path: Path) -> None:
    """Exact IDs do not make a fill durable when required accounting fields are malformed."""
    coordinator, store, exchange = _coordinator(tmp_path)
    exchange.fills = [
        {
            "id": "malformed-fill",
            "order": "external-order",
            "timestamp": 10,
            "symbol": "BTC/USD",
            "side": "buy",
            "price": "100",
            "amount": None,
        },
    ]

    result = coordinator.reconcile_fills(since_ms=0, end_ms=20)

    assert result.complete is False
    assert result.unresolved_fill_ids == ("malformed-fill",)
    assert store.get_sync_cursor("fills:account") is None


def test_adopts_only_exact_refid_conditional_stop_child(tmp_path: Path) -> None:
    """A stop child is adopted only through its exact known ENTRY parent refid."""
    coordinator, store, exchange = _coordinator(tmp_path)
    submission = coordinator.submit_entry("entry-child", "BTC/USD", Decimal(1), Decimal(100))
    assert submission.exchange_order_id is not None
    exchange.fills = [_fill("parent-fill", submission.exchange_order_id, "1", 10)]
    coordinator.reconcile_fills(since_ms=0, end_ms=10)
    exchange.orders[submission.exchange_order_id]["status"] = "closed"
    exchange.orders["conditional-stop"] = {
        "id": "conditional-stop",
        "clientOrderId": None,
        "symbol": "BTC/USD",
        "side": "sell",
        "type": "stop-loss",
        "amount": Decimal(1),
        "stopLossPrice": Decimal(90),
        "status": "open",
        "timestamp": 11,
        "info": {"refid": submission.exchange_order_id},
    }

    result = coordinator.reconcile_open_orders("BTC/USD")

    assert len(result.adopted_protection_ids) == 1
    protection = store.get_protection(result.adopted_protection_ids[0])
    assert protection is not None
    assert protection["exchange_order_id"] == "conditional-stop"
    assert Decimal(protection["quantity"]) == Decimal(1)
    assert Decimal(protection["trigger_price"]) == Decimal(90)
    assert protection["epoch_id"] == store.get_position("BTC/USD").epoch_id  # type: ignore[union-attr]


def test_adopts_conditional_stop_that_filled_before_open_order_recovery(tmp_path: Path) -> None:
    """A closed stop child is proven by its exact parent before its fill is applied."""
    coordinator, store, exchange = _coordinator(tmp_path)
    entry = coordinator.submit_entry("entry-fast-stop", "BTC/USD", Decimal(1), Decimal(100))
    assert entry.exchange_order_id is not None
    exchange.fills = [_fill("fast-parent-fill", entry.exchange_order_id, "1", 10)]
    coordinator.reconcile_fills(since_ms=0, end_ms=10)
    exchange.orders[entry.exchange_order_id]["status"] = "closed"
    exchange.orders["fast-stop"] = {
        "id": "fast-stop",
        "clientOrderId": None,
        "symbol": "BTC/USD",
        "side": "sell",
        "type": "stop-loss",
        "amount": Decimal(1),
        "stopLossPrice": Decimal(90),
        "status": "closed",
        "timestamp": 11,
        "info": {"refid": entry.exchange_order_id},
    }
    exchange.fills.append(_fill("fast-stop-fill", "fast-stop", "1", 20, side="sell"))

    result = coordinator.reconcile_fills(since_ms=10, end_ms=20)

    assert result.complete is True
    assert store.get_position("BTC/USD") is None
    protection = store.get_protection_by_exchange_id("fast-stop")
    assert protection is not None
    assert protection["state"] == OrderState.FILLED.value
    assert not any(
        issue["kind"] == "HALT_UNKNOWN_ORDER_FILL" for issue in store.list_reconciliation_issues()
    )


def test_unknown_open_order_halts_readiness(tmp_path: Path) -> None:
    """An open order without an exact known identity or conditional refid is never adopted."""
    coordinator, store, exchange = _coordinator(tmp_path)
    exchange.orders["external-open"] = {
        "id": "external-open",
        "clientOrderId": "external-client",
        "symbol": "BTC/USD",
        "side": "buy",
        "type": "limit",
        "amount": Decimal(1),
        "status": "open",
        "timestamp": 1,
    }

    result = coordinator.reconcile_open_orders("BTC/USD")

    assert result.unknown_exchange_order_ids == ("external-open",)
    assert store.get_protection_by_exchange_id("external-open") is None
    assert store.list_reconciliation_issues()[0]["kind"] == "HALT_UNKNOWN_OPEN_ORDER"
    assert coordinator.ready is False


def test_emergency_exit_uses_exact_market_mutation(tmp_path: Path) -> None:
    """Emergency exits bypass the entry gate but retain a deterministic client ID."""
    coordinator, _store, exchange = _coordinator(tmp_path)

    result = coordinator.submit_exit(
        "exit-emergency",
        "BTC/USD",
        Decimal("0.5"),
        Decimal(99),
        emergency=True,
    )

    assert result.outcome is SubmissionOutcome.SUBMITTED
    assert exchange.mutation_calls == [("market-sell", result.client_order_id)]


def test_only_known_stale_entry_orders_are_canceled(tmp_path: Path) -> None:
    """Stale cleanup does not touch known exits or unknown account orders."""
    coordinator, store, exchange = _coordinator(tmp_path)
    entry = coordinator.submit_entry("entry-stale", "BTC/USD", Decimal(1), Decimal(100))
    exit_order = coordinator.submit_exit("exit-stale", "BTC/USD", Decimal(1), Decimal(101))
    exchange.orders["external-order"] = {
        "id": "external-order",
        "clientOrderId": "external-client",
        "symbol": "BTC/USD",
        "side": "buy",
        "amount": Decimal(1),
        "status": "open",
        "timestamp": 1,
    }

    canceled = coordinator.cancel_stale_entries(1, now_ms=5_000)

    assert [result.exchange_order_id for result in canceled] == [entry.exchange_order_id]
    assert exchange.cancelled == [entry.exchange_order_id]
    assert store.get_order(exit_order.order_id or 0).state is OrderState.OPEN
    assert exchange.orders["external-order"]["status"] == "open"


def test_balance_and_exact_protective_stop_drive_readiness(tmp_path: Path) -> None:
    """Exact balance and stop mismatches open issues that clear after correction."""
    coordinator, _store, exchange = _coordinator(tmp_path)
    exchange.balances["BTC"] = Decimal(1)
    assert coordinator.compare_balance("BTC/USD").matches is False
    assert coordinator.ready is False
    exchange.balances["BTC"] = Decimal(0)
    assert coordinator.compare_balance("BTC/USD").matches is True
    assert coordinator.ready is True


def test_protective_stop_ratchets_in_place_and_is_audited(tmp_path: Path) -> None:
    """A higher desired trigger amends the exact stop without a coverage gap."""
    coordinator, store, exchange = _coordinator(tmp_path)
    entry = coordinator.submit_entry("entry-ratchet", "BTC/USD", Decimal(1), Decimal(100))
    assert entry.exchange_order_id is not None
    exchange.fills = [_fill("ratchet-buy", entry.exchange_order_id, "1", 10)]
    coordinator.reconcile_fills(since_ms=0, end_ms=10)
    stop = coordinator.ensure_protective_stop(
        "BTC/USD",
        Decimal(1),
        Decimal(90),
        Decimal(100),
        Decimal(100),
    )

    assert coordinator.ratchet_protective_stops("BTC/USD", Decimal(95), Decimal(110))

    stored = store.get_protection(stop.protection_id)
    assert stored is not None
    assert Decimal(stored["trigger_price"]) == Decimal(95)
    assert exchange.orders[stop.exchange_order_id]["stopLossPrice"] == Decimal(95)
    amendments = store.connection.execute("SELECT * FROM protection_amendments").fetchall()
    assert len(amendments) == 1


def test_exact_protective_stop_mismatch_drives_readiness(tmp_path: Path) -> None:
    """A read-back mismatch blocks readiness until exact values match again."""
    coordinator, store, exchange = _coordinator(tmp_path)
    store.record_protection(
        "stop-1",
        "BTC/USD",
        "STOP_LOSS",
        Decimal(2),
        Decimal(90),
        exchange_order_id="stop-order-1",
    )
    exchange.orders["stop-order-1"] = {
        "id": "stop-order-1",
        "clientOrderId": "stop-client-1",
        "symbol": "BTC/USD",
        "side": "sell",
        "amount": Decimal(2),
        "stopLossPrice": Decimal(90),
        "status": "open",
        "timestamp": 1,
    }

    assert coordinator.verify_protective_stop("stop-1").valid is True
    exchange.orders["stop-order-1"]["amount"] = Decimal(1)
    mismatch = coordinator.verify_protective_stop("stop-1")
    assert mismatch.valid is False
    assert coordinator.ready is False
    exchange.orders["stop-order-1"]["amount"] = Decimal(2)
    assert coordinator.verify_protective_stop("stop-1").valid is True
    assert coordinator.ready is True


def test_protected_exit_rejection_restores_previous_exact_stop(tmp_path: Path) -> None:
    """A rejected market handoff re-arms the canceled stop with a new exact identity."""
    coordinator, store, exchange = _coordinator(tmp_path)
    entry = coordinator.submit_entry("entry-handoff", "BTC/USD", Decimal(1), Decimal(100))
    assert entry.exchange_order_id is not None
    exchange.fills = [_fill("handoff-buy", entry.exchange_order_id, "1", 10)]
    coordinator.reconcile_fills(since_ms=0, end_ms=10)
    protection = coordinator.ensure_protective_stop(
        "BTC/USD",
        Decimal(1),
        Decimal(90),
        Decimal(100),
        Decimal(100),
    )
    assert protection.outcome is SubmissionOutcome.SUBMITTED
    exchange.reject_next = True

    handoff = coordinator.submit_protected_exit(
        "exit-handoff",
        "BTC/USD",
        Decimal(1),
        Decimal(99),
    )

    assert handoff.submission is not None
    assert handoff.submission.outcome is SubmissionOutcome.REJECTED
    assert handoff.canceled_protection_ids == (protection.protection_id,)
    assert len(handoff.restored) == 1
    assert handoff.restored[0].outcome is SubmissionOutcome.SUBMITTED
    active = store.list_protection("BTC/USD", active_only=True)
    assert [item["protection_id"] for item in active] == [handoff.restored[0].protection_id]
    assert active[0]["exchange_order_id"] != protection.exchange_order_id


def test_canceled_exit_response_restores_previous_exact_stop(tmp_path: Path) -> None:
    """A venue-canceled placement is a failed handoff, never a successful exit."""
    coordinator, store, exchange = _coordinator(tmp_path)
    entry = coordinator.submit_entry("entry-canceled-exit", "BTC/USD", Decimal(1), Decimal(100))
    assert entry.exchange_order_id is not None
    exchange.fills = [_fill("canceled-exit-buy", entry.exchange_order_id, "1", 10)]
    coordinator.reconcile_fills(since_ms=0, end_ms=10)
    protection = coordinator.ensure_protective_stop(
        "BTC/USD",
        Decimal(1),
        Decimal(90),
        Decimal(100),
        Decimal(100),
    )
    exchange.cancel_next = True

    handoff = coordinator.submit_protected_exit(
        "exit-canceled",
        "BTC/USD",
        Decimal(1),
        Decimal(99),
    )

    assert handoff.submission is not None
    assert handoff.submission.outcome is SubmissionOutcome.REJECTED
    assert len(handoff.restored) == 1
    assert handoff.restored[0].outcome is SubmissionOutcome.SUBMITTED
    active_ids = [
        item["protection_id"] for item in store.list_protection("BTC/USD", active_only=True)
    ]
    assert active_ids == [handoff.restored[0].protection_id]


def test_created_order_is_terminalized_without_exchange_lookup(tmp_path: Path) -> None:
    """A crash before SUBMITTING leaves an intent that is known not to have been sent."""
    coordinator, store, exchange = _coordinator(tmp_path)
    store.create_order_intent("never-sent", "BTC/USD", "BUY", Decimal(1), limit_price=100)
    order = store.record_order("never-sent")

    recovered = coordinator.recover_unknown_orders()

    assert recovered == ()
    assert store.get_order(order.order_id).state is OrderState.FAILED
    assert exchange.mutation_calls == []


def test_coordinator_executes_delayed_paper_entry_protection_and_exit(tmp_path: Path) -> None:
    """The exact coordinator runs unchanged against bounded later-event PAPER mutations."""
    symbol = "BTC/USD"
    paper = PaperExchange(
        {"USD": Decimal(1000), "BTC": Decimal(0)},
        {
            symbol: MarketRules(
                "BTC",
                "USD",
                price_increment="1",
                amount_increment="1",
                min_notional="10",
                min_amount="1",
            ),
        },
        start_timestamp=0,
    )
    paper.process_market_event(symbol, bid="99", ask="101", last="100", timestamp=1_000)
    store = ExecutionStore(tmp_path / "paper-execution.db", "paper:kraken:test")
    coordinator = ExecutionCoordinator(
        store,
        paper,  # type: ignore[arg-type]
        live_entries_enabled=True,
    )

    entry = coordinator.submit_entry("paper-entry", symbol, Decimal(1), Decimal(99))
    assert entry.outcome is SubmissionOutcome.SUBMITTED
    assert paper.fetch_my_trades() == []
    paper.process_market_event(symbol, bid="98", ask="99", last="99", timestamp=2_000)
    coordinator.reconcile_fills(since_ms=0, end_ms=2_000)
    position = store.get_position(symbol)
    assert position is not None
    stop = coordinator.ensure_protective_stop(
        symbol,
        position.quantity,
        Decimal(90),
        position.average_cost,
        Decimal(99),
    )
    assert stop.verification is not None
    assert stop.verification.valid

    handoff = coordinator.submit_protected_exit(
        "paper-exit",
        symbol,
        position.quantity,
        Decimal(98),
    )
    assert handoff.submission is not None
    assert handoff.submission.outcome is SubmissionOutcome.SUBMITTED
    assert store.get_position(symbol) is not None
    paper.process_market_event(symbol, bid="97", ask="98", last="97", timestamp=3_000)
    coordinator.reconcile_fills(since_ms=2_000, end_ms=3_000)
    assert store.get_position(symbol) is None
    summary = coordinator.recover(symbol, Decimal(0))
    assert summary.readiness.ready is True


def test_live_scope_without_fill_cursor_refuses_epoch_zero_reconciliation(
    tmp_path: Path,
) -> None:
    """A fresh LIVE store must not import the whole account history from epoch 0."""
    coordinator, store, exchange = _coordinator(tmp_path)
    exchange.fills = [_fill("preexisting", "unknown-order", "1", 10)]

    result = coordinator.reconcile_fills()

    assert result.complete is False
    assert result.error is not None and "cursor" in result.error
    kinds = {issue["kind"] for issue in store.list_reconciliation_issues()}
    assert kinds == {"FILL_RECONCILIATION_FAILED"}
    # An explicit window still works (rebuild seeds the cursor that way).
    seeded = coordinator.reconcile_fills(since_ms=0, end_ms=5)
    assert seeded.complete is True
    followup = coordinator.reconcile_fills(end_ms=20)
    assert followup.error is None or "cursor" not in followup.error


def test_protection_issues_permit_remediation_but_not_full_readiness(
    tmp_path: Path,
) -> None:
    """Handoff and rearm failures allow stop re-arming instead of freezing it."""
    coordinator, store, _exchange = _coordinator(tmp_path)
    store.record_reconciliation_issue(
        "PROTECTED_EXIT_HANDOFF_FAILED",
        "cycle-1:exit:BTC/USD",
        {"message": "cancel timed out"},
    )
    assert coordinator.ready is False
    assert coordinator.protection_remediation_only is True

    store.record_reconciliation_issue("INVALID_FILL", "fill-x", {"message": "conflict"})
    assert coordinator.protection_remediation_only is False


def test_successful_protected_exit_resolves_prior_cycle_handoff_failure(
    tmp_path: Path,
) -> None:
    """A next-cycle retry closes the stale intent-keyed handoff issue for its symbol."""
    coordinator, store, exchange = _coordinator(tmp_path)
    entry = coordinator.submit_entry("entry-retry", "BTC/USD", Decimal(1), Decimal(100))
    assert entry.exchange_order_id is not None
    exchange.fills = [_fill("retry-buy", entry.exchange_order_id, "1", 10)]
    coordinator.reconcile_fills(since_ms=0, end_ms=10)
    store.record_reconciliation_issue(
        "PROTECTED_EXIT_HANDOFF_FAILED",
        "cycle-1:exit:BTC/USD",
        {"message": "prior cycle failed"},
    )

    handoff = coordinator.submit_protected_exit(
        "cycle-2:exit:BTC/USD",
        "BTC/USD",
        Decimal(1),
        Decimal(99),
    )

    assert handoff.submission is not None
    assert handoff.submission.outcome is SubmissionOutcome.SUBMITTED
    kinds = {issue["kind"] for issue in store.list_reconciliation_issues()}
    assert "PROTECTED_EXIT_HANDOFF_FAILED" not in kinds
    assert "EXIT_PENDING_FILL" in kinds
