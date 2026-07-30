"""Behavioral contract for the standalone v10 paper exchange."""

import json
from collections.abc import Callable
from decimal import Decimal
from typing import Any

import pytest

from paper_exchange import MarketRules, OrderNotFoundError, PaperExchange

D = Decimal
SYMBOL = "BTC/USDT"


def make_exchange(  # noqa: PLR0913
    *,
    usdt: str = "1000",
    btc: str = "2",
    maker_fee: str = "0.001",
    taker_fee: str = "0.002",
    slippage: str = "0.01",
    max_open_orders: int = 8,
    max_closed_orders: int = 16,
    max_fill_events: int = 16,
    fill_sink: Callable[[dict[str, Any]], None] | None = None,
) -> PaperExchange:
    """Build a deterministic exchange with simple exact precision rules."""
    return PaperExchange(
        initial_balances={"USDT": usdt, "BTC": btc},
        markets={
            SYMBOL: MarketRules(
                base="BTC",
                quote="USDT",
                price_increment="0.01",
                amount_increment="0.1",
                min_notional="10",
                min_amount="0.1",
            ),
        },
        maker_fee_rate=maker_fee,
        taker_fee_rate=taker_fee,
        stop_slippage=slippage,
        max_open_orders=max_open_orders,
        max_closed_orders=max_closed_orders,
        max_fill_events=max_fill_events,
        fill_sink=fill_sink,
    )


def publish(
    exchange: PaperExchange,
    bid: str = "99",
    ask: str = "101",
    last: str = "100",
) -> list[dict[str, Any]]:
    """Publish a market event using Decimal-compatible strings."""
    return exchange.process_market_event(SYMBOL, bid=bid, ask=ask, last=last)


def test_buy_reserves_quote_and_only_fills_on_later_event() -> None:
    """Signal-event placement must lock cash without receiving a same-event fill."""
    exchange = make_exchange()
    publish(exchange)

    order = exchange.place_maker_buy(SYMBOL, price="100", amount="1")

    assert order["status"] == "open"
    assert order["filled"] == D("0")
    assert exchange.get_total_balance("USDT") == D("1000")
    assert exchange.get_free_balance("USDT") == D("899.900")
    assert exchange.fetch_my_trades(SYMBOL) == []

    fills = exchange.process_market_event(
        SYMBOL,
        bid="98",
        ask="100",
        last="99",
        available_liquidity="0.4",
    )

    assert len(fills) == 1
    assert fills[0]["liquidity"] == "maker"
    assert fills[0]["fee"]["cost"] == D("0.0400")
    partial = exchange.get_order(order["id"])
    assert partial["status"] == "partially_filled"
    assert partial["filled"] == D("0.4")
    assert partial["remaining"] == D("0.6")
    assert exchange.get_total_balance("USDT") == D("959.9600")
    # Remaining reservation falls with total, so no unreserved cash is invented.
    assert exchange.get_free_balance("USDT") == D("899.9000")
    assert exchange.get_total_balance("BTC") == D("2.4")


def test_partial_buy_finishes_with_actual_maker_fee_and_closed_query() -> None:
    """Multiple fills aggregate exact cost/fee before the order closes."""
    exchange = make_exchange()
    publish(exchange)
    order = exchange.place_limit_order(SYMBOL, "buy", "1", "100")
    exchange.process_market_event(
        SYMBOL,
        bid="99",
        ask="100",
        last="100",
        available_liquidity="0.4",
    )

    fills = exchange.process_market_event(
        SYMBOL,
        bid="99",
        ask="100",
        last="100",
        available_liquidity="0.6",
    )

    assert fills[0]["amount"] == D("0.6")
    assert exchange.get_open_orders(SYMBOL) == []
    closed = exchange.get_closed_orders(SYMBOL)
    assert closed[0]["id"] == order["id"]
    assert closed[0]["status"] == "filled"
    assert closed[0]["cost"] == D("100.0")
    assert closed[0]["fee"]["cost"] == D("0.1000")
    assert exchange.get_total_balance("USDT") == D("899.9000")
    assert exchange.get_free_balance("USDT") == D("899.9000")


def test_post_only_cross_rejects_without_reserving_cash() -> None:
    """A limit that would take liquidity is retained as a rejected order."""
    exchange = make_exchange()
    publish(exchange)

    rejected = exchange.place_limit_order(SYMBOL, "buy", "1", "101")

    assert rejected["status"] == "rejected"
    assert "cross" in rejected["info"]["reject_reason"]
    assert exchange.get_open_orders() == []
    assert exchange.get_closed_orders()[0]["id"] == rejected["id"]
    assert exchange.get_balance("USDT") == {
        "free": D("1000"),
        "used": D("0"),
        "total": D("1000"),
    }


def test_cancel_releases_only_unfilled_reservation() -> None:
    """Canceling a partial buy keeps settled cash spent and unlocks the remainder."""
    exchange = make_exchange()
    publish(exchange)
    order = exchange.place_limit_order(SYMBOL, "buy", "1", "100")
    exchange.process_market_event(
        SYMBOL,
        bid="99",
        ask="100",
        last="100",
        available_liquidity="0.4",
    )

    canceled = exchange.cancel_order(order["id"], SYMBOL)

    assert canceled["status"] == "canceled"
    assert canceled["filled"] == D("0.4")
    assert exchange.get_free_balance("USDT") == D("959.9600")
    assert exchange.get_total_balance("USDT") == D("959.9600")
    with pytest.raises(OrderNotFoundError):
        exchange.cancel_order(order["id"])


def test_sell_reserves_base_and_credits_net_maker_proceeds() -> None:
    """Sell inventory stays in total while open and fees reduce quote proceeds."""
    exchange = make_exchange()
    publish(exchange)

    order = exchange.place_maker_sell(SYMBOL, price="102", amount="1")
    assert exchange.get_total_balance("BTC") == D("2")
    assert exchange.get_free_balance("BTC") == D("1")

    exchange.process_market_event(SYMBOL, bid="102", ask="103", last="102")

    assert exchange.get_order(order["id"])["status"] == "filled"
    assert exchange.get_total_balance("BTC") == D("1")
    assert exchange.get_total_balance("USDT") == D("1101.898")
    fill = exchange.fetch_my_trades(SYMBOL)[0]
    assert fill["fee"]["cost"] == D("0.102")
    assert fill["liquidity"] == "maker"


@pytest.mark.parametrize(
    ("amount", "price", "reason"),
    [
        ("0.15", "100", "amount must be a multiple"),
        ("0.1", "100.001", "price must be a multiple"),
        ("0.1", "50", "notional is below minimum"),
    ],
)
def test_precision_and_notional_failures_are_rejected(
    amount: str,
    price: str,
    reason: str,
) -> None:
    """Exchange precision and minimums are strict rather than silently rounded."""
    exchange = make_exchange()
    publish(exchange)

    order = exchange.place_limit_order(SYMBOL, "buy", amount, price)

    assert order["status"] == "rejected"
    assert reason in order["info"]["reject_reason"]
    assert exchange.get_free_balance("USDT") == D("1000")


def test_insufficient_fee_reserve_rejects_buy() -> None:
    """A buy needs quote for both limit notional and its actual maker fee."""
    exchange = make_exchange(usdt="100")
    publish(exchange)

    order = exchange.place_limit_order(SYMBOL, "buy", "1", "100")

    assert order["status"] == "rejected"
    assert order["info"]["reject_reason"] == "insufficient free USDT"
    assert exchange.get_total_balance("USDT") == D("100")


def test_stop_market_triggers_later_with_slippage_and_taker_fee() -> None:
    """A stop reserves inventory and settles at slipped bid using the taker rate."""
    exchange = make_exchange()
    publish(exchange)

    order = exchange.place_stop_market_order(SYMBOL, "sell", "1", "90")
    assert order["status"] == "open"
    assert exchange.get_free_balance("BTC") == D("1")
    assert exchange.fetch_my_trades() == []

    fills = exchange.process_market_event(SYMBOL, bid="89", ask="90", last="89")

    # 89 bid less 1% slippage = 88.11, then 0.2% taker fee.
    assert fills[0]["price"] == D("88.11")
    assert fills[0]["fee"]["cost"] == D("0.17622")
    assert fills[0]["liquidity"] == "taker"
    closed = exchange.get_order(order["id"])
    assert closed["status"] == "filled"
    assert closed["info"]["triggered"] is True
    assert exchange.get_total_balance("BTC") == D("1")
    assert exchange.get_total_balance("USDT") == D("1087.93378")


def test_triggered_stop_can_fill_partially_across_events() -> None:
    """Once triggered, a stop remains a taker order until all reserved base fills."""
    exchange = make_exchange()
    publish(exchange)
    order = exchange.place_stop_market_order(SYMBOL, "sell", "1", "90")

    exchange.process_market_event(
        SYMBOL,
        bid="89",
        ask="90",
        last="89",
        available_liquidity="0.4",
    )
    partial = exchange.get_order(order["id"])
    assert partial["status"] == "partially_filled"
    assert partial["info"]["triggered"] is True
    assert exchange.get_free_balance("BTC") == D("1")

    exchange.process_market_event(
        SYMBOL,
        bid="92",
        ask="93",
        last="92",
        available_liquidity="0.6",
    )
    assert exchange.get_order(order["id"])["status"] == "filled"
    assert [fill["price"] for fill in exchange.fetch_my_trades()] == [D("88.11"), D("91.08")]


def test_buy_stop_is_explicitly_rejected_as_not_cash_bounded() -> None:
    """Unbounded-gap buy stops are outside this cash-safe spot model."""
    exchange = make_exchange()
    publish(exchange)

    order = exchange.place_stop_market_order(SYMBOL, "buy", "1", "110")

    assert order["status"] == "rejected"
    assert "buy stop-market" in order["info"]["reject_reason"]


def test_client_id_conditional_stop_child_and_exact_fetch_compatibility() -> None:
    """A protected entry spawns a later-event stop child linked by exact refid."""
    exchange = make_exchange()
    exchange.process_market_event(SYMBOL, bid="99", ask="101", last="100", timestamp=1_000)
    parent = exchange.place_maker_buy(
        SYMBOL,
        price="100",
        amount="1",
        client_order_id="paper-client-1",
        stop_loss_price="90",
    )
    exchange.process_market_event(SYMBOL, bid="98", ask="100", last="100", timestamp=2_000)

    fetched_parent = exchange.fetch_order(
        parent["id"],
        SYMBOL,
        client_order_id="paper-client-1",
    )
    assert fetched_parent["status"] == "filled"
    children = exchange.fetch_open_orders(SYMBOL)
    assert len(children) == 1
    assert children[0]["info"]["refid"] == parent["id"]
    assert children[0]["stopLossPrice"] == D("90")
    assert children[0]["clientOrderId"] is None


def test_order_and_fill_collections_are_explicitly_bounded() -> None:
    """Open, closed, and fill retention obey constructor caps deterministically."""
    exchange = make_exchange(max_open_orders=1, max_closed_orders=1, max_fill_events=1)
    publish(exchange)
    first = exchange.place_limit_order(SYMBOL, "buy", "0.1", "100")
    rejected = exchange.place_limit_order(SYMBOL, "buy", "0.1", "99")
    assert rejected["status"] == "rejected"
    assert "max_open_orders" in rejected["info"]["reject_reason"]
    assert len(exchange.get_open_orders()) == 1
    assert len(exchange.get_closed_orders()) == 1

    exchange.process_market_event(SYMBOL, bid="99", ask="100", last="100")
    assert exchange.get_order(first["id"])["status"] == "filled"
    assert len(exchange.get_closed_orders()) == 1
    with pytest.raises(OrderNotFoundError):
        exchange.get_order(rejected["id"])

    second = exchange.place_limit_order(SYMBOL, "sell", "0.1", "102")
    exchange.process_market_event(SYMBOL, bid="102", ask="103", last="102")
    assert exchange.get_order(second["id"])["status"] == "filled"
    fills = exchange.fetch_my_trades()
    assert len(fills) == 1
    assert fills[0]["order"] == second["id"]


def test_state_is_json_serializable_and_resumes_ids_without_fill_duplication() -> None:
    """Checkpoint restore preserves accounting and sequences, not ledger history."""
    persisted_fills = []
    exchange = make_exchange(fill_sink=persisted_fills.append)
    publish(exchange)
    first = exchange.place_limit_order(SYMBOL, "buy", "0.1", "100")
    exchange.process_market_event(SYMBOL, bid="99", ask="100", last="100")
    assert len(persisted_fills) == 1

    state = exchange.to_state()
    encoded = json.dumps(state)
    restored_sink = []
    restored = PaperExchange.from_state(json.loads(encoded), fill_sink=restored_sink.append)

    assert restored.fetch_my_trades() == []
    assert restored.get_total_balance("USDT") == exchange.get_total_balance("USDT")
    assert restored.get_order(first["id"])["status"] == "filled"
    second = restored.place_limit_order(SYMBOL, "buy", "0.2", "99")
    assert second["id"] == "paper-order-0000000002"
    restored.process_market_event(SYMBOL, bid="98", ask="99", last="99")
    assert restored_sink[0]["id"] == "paper-fill-0000000002"


def test_ccxt_shaped_adapter_and_balance_snapshot() -> None:
    """Normalized internals retain practical Kraken/CCXT-shaped entry points."""
    exchange = make_exchange()
    publish(exchange)

    order = exchange.create_order(
        symbol=SYMBOL,
        type="limit",
        side="buy",
        amount="0.1",
        price="100",
        params={"postOnly": True},
    )

    assert order["postOnly"] is True
    assert exchange.fetch_open_orders(SYMBOL)[0]["id"] == order["id"]
    balances = exchange.fetch_balance()
    assert balances["total"]["USDT"] == D("1000")
    assert balances["used"]["USDT"] == D("10.0100")
    assert balances["free"]["USDT"] == D("989.9900")
