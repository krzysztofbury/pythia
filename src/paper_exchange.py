"""Deterministic, cash-correct spot exchange simulator.

``PaperExchange`` is deliberately independent from the existing paper portfolio. A
market event must be processed before an order can be accepted, and an accepted
order can only fill on a later event. This prevents a signal from seeing a bar and
then receiving an impossible fill on that same bar.

The optional ``fill_sink`` is the integration boundary for a durable execution
ledger. It is called once per fill with a deterministic fill ID; implementations
should persist idempotently by that ID. ``to_state()`` checkpoints balances and
aggregate order state, but intentionally omits the rolling fill queue so restoring
a checkpoint does not duplicate the execution ledger.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal, InvalidOperation
from typing import Any, Literal, TypeAlias

DecimalInput: TypeAlias = Decimal | str | int | float
Side: TypeAlias = Literal["buy", "sell"]
FillSink: TypeAlias = Callable[[dict[str, Any]], None]

_ZERO = Decimal(0)
_ONE = Decimal(1)
_OPEN_STATUSES = frozenset({"open", "partially_filled"})
_CLOSED_STATUSES = frozenset({"filled", "canceled", "rejected"})


class PaperExchangeError(Exception):
    """Base exception for paper exchange errors."""


class OrderNotFoundError(PaperExchangeError):
    """Raised when an order ID is not retained by the bounded exchange state."""


def _decimal(value: DecimalInput, name: str) -> Decimal:
    """Convert external numeric input without inheriting binary float error."""
    if isinstance(value, bool):
        msg = f"{name} must be numeric"
        raise TypeError(msg)
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        msg = f"{name} must be a finite decimal"
        raise ValueError(msg) from exc
    if not result.is_finite():
        msg = f"{name} must be a finite decimal"
        raise ValueError(msg)
    return result


def _positive_limit(value: int, name: str) -> int:
    """Validate a collection bound."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        msg = f"{name} must be a positive integer"
        raise ValueError(msg)
    return value


@dataclass(frozen=True, slots=True, init=False)
class MarketRules:
    """Precision and minimums for one base/quote spot market."""

    base: str
    quote: str
    price_increment: Decimal
    amount_increment: Decimal
    min_notional: Decimal
    min_amount: Decimal

    def __init__(  # noqa: PLR0913
        self,
        base: str,
        quote: str,
        price_increment: DecimalInput,
        amount_increment: DecimalInput,
        min_notional: DecimalInput,
        min_amount: DecimalInput = _ZERO,
    ) -> None:
        """Normalize broad numeric inputs into immutable Decimal rules."""
        normalized_base = base.strip().upper()
        normalized_quote = quote.strip().upper()
        price_value = _decimal(price_increment, "price_increment")
        amount_value = _decimal(amount_increment, "amount_increment")
        notional_value = _decimal(min_notional, "min_notional")
        minimum_amount = _decimal(min_amount, "min_amount")

        base = normalized_base
        quote = normalized_quote
        if not base or not quote or base == quote:
            msg = "base and quote must be distinct non-empty currencies"
            raise ValueError(msg)
        if price_value <= 0:
            msg = "price_increment must be positive"
            raise ValueError(msg)
        if amount_value <= 0:
            msg = "amount_increment must be positive"
            raise ValueError(msg)
        if notional_value < 0:
            msg = "min_notional must be non-negative"
            raise ValueError(msg)
        if minimum_amount < 0:
            msg = "min_amount must be non-negative"
            raise ValueError(msg)

        object.__setattr__(self, "base", base)
        object.__setattr__(self, "quote", quote)
        object.__setattr__(self, "price_increment", price_value)
        object.__setattr__(self, "amount_increment", amount_value)
        object.__setattr__(self, "min_notional", notional_value)
        object.__setattr__(self, "min_amount", minimum_amount)

    def to_state(self) -> dict[str, str]:
        """Return a JSON-safe rule representation."""
        return {
            "base": self.base,
            "quote": self.quote,
            "price_increment": str(self.price_increment),
            "amount_increment": str(self.amount_increment),
            "min_notional": str(self.min_notional),
            "min_amount": str(self.min_amount),
        }


@dataclass(slots=True)
class _MarketSnapshot:
    """Latest quote used for post-only checks and later-event matching."""

    bid: Decimal
    ask: Decimal
    last: Decimal
    event_sequence: int
    timestamp: int


@dataclass(slots=True)
class _Order:
    """Mutable internal order state; public callers receive detached dictionaries."""

    id: str
    timestamp: int
    updated_timestamp: int
    symbol: str
    type: Literal["limit", "market", "stop_market"]
    side: Side
    amount: Decimal
    price: Decimal | None
    stop_price: Decimal | None
    post_only: bool
    status: str
    filled: Decimal
    remaining: Decimal
    cost: Decimal
    fee_paid: Decimal
    fee_currency: str
    created_event_sequence: int
    triggered: bool = False
    reserved_currency: str | None = None
    reserved_remaining: Decimal = _ZERO
    reject_reason: str | None = None
    last_trade_timestamp: int | None = None
    last_liquidity: Literal["maker", "taker"] | None = None
    client_order_id: str | None = None
    conditional_stop_price: Decimal | None = None
    parent_order_id: str | None = None


class PaperExchange:
    """Bounded, deterministic event-driven simulation of a cash spot exchange."""

    STATE_VERSION = 1
    fill_window_granularity_ms = 1

    def __init__(  # noqa: PLR0913
        self,
        initial_balances: Mapping[str, DecimalInput],
        markets: Mapping[str, MarketRules | Mapping[str, Any]],
        *,
        maker_fee_rate: DecimalInput = "0.0016",
        taker_fee_rate: DecimalInput = "0.0026",
        stop_slippage: DecimalInput = "0.001",
        max_open_orders: int = 256,
        max_closed_orders: int = 1024,
        max_fill_events: int = 2048,
        max_markets: int = 64,
        start_timestamp: int = 0,
        fill_sink: FillSink | None = None,
    ) -> None:
        """Initialize balances, fee schedule, bounds, and fixed market rules."""
        self._max_open_orders = _positive_limit(max_open_orders, "max_open_orders")
        self._max_closed_orders = _positive_limit(max_closed_orders, "max_closed_orders")
        self._max_fill_events = _positive_limit(max_fill_events, "max_fill_events")
        self._max_markets = _positive_limit(max_markets, "max_markets")
        if len(markets) > self._max_markets:
            msg = f"markets exceeds max_markets={self._max_markets}"
            raise ValueError(msg)
        if isinstance(start_timestamp, bool) or not isinstance(start_timestamp, int):
            msg = "start_timestamp must be an integer"
            raise TypeError(msg)
        if start_timestamp < 0:
            msg = "start_timestamp must be non-negative"
            raise ValueError(msg)

        self._maker_fee_rate = _decimal(maker_fee_rate, "maker_fee_rate")
        self._taker_fee_rate = _decimal(taker_fee_rate, "taker_fee_rate")
        self._stop_slippage = _decimal(stop_slippage, "stop_slippage")
        self._validate_rates()

        self._markets: dict[str, MarketRules] = {}
        for raw_symbol, raw_rules in markets.items():
            symbol = raw_symbol.strip().upper()
            rules = self._coerce_rules(symbol, raw_rules)
            if symbol != f"{rules.base}/{rules.quote}":
                msg = f"market key {symbol!r} does not match {rules.base}/{rules.quote}"
                raise ValueError(msg)
            self._markets[symbol] = rules

        self._balances: dict[str, Decimal] = {}
        for raw_currency, raw_amount in initial_balances.items():
            currency = raw_currency.strip().upper()
            amount = _decimal(raw_amount, f"balance[{currency}]")
            if not currency:
                msg = "balance currency must be non-empty"
                raise ValueError(msg)
            if amount < 0:
                msg = f"balance[{currency}] must be non-negative"
                raise ValueError(msg)
            self._balances[currency] = amount
        for rules in self._markets.values():
            self._balances.setdefault(rules.base, _ZERO)
            self._balances.setdefault(rules.quote, _ZERO)

        self._reserved = dict.fromkeys(self._balances, _ZERO)
        self._latest_markets: dict[str, _MarketSnapshot] = {}
        self._open_orders: dict[str, _Order] = {}
        self._closed_orders: dict[str, _Order] = {}
        self._fills: list[dict[str, Any]] = []
        self._clock = start_timestamp
        self._event_sequence = 0
        self._order_sequence = 0
        self._fill_sequence = 0
        self._fill_sink = fill_sink
        self._assert_accounting()

    def _validate_rates(self) -> None:
        """Validate fee and slippage fractions explicitly."""
        if not _ZERO <= self._maker_fee_rate < _ONE:
            msg = "maker_fee_rate must be in [0, 1)"
            raise ValueError(msg)
        if not _ZERO <= self._taker_fee_rate < _ONE:
            msg = "taker_fee_rate must be in [0, 1)"
            raise ValueError(msg)
        if not _ZERO <= self._stop_slippage < _ONE:
            msg = "stop_slippage must be in [0, 1)"
            raise ValueError(msg)

    @staticmethod
    def _coerce_rules(
        symbol: str,
        raw_rules: MarketRules | Mapping[str, Any],
    ) -> MarketRules:
        """Normalize mapping-based rules while keeping a typed public option."""
        if isinstance(raw_rules, MarketRules):
            return raw_rules
        base, separator, quote = symbol.partition("/")
        if not separator:
            msg = f"invalid symbol {symbol!r}; expected BASE/QUOTE"
            raise ValueError(msg)
        return MarketRules(
            base=str(raw_rules.get("base", base)),
            quote=str(raw_rules.get("quote", quote)),
            price_increment=raw_rules["price_increment"],
            amount_increment=raw_rules["amount_increment"],
            min_notional=raw_rules["min_notional"],
            min_amount=raw_rules.get("min_amount", _ZERO),
        )

    def _next_timestamp(self, proposed: int | None = None) -> int:
        """Advance the deterministic logical clock, optionally to external time."""
        if proposed is None:
            self._clock += 1
            return self._clock
        if isinstance(proposed, bool) or not isinstance(proposed, int):
            msg = "timestamp must be an integer"
            raise TypeError(msg)
        if proposed <= self._clock:
            msg = f"timestamp must be greater than logical clock {self._clock}"
            raise ValueError(msg)
        self._clock = proposed
        return self._clock

    def _next_order_id(self) -> str:
        """Return a stable monotonic paper order ID."""
        self._order_sequence += 1
        return f"paper-order-{self._order_sequence:010d}"

    def _next_fill_id(self) -> str:
        """Return a stable monotonic paper fill ID."""
        self._fill_sequence += 1
        return f"paper-fill-{self._fill_sequence:010d}"

    def symbol_exists(self, symbol: str) -> bool:
        """Return whether fixed precision rules exist for a symbol."""
        return symbol.strip().upper() in self._markets

    def fetch_price(self, symbol: str) -> Decimal:
        """Return the latest event's last price."""
        normalized = self._require_symbol(symbol)
        snapshot = self._latest_markets.get(normalized)
        if snapshot is None:
            msg = f"no market event available for {normalized}"
            raise PaperExchangeError(msg)
        return snapshot.last

    def get_market_structure(self, symbol: str) -> dict[str, Decimal]:
        """Return the latest bid, ask, and fractional spread."""
        normalized = self._require_symbol(symbol)
        snapshot = self._latest_markets.get(normalized)
        if snapshot is None:
            msg = f"no market event available for {normalized}"
            raise PaperExchangeError(msg)
        return {
            "bid": snapshot.bid,
            "ask": snapshot.ask,
            "spread_pct": (snapshot.ask - snapshot.bid) / snapshot.bid,
        }

    def process_market_event(  # noqa: C901, PLR0913
        self,
        symbol: str,
        *,
        bid: DecimalInput,
        ask: DecimalInput,
        last: DecimalInput | None = None,
        available_liquidity: DecimalInput | None = None,
        timestamp: int | None = None,
    ) -> list[dict[str, Any]]:
        """Publish one quote event and match only orders from earlier events.

        ``available_liquidity`` is shared FIFO across executable orders and is in
        base units. ``None`` means enough liquidity to fill every matching order;
        an explicit value enables deterministic partial fills.
        """
        normalized = self._require_symbol(symbol)
        bid_value = _decimal(bid, "bid")
        ask_value = _decimal(ask, "ask")
        last_value = _decimal(last, "last") if last is not None else (bid_value + ask_value) / 2
        if bid_value <= 0 or ask_value <= 0 or last_value <= 0:
            msg = "bid, ask, and last must be positive"
            raise ValueError(msg)
        if bid_value > ask_value:
            msg = "bid must not exceed ask"
            raise ValueError(msg)

        liquidity = None
        if available_liquidity is not None:
            liquidity = _decimal(available_liquidity, "available_liquidity")
            if liquidity < 0:
                msg = "available_liquidity must be non-negative"
                raise ValueError(msg)

        event_timestamp = self._next_timestamp(timestamp)
        self._event_sequence += 1
        snapshot = _MarketSnapshot(
            bid=bid_value,
            ask=ask_value,
            last=last_value,
            event_sequence=self._event_sequence,
            timestamp=event_timestamp,
        )
        self._latest_markets[normalized] = snapshot

        event_fills: list[dict[str, Any]] = []
        for order_id in list(self._open_orders):
            order = self._open_orders.get(order_id)
            if order is None or order.symbol != normalized:
                continue
            if order.created_event_sequence >= snapshot.event_sequence:
                continue
            if liquidity is not None and liquidity <= 0:
                break

            fill_price = self._matching_price(order, snapshot)
            if fill_price is None:
                continue
            fill_amount = self._fill_amount(order, liquidity)
            if fill_amount <= 0:
                continue

            fill = self._execute_fill(order, fill_amount, fill_price, event_timestamp)
            event_fills.append(fill)
            if liquidity is not None:
                liquidity -= fill_amount

        return deepcopy(event_fills)

    def _matching_price(
        self,
        order: _Order,
        snapshot: _MarketSnapshot,
    ) -> Decimal | None:
        """Return an executable price, activating stops before matching."""
        if order.type == "limit":
            assert order.price is not None  # noqa: S101
            if order.side == "buy" and snapshot.ask <= order.price:
                return order.price
            if order.side == "sell" and snapshot.bid >= order.price:
                return order.price
            return None

        if order.type == "market":
            assert order.side == "sell"  # noqa: S101 - only bounded market sells are accepted
            return snapshot.bid

        assert order.type == "stop_market"  # noqa: S101
        assert order.stop_price is not None  # noqa: S101
        assert order.side == "sell"  # noqa: S101 - buy stops are rejected at placement
        if not order.triggered:
            if snapshot.last > order.stop_price:
                return None
            order.triggered = True
            order.updated_timestamp = snapshot.timestamp
        rules = self._markets[order.symbol]
        slipped_bid = snapshot.bid * (_ONE - self._stop_slippage)
        execution_price = self._quantize(slipped_bid, rules.price_increment, ROUND_DOWN)
        return execution_price if execution_price > 0 else None

    def _fill_amount(
        self,
        order: _Order,
        liquidity: Decimal | None,
    ) -> Decimal:
        """Bound a fill by remaining event liquidity and amount precision."""
        if liquidity is None:
            return order.remaining
        rules = self._markets[order.symbol]
        raw_amount = min(order.remaining, liquidity)
        return self._quantize(raw_amount, rules.amount_increment, ROUND_DOWN)

    @staticmethod
    def _quantize(value: Decimal, increment: Decimal, rounding: str) -> Decimal:
        """Round to an arbitrary exchange increment, not only powers of ten."""
        units = (value / increment).to_integral_value(rounding=rounding)
        return units * increment

    def place_limit_order(  # noqa: PLR0913
        self,
        symbol: str,
        side: Side,
        amount: DecimalInput,
        price: DecimalInput,
        *,
        post_only: bool = True,
        client_order_id: str | None = None,
        conditional_stop_price: DecimalInput | None = None,
    ) -> dict[str, Any]:
        """Place a strictly post-only spot limit order or return a rejection."""
        normalized = self._require_symbol(symbol)
        normalized_side = self._require_side(side)
        amount_value = _decimal(amount, "amount")
        price_value = _decimal(price, "price")
        stop_value = (
            None
            if conditional_stop_price is None
            else _decimal(conditional_stop_price, "conditional_stop_price")
        )
        timestamp = self._next_timestamp()
        order = self._new_order(
            symbol=normalized,
            order_type="limit",
            side=normalized_side,
            amount=amount_value,
            price=price_value,
            stop_price=None,
            post_only=post_only,
            timestamp=timestamp,
            client_order_id=client_order_id,
            conditional_stop_price=stop_value,
        )

        reason = self._limit_rejection_reason(order)
        if reason is not None:
            return self._reject(order, reason)

        rules = self._markets[normalized]
        reservation = (
            amount_value * price_value * (_ONE + self._maker_fee_rate)
            if normalized_side == "buy"
            else amount_value
        )
        currency = rules.quote if normalized_side == "buy" else rules.base
        if self.get_free_balance(currency) < reservation:
            return self._reject(order, f"insufficient free {currency}")

        self._reserve(order, currency, reservation)
        self._open_orders[order.id] = order
        self._assert_accounting()
        return self._order_dict(order)

    def _limit_rejection_reason(self, order: _Order) -> str | None:  # noqa: C901, PLR0911
        """Return the first deterministic limit-order rejection reason."""
        client_reason = self._client_order_id_rejection(order.client_order_id)
        if client_reason is not None:
            return client_reason
        if len(self._open_orders) >= self._max_open_orders:
            return f"max_open_orders={self._max_open_orders} reached"
        if not order.post_only:
            return "only post-only limit orders are supported"
        rules = self._markets[order.symbol]
        reason = self._precision_or_minimum_reason(order.amount, order.price, rules)
        if reason is not None:
            return reason
        snapshot = self._latest_markets.get(order.symbol)
        if snapshot is None:
            return "market event required before order placement"
        assert order.price is not None  # noqa: S101
        if order.side == "buy" and order.price >= snapshot.ask:
            return "post-only buy would cross the ask"
        if order.side == "sell" and order.price <= snapshot.bid:
            return "post-only sell would cross the bid"
        if order.conditional_stop_price is not None:
            if order.side != "buy":
                return "conditional stops are supported only for buy entries"
            stop_reason = self._precision_or_minimum_reason(
                order.amount,
                order.conditional_stop_price,
                rules,
            )
            if stop_reason is not None:
                return f"conditional stop {stop_reason}"
            if order.conditional_stop_price >= snapshot.last:
                return "conditional sell stop must be below current price"
        return None

    def place_stop_market_order(  # noqa: PLR0913
        self,
        symbol: str,
        side: Side,
        amount: DecimalInput,
        stop_price: DecimalInput,
        *,
        client_order_id: str | None = None,
        parent_order_id: str | None = None,
    ) -> dict[str, Any]:
        """Place a reserved stop-market sell, filled as taker after a later trigger."""
        normalized = self._require_symbol(symbol)
        normalized_side = self._require_side(side)
        amount_value = _decimal(amount, "amount")
        stop_value = _decimal(stop_price, "stop_price")
        timestamp = self._next_timestamp()
        order = self._new_order(
            symbol=normalized,
            order_type="stop_market",
            side=normalized_side,
            amount=amount_value,
            price=None,
            stop_price=stop_value,
            post_only=False,
            timestamp=timestamp,
            client_order_id=client_order_id,
            parent_order_id=parent_order_id,
        )

        reason = self._stop_rejection_reason(order)
        if reason is not None:
            return self._reject(order, reason)

        rules = self._markets[normalized]
        if self.get_free_balance(rules.base) < amount_value:
            return self._reject(order, f"insufficient free {rules.base}")

        self._reserve(order, rules.base, amount_value)
        self._open_orders[order.id] = order
        self._assert_accounting()
        return self._order_dict(order)

    def _stop_rejection_reason(self, order: _Order) -> str | None:  # noqa: PLR0911
        """Return the first deterministic stop-order rejection reason."""
        client_reason = self._client_order_id_rejection(order.client_order_id)
        if client_reason is not None:
            return client_reason
        if len(self._open_orders) >= self._max_open_orders:
            return f"max_open_orders={self._max_open_orders} reached"
        if order.side != "sell":
            return "cash-safe buy stop-market orders are not supported"
        rules = self._markets[order.symbol]
        reason = self._precision_or_minimum_reason(order.amount, order.stop_price, rules)
        if reason is not None:
            return reason
        snapshot = self._latest_markets.get(order.symbol)
        if snapshot is None:
            return "market event required before order placement"
        assert order.stop_price is not None  # noqa: S101
        if order.stop_price >= snapshot.last:
            return "sell stop would trigger immediately"
        return None

    def place_market_sell(
        self,
        symbol: str,
        amount: DecimalInput,
        *,
        client_order_id: str | None = None,
        reference_price: DecimalInput | None = None,
    ) -> dict[str, Any]:
        """Queue a bounded market sell that can fill only on a later event."""
        normalized = self._require_symbol(symbol)
        amount_value = _decimal(amount, "amount")
        snapshot = self._latest_markets.get(normalized)
        validation_price = (
            _decimal(reference_price, "reference_price")
            if reference_price is not None
            else (snapshot.bid if snapshot is not None else None)
        )
        timestamp = self._next_timestamp()
        order = self._new_order(
            symbol=normalized,
            order_type="market",
            side="sell",
            amount=amount_value,
            price=None,
            stop_price=None,
            post_only=False,
            timestamp=timestamp,
            client_order_id=client_order_id,
        )
        reason = self._market_sell_rejection_reason(order, validation_price)
        if reason is not None:
            return self._reject(order, reason)
        rules = self._markets[normalized]
        if self.get_free_balance(rules.base) < amount_value:
            return self._reject(order, f"insufficient free {rules.base}")
        self._reserve(order, rules.base, amount_value)
        self._open_orders[order.id] = order
        self._assert_accounting()
        return self._order_dict(order)

    def _market_sell_rejection_reason(
        self,
        order: _Order,
        validation_price: Decimal | None,
    ) -> str | None:
        client_reason = self._client_order_id_rejection(order.client_order_id)
        if client_reason is not None:
            return client_reason
        if len(self._open_orders) >= self._max_open_orders:
            return f"max_open_orders={self._max_open_orders} reached"
        if validation_price is None:
            return "market event required before market sell"
        return self._precision_or_minimum_reason(
            order.amount,
            validation_price,
            self._markets[order.symbol],
        )

    def _client_order_id_rejection(self, client_order_id: str | None) -> str | None:
        if client_order_id is None:
            return None
        if not isinstance(client_order_id, str):
            return "clientOrderId must be a string"
        if not client_order_id or client_order_id.strip() != client_order_id:
            return "clientOrderId must be exact and non-empty"
        retained = (*self._open_orders.values(), *self._closed_orders.values())
        if any(order.client_order_id == client_order_id for order in retained):
            return f"duplicate clientOrderId {client_order_id}"
        return None

    @staticmethod
    def _precision_or_minimum_reason(  # noqa: PLR0911
        amount: Decimal,
        price: Decimal | None,
        rules: MarketRules,
    ) -> str | None:
        """Validate positivity, exact increments, amount, and notional minimums."""
        if amount <= 0:
            return "amount must be positive"
        if price is None or price <= 0:
            return "price must be positive"
        if amount % rules.amount_increment != 0:
            return f"amount must be a multiple of {rules.amount_increment}"
        if price % rules.price_increment != 0:
            return f"price must be a multiple of {rules.price_increment}"
        if amount < rules.min_amount:
            return f"amount is below minimum {rules.min_amount}"
        if amount * price < rules.min_notional:
            return f"notional is below minimum {rules.min_notional}"
        return None

    def _new_order(  # noqa: PLR0913
        self,
        *,
        symbol: str,
        order_type: Literal["limit", "market", "stop_market"],
        side: Side,
        amount: Decimal,
        price: Decimal | None,
        stop_price: Decimal | None,
        post_only: bool,
        timestamp: int,
        client_order_id: str | None = None,
        conditional_stop_price: Decimal | None = None,
        parent_order_id: str | None = None,
    ) -> _Order:
        """Create an unreserved order attempt with a deterministic ID."""
        return _Order(
            id=self._next_order_id(),
            timestamp=timestamp,
            updated_timestamp=timestamp,
            symbol=symbol,
            type=order_type,
            side=side,
            amount=amount,
            price=price,
            stop_price=stop_price,
            post_only=post_only,
            status="open",
            filled=_ZERO,
            remaining=max(amount, _ZERO),
            cost=_ZERO,
            fee_paid=_ZERO,
            fee_currency=self._markets[symbol].quote,
            created_event_sequence=self._event_sequence,
            client_order_id=client_order_id,
            conditional_stop_price=conditional_stop_price,
            parent_order_id=parent_order_id,
        )

    def _reject(self, order: _Order, reason: str) -> dict[str, Any]:
        """Move an order attempt to bounded rejected history."""
        order.status = "rejected"
        order.reject_reason = reason
        order.remaining = max(order.amount, _ZERO)
        self._archive(order)
        self._assert_accounting()
        return self._order_dict(order)

    def _reserve(self, order: _Order, currency: str, amount: Decimal) -> None:
        """Lock free cash or inventory for an accepted order."""
        assert amount > 0  # noqa: S101
        assert self.get_free_balance(currency) >= amount  # noqa: S101
        self._reserved[currency] += amount
        order.reserved_currency = currency
        order.reserved_remaining = amount

    def _consume_reservation(self, order: _Order, amount: Decimal) -> None:
        """Consume locked funds as a fill settles."""
        currency = order.reserved_currency
        assert currency is not None  # noqa: S101
        assert amount > 0  # noqa: S101
        assert order.reserved_remaining >= amount  # noqa: S101
        assert self._reserved[currency] >= amount  # noqa: S101
        order.reserved_remaining -= amount
        self._reserved[currency] -= amount

    def _release_reservation(self, order: _Order) -> None:
        """Release all unconsumed funds on terminal transition."""
        currency = order.reserved_currency
        if currency is None or order.reserved_remaining == 0:
            return
        assert self._reserved[currency] >= order.reserved_remaining  # noqa: S101
        self._reserved[currency] -= order.reserved_remaining
        order.reserved_remaining = _ZERO

    def _execute_fill(
        self,
        order: _Order,
        amount: Decimal,
        price: Decimal,
        timestamp: int,
    ) -> dict[str, Any]:
        """Settle one maker or taker fill against total and reserved balances."""
        assert order.id in self._open_orders  # noqa: S101
        assert order.status in _OPEN_STATUSES  # noqa: S101
        assert _ZERO < amount <= order.remaining  # noqa: S101
        rules = self._markets[order.symbol]
        liquidity: Literal["maker", "taker"] = "maker" if order.type == "limit" else "taker"
        fee_rate = self._maker_fee_rate if liquidity == "maker" else self._taker_fee_rate
        cost = price * amount
        fee = cost * fee_rate

        if order.side == "buy":
            debit = cost + fee
            self._consume_reservation(order, debit)
            assert self._balances[rules.quote] >= debit  # noqa: S101
            self._balances[rules.quote] -= debit
            self._balances[rules.base] += amount
        else:
            self._consume_reservation(order, amount)
            assert self._balances[rules.base] >= amount  # noqa: S101
            self._balances[rules.base] -= amount
            self._balances[rules.quote] += cost - fee

        order.filled += amount
        order.remaining -= amount
        order.cost += cost
        order.fee_paid += fee
        order.updated_timestamp = timestamp
        order.last_trade_timestamp = timestamp
        order.last_liquidity = liquidity
        if order.remaining == 0:
            order.status = "filled"
            self._release_reservation(order)
            self._archive(order)
            if order.conditional_stop_price is not None:
                self._activate_conditional_stop(order, timestamp)
        else:
            order.status = "partially_filled"

        fill = {
            "id": self._next_fill_id(),
            "order": order.id,
            "timestamp": timestamp,
            "symbol": order.symbol,
            "type": order.type,
            "side": order.side,
            "price": price,
            "amount": amount,
            "cost": cost,
            "fee": {"cost": fee, "currency": rules.quote},
            "liquidity": liquidity,
        }
        if len(self._fills) >= self._max_fill_events:
            del self._fills[0]
        self._fills.append(fill)
        self._assert_accounting()
        if self._fill_sink is not None:
            self._fill_sink(deepcopy(fill))
        return fill

    def _activate_conditional_stop(self, parent: _Order, timestamp: int) -> None:
        """Create one exact refid-linked stop after a conditional entry fully fills."""
        assert parent.side == "buy"  # noqa: S101
        assert parent.conditional_stop_price is not None  # noqa: S101
        child = self._new_order(
            symbol=parent.symbol,
            order_type="stop_market",
            side="sell",
            amount=parent.filled,
            price=None,
            stop_price=parent.conditional_stop_price,
            post_only=False,
            timestamp=timestamp,
            parent_order_id=parent.id,
        )
        rules = self._markets[parent.symbol]
        reason = self._precision_or_minimum_reason(
            child.amount,
            child.stop_price,
            rules,
        )
        if reason is not None or self.get_free_balance(rules.base) < child.amount:
            self._reject(child, reason or f"insufficient free {rules.base}")
            return
        self._reserve(child, rules.base, child.amount)
        self._open_orders[child.id] = child

    def _archive(self, order: _Order) -> None:
        """Move a terminal order into bounded closed history."""
        assert order.status in _CLOSED_STATUSES  # noqa: S101
        self._open_orders.pop(order.id, None)
        if len(self._closed_orders) >= self._max_closed_orders:
            oldest_id = next(iter(self._closed_orders))
            del self._closed_orders[oldest_id]
        self._closed_orders[order.id] = order

    def cancel_order(
        self,
        order_id: str,
        symbol: str | None = None,
        *,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        """Cancel one open order and release its unfilled reservation."""
        order = self._open_orders.get(order_id)
        if order is None:
            msg = f"open order {order_id!r} not found"
            raise OrderNotFoundError(msg)
        if symbol is not None and order.symbol != symbol.strip().upper():
            msg = f"order {order_id!r} does not belong to {symbol}"
            raise OrderNotFoundError(msg)
        if client_order_id is not None and order.client_order_id != client_order_id:
            msg = f"order {order_id!r} does not have clientOrderId {client_order_id!r}"
            raise OrderNotFoundError(msg)
        order.status = "canceled"
        order.updated_timestamp = self._next_timestamp()
        self._release_reservation(order)
        self._archive(order)
        self._assert_accounting()
        return self._order_dict(order)

    def cancel_all_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """Cancel all retained open orders, optionally for one symbol."""
        normalized = self._require_symbol(symbol) if symbol is not None else None
        order_ids = [
            order.id
            for order in self._open_orders.values()
            if normalized is None or order.symbol == normalized
        ]
        return [self.cancel_order(order_id) for order_id in order_ids]

    def create_order(  # noqa: PLR0913
        self,
        symbol: str,
        type: str,  # noqa: A002
        side: str,
        amount: DecimalInput,
        price: DecimalInput | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """CCXT-shaped adapter over normalized limit and stop-market methods."""
        normalized_type = type.strip().lower().replace("-", "_")
        normalized_side = self._require_side(side)
        options = params or {}
        if normalized_type == "limit":
            if price is None:
                msg = "limit orders require price"
                raise ValueError(msg)
            return self.place_limit_order(
                symbol,
                normalized_side,
                amount,
                price,
                post_only=options.get("postOnly", True) is True,
                client_order_id=options.get("clientOrderId"),
                conditional_stop_price=(
                    options.get("close", {}).get("price")
                    if isinstance(options.get("close"), Mapping)
                    else None
                ),
            )
        if normalized_type == "market":
            if normalized_side != "sell":
                msg = "only bounded market sells are supported"
                raise ValueError(msg)
            return self.place_market_sell(
                symbol,
                amount,
                client_order_id=options.get("clientOrderId"),
                reference_price=price,
            )
        if normalized_type in {"stop_market", "stop_loss", "stop"}:
            trigger = options.get("stopPrice", options.get("price", price))
            if trigger is None:
                msg = "stop-market orders require stopPrice"
                raise ValueError(msg)
            return self.place_stop_market_order(
                symbol,
                normalized_side,
                amount,
                trigger,
                client_order_id=options.get("clientOrderId"),
            )
        msg = f"unsupported paper order type {type!r}"
        raise ValueError(msg)

    def place_maker_buy(
        self,
        symbol: str,
        price: DecimalInput,
        amount: DecimalInput,
        *,
        client_order_id: str | None = None,
        stop_loss_price: DecimalInput | None = None,
    ) -> dict[str, Any]:
        """KrakenExchange-compatible post-only buy convenience method."""
        return self.place_limit_order(
            symbol,
            "buy",
            amount,
            price,
            client_order_id=client_order_id,
            conditional_stop_price=stop_loss_price,
        )

    def normalize_limit_order(
        self,
        symbol: str,
        amount: DecimalInput,
        price: DecimalInput,
        *,
        stop_loss_price: DecimalInput | None = None,
    ) -> tuple[Decimal, Decimal, Decimal | None]:
        """Return deterministic venue-rounded limit values without mutation."""
        normalized = self._require_symbol(symbol)
        rules = self._markets[normalized]
        quantity = self._quantize(_decimal(amount, "amount"), rules.amount_increment, ROUND_DOWN)
        limit = self._quantize(_decimal(price, "price"), rules.price_increment, ROUND_DOWN)
        stop = (
            None
            if stop_loss_price is None
            else self._quantize(
                _decimal(stop_loss_price, "stop_loss_price"),
                rules.price_increment,
                ROUND_DOWN,
            )
        )
        reason = self._precision_or_minimum_reason(quantity, limit, rules)
        if reason is not None:
            raise ValueError(reason)
        if stop is not None:
            reason = self._precision_or_minimum_reason(quantity, stop, rules)
            if reason is not None:
                message = f"conditional stop {reason}"
                raise ValueError(message)
        return quantity, limit, stop

    def normalize_market_sell(
        self,
        symbol: str,
        amount: DecimalInput,
        *,
        reference_price: DecimalInput | None = None,
    ) -> tuple[Decimal, Decimal | None]:
        """Return deterministic venue-rounded market values without mutation."""
        normalized = self._require_symbol(symbol)
        rules = self._markets[normalized]
        quantity = self._quantize(_decimal(amount, "amount"), rules.amount_increment, ROUND_DOWN)
        snapshot = self._latest_markets.get(normalized)
        raw_price = (
            _decimal(reference_price, "reference_price")
            if reference_price is not None
            else (snapshot.bid if snapshot is not None else None)
        )
        if raw_price is None:
            message = "market event or reference_price is required"
            raise ValueError(message)
        price = self._quantize(raw_price, rules.price_increment, ROUND_DOWN)
        reason = self._precision_or_minimum_reason(quantity, price, rules)
        if reason is not None:
            raise ValueError(reason)
        return quantity, price

    def normalize_stop_loss(
        self,
        symbol: str,
        amount: DecimalInput,
        trigger_price: DecimalInput,
    ) -> tuple[Decimal, Decimal]:
        """Return deterministic venue-rounded stop values without mutation."""
        normalized = self._require_symbol(symbol)
        rules = self._markets[normalized]
        quantity = self._quantize(_decimal(amount, "amount"), rules.amount_increment, ROUND_DOWN)
        trigger = self._quantize(
            _decimal(trigger_price, "trigger_price"),
            rules.price_increment,
            ROUND_DOWN,
        )
        reason = self._precision_or_minimum_reason(quantity, trigger, rules)
        if reason is not None:
            raise ValueError(reason)
        return quantity, trigger

    def place_maker_sell(
        self,
        symbol: str,
        price: DecimalInput,
        amount: DecimalInput,
        *,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        """KrakenExchange-compatible post-only sell convenience method."""
        return self.place_limit_order(
            symbol,
            "sell",
            amount,
            price,
            client_order_id=client_order_id,
        )

    def place_stop_loss(
        self,
        symbol: str,
        amount: DecimalInput,
        trigger_price: DecimalInput,
        *,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        """KrakenExchange-compatible exact stop-loss convenience method."""
        return self.place_stop_market_order(
            symbol,
            "sell",
            amount,
            trigger_price,
            client_order_id=client_order_id,
        )

    def amend_stop_loss(
        self,
        order_id: str,
        symbol: str,
        amount: DecimalInput,
        trigger_price: DecimalInput,
    ) -> dict[str, Any]:
        """Amend one retained PAPER stop without releasing its reservation."""
        order = self._open_orders.get(order_id)
        normalized = self._require_symbol(symbol)
        if order is None or order.symbol != normalized or order.type != "stop_market":
            message = f"open stop order {order_id!r} not retained for {normalized}"
            raise OrderNotFoundError(message)
        quantity, trigger = self.normalize_stop_loss(normalized, amount, trigger_price)
        if quantity != order.amount:
            message = "PAPER stop amendment cannot resize reserved inventory"
            raise ValueError(message)
        snapshot = self._latest_markets.get(normalized)
        if snapshot is None or trigger >= snapshot.last:
            message = "amended sell stop must remain below current price"
            raise ValueError(message)
        order.stop_price = trigger
        order.updated_timestamp = self._next_timestamp()
        return self._order_dict(order)

    def place_protection_orders(
        self,
        symbol: str,
        amount: DecimalInput,
        _entry_price: DecimalInput,
        sl_price: DecimalInput,
        _tp_price: DecimalInput,
        *,
        client_order_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Place the reserved stop-market protection supported by spot cash."""
        return [
            self.place_stop_market_order(
                symbol,
                "sell",
                amount,
                sl_price,
                client_order_id=client_order_id,
            ),
        ]

    def get_order(self, order_id: str) -> dict[str, Any]:
        """Return a detached retained order snapshot."""
        order = self._open_orders.get(order_id) or self._closed_orders.get(order_id)
        if order is None:
            msg = f"order {order_id!r} not retained"
            raise OrderNotFoundError(msg)
        return self._order_dict(order)

    def fetch_order(
        self,
        order_id: str,
        symbol: str | None = None,
        *,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        """Fetch one retained order by exact exchange ID and optional client ID."""
        order = self.get_order(order_id)
        if symbol is not None and order["symbol"] != symbol.strip().upper():
            msg = f"order {order_id!r} does not belong to {symbol}"
            raise OrderNotFoundError(msg)
        if client_order_id is not None and order["clientOrderId"] != client_order_id:
            msg = f"order {order_id!r} does not have clientOrderId {client_order_id!r}"
            raise OrderNotFoundError(msg)
        return order

    def get_open_orders(
        self,
        symbol: str | None = None,
        *,
        client_order_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return accepted open and partially filled orders in FIFO order."""
        normalized = self._require_symbol(symbol) if symbol is not None else None
        return [
            self._order_dict(order)
            for order in self._open_orders.values()
            if normalized is None or order.symbol == normalized
            if client_order_id is None or order.client_order_id == client_order_id
        ]

    def fetch_open_orders(
        self,
        symbol: str | None = None,
        *,
        client_order_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return exact matching open orders."""
        return self.get_open_orders(symbol, client_order_id=client_order_id)

    def get_closed_orders(
        self,
        symbol: str | None = None,
        *,
        limit: int | None = None,
        client_order_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return bounded terminal history, newest first."""
        normalized = self._require_symbol(symbol) if symbol is not None else None
        if limit is not None and (isinstance(limit, bool) or limit <= 0):
            msg = "limit must be a positive integer"
            raise ValueError(msg)
        orders = [
            self._order_dict(order)
            for order in reversed(self._closed_orders.values())
            if normalized is None or order.symbol == normalized
            if client_order_id is None or order.client_order_id == client_order_id
        ]
        return orders if limit is None else orders[:limit]

    def fetch_closed_orders(
        self,
        symbol: str | None = None,
        *,
        client_order_id: str | None = None,
        since_ms: int | None = None,
        until_ms: int | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return exact matching retained terminal orders within timestamp bounds."""
        if since_ms is not None and (isinstance(since_ms, bool) or since_ms < 0):
            msg = "since_ms must be a non-negative integer"
            raise ValueError(msg)
        if until_ms is not None and (isinstance(until_ms, bool) or until_ms < 0):
            msg = "until_ms must be a non-negative integer"
            raise ValueError(msg)
        orders = self.get_closed_orders(
            symbol,
            client_order_id=client_order_id,
        )
        bounded = [
            order
            for order in orders
            if (since_ms is None or order["timestamp"] >= since_ms)
            and (until_ms is None or order["timestamp"] <= until_ms)
        ]
        return bounded if limit is None else bounded[:limit]

    def fetch_my_trades(
        self,
        symbol: str | None = None,
        since_ms: int | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return the bounded in-process fill queue, oldest first."""
        normalized = self._require_symbol(symbol) if symbol is not None else None
        if since_ms is not None and (isinstance(since_ms, bool) or since_ms < 0):
            msg = "since_ms must be a non-negative integer"
            raise ValueError(msg)
        if limit is not None and (isinstance(limit, bool) or limit <= 0):
            msg = "limit must be a positive integer"
            raise ValueError(msg)
        fills = [
            fill
            for fill in self._fills
            if (normalized is None or fill["symbol"] == normalized)
            and (since_ms is None or fill["timestamp"] >= since_ms)
        ]
        if limit is not None:
            fills = fills[-limit:]
        return deepcopy(fills)

    def fetch_account_fills(
        self,
        since_ms: int,
        end_ms: int,
        *,
        page_size: int = 50,
        max_pages: int = 20,
    ) -> list[dict[str, Any]]:
        """Return a bounded exact account-wide fill window in chronological order."""
        if isinstance(since_ms, bool) or not isinstance(since_ms, int) or since_ms < 0:
            msg = "since_ms must be a non-negative integer"
            raise ValueError(msg)
        if isinstance(end_ms, bool) or not isinstance(end_ms, int) or end_ms < since_ms:
            msg = "end_ms must be an integer not less than since_ms"
            raise ValueError(msg)
        if isinstance(page_size, bool) or page_size <= 0:
            msg = "page_size must be a positive integer"
            raise ValueError(msg)
        if isinstance(max_pages, bool) or max_pages <= 0:
            msg = "max_pages must be a positive integer"
            raise ValueError(msg)
        fills = [fill for fill in self._fills if since_ms <= fill["timestamp"] <= end_ms]
        if len(fills) > page_size * max_pages:
            msg = "paper fill window exceeds the configured pagination bound"
            raise PaperExchangeError(msg)
        for fill in fills:
            if not fill.get("id") or not fill.get("order"):
                msg = "paper fill is missing an exact fill or order id"
                raise PaperExchangeError(msg)
        return deepcopy(fills)

    def get_balance(self, currency: str) -> dict[str, Decimal]:
        """Return total, reserved/used, and free currency balance."""
        normalized = currency.strip().upper()
        total = self._balances.get(normalized, _ZERO)
        used = self._reserved.get(normalized, _ZERO)
        free = total - used
        assert total >= 0  # noqa: S101
        assert _ZERO <= used <= total  # noqa: S101
        return {"free": free, "used": used, "total": total}

    def get_free_balance(self, currency: str) -> Decimal:
        """Return balance not locked by accepted orders."""
        return self.get_balance(currency)["free"]

    def get_total_balance(self, currency: str) -> Decimal:
        """Return balance including amounts locked by accepted orders."""
        return self.get_balance(currency)["total"]

    def fetch_balance(self) -> dict[str, dict[str, Decimal]]:
        """Return a CCXT-shaped detached balance snapshot."""
        currencies = sorted(self._balances)
        return {
            key: {currency: self.get_balance(currency)[key] for currency in currencies}
            for key in ("free", "used", "total")
        }

    def has_open_position(self, symbol: str) -> bool:
        """Return whether any base asset is held, including reserved inventory."""
        normalized = self._require_symbol(symbol)
        return self.get_total_balance(self._markets[normalized].base) > 0

    def _require_symbol(self, symbol: str) -> str:
        """Normalize a symbol and reject markets without fixed rules."""
        normalized = symbol.strip().upper()
        if normalized not in self._markets:
            msg = f"unknown paper market {symbol!r}"
            raise ValueError(msg)
        return normalized

    @staticmethod
    def _require_side(side: str) -> Side:
        """Normalize and validate a spot order side."""
        normalized = side.strip().lower()
        if normalized not in {"buy", "sell"}:
            msg = f"unsupported side {side!r}"
            raise ValueError(msg)
        return normalized  # type: ignore[return-value]

    def _order_dict(self, order: _Order) -> dict[str, Any]:
        """Build a normalized CCXT-like order snapshot using Decimal values."""
        average = order.cost / order.filled if order.filled > 0 else None
        return {
            "id": order.id,
            "clientOrderId": order.client_order_id,
            "timestamp": order.timestamp,
            "lastTradeTimestamp": order.last_trade_timestamp,
            "symbol": order.symbol,
            "type": order.type,
            "side": order.side,
            "price": order.price,
            "stopPrice": order.stop_price,
            "stopLossPrice": order.stop_price if order.type == "stop_market" else None,
            "amount": order.amount,
            "filled": order.filled,
            "remaining": order.remaining,
            "cost": order.cost,
            "average": average,
            "status": order.status,
            "postOnly": order.post_only,
            "fee": {"cost": order.fee_paid, "currency": order.fee_currency},
            "info": {
                "updated_timestamp": order.updated_timestamp,
                "created_event_sequence": order.created_event_sequence,
                "triggered": order.triggered,
                "liquidity": order.last_liquidity,
                "reject_reason": order.reject_reason,
                "cl_ord_id": order.client_order_id,
                "refid": order.parent_order_id,
                "descr": {
                    "ordertype": "stop-loss" if order.type == "stop_market" else order.type,
                },
            },
        }

    def _assert_accounting(self) -> None:
        """Check paired total/free and per-order reservation invariants."""
        for currency, total in self._balances.items():
            reserved = self._reserved.get(currency, _ZERO)
            assert total >= 0, f"negative total balance for {currency}"  # noqa: S101
            assert reserved >= 0, f"negative reservation for {currency}"  # noqa: S101
            assert reserved <= total, f"reservation exceeds {currency} total"  # noqa: S101

        order_reservations = dict.fromkeys(self._balances, _ZERO)
        for order in self._open_orders.values():
            assert order.status in _OPEN_STATUSES  # noqa: S101
            assert order.reserved_currency is not None  # noqa: S101
            assert order.reserved_remaining >= 0  # noqa: S101
            order_reservations.setdefault(order.reserved_currency, _ZERO)
            order_reservations[order.reserved_currency] += order.reserved_remaining
        assert order_reservations == self._reserved, "reservation totals do not match orders"  # noqa: S101

    def to_state(self) -> dict[str, Any]:
        """Return JSON-safe restart state, intentionally excluding individual fills."""
        return {
            "version": self.STATE_VERSION,
            "config": {
                "maker_fee_rate": str(self._maker_fee_rate),
                "taker_fee_rate": str(self._taker_fee_rate),
                "stop_slippage": str(self._stop_slippage),
                "max_open_orders": self._max_open_orders,
                "max_closed_orders": self._max_closed_orders,
                "max_fill_events": self._max_fill_events,
                "max_markets": self._max_markets,
            },
            "markets": {symbol: rules.to_state() for symbol, rules in self._markets.items()},
            "balances": {
                "total": {currency: str(value) for currency, value in self._balances.items()},
                "reserved": {currency: str(value) for currency, value in self._reserved.items()},
            },
            "latest_markets": {
                symbol: {
                    "bid": str(snapshot.bid),
                    "ask": str(snapshot.ask),
                    "last": str(snapshot.last),
                    "event_sequence": snapshot.event_sequence,
                    "timestamp": snapshot.timestamp,
                }
                for symbol, snapshot in self._latest_markets.items()
            },
            "open_orders": [self._order_state(order) for order in self._open_orders.values()],
            "closed_orders": [self._order_state(order) for order in self._closed_orders.values()],
            "sequences": {
                "clock": self._clock,
                "event": self._event_sequence,
                "order": self._order_sequence,
                "fill": self._fill_sequence,
            },
        }

    export_state = to_state

    @staticmethod
    def _order_state(order: _Order) -> dict[str, Any]:
        """Serialize all aggregate order fields needed for exact continuation."""
        return {
            "id": order.id,
            "timestamp": order.timestamp,
            "updated_timestamp": order.updated_timestamp,
            "symbol": order.symbol,
            "type": order.type,
            "side": order.side,
            "amount": str(order.amount),
            "price": None if order.price is None else str(order.price),
            "stop_price": None if order.stop_price is None else str(order.stop_price),
            "post_only": order.post_only,
            "status": order.status,
            "filled": str(order.filled),
            "remaining": str(order.remaining),
            "cost": str(order.cost),
            "fee_paid": str(order.fee_paid),
            "fee_currency": order.fee_currency,
            "created_event_sequence": order.created_event_sequence,
            "triggered": order.triggered,
            "reserved_currency": order.reserved_currency,
            "reserved_remaining": str(order.reserved_remaining),
            "reject_reason": order.reject_reason,
            "last_trade_timestamp": order.last_trade_timestamp,
            "last_liquidity": order.last_liquidity,
            "client_order_id": order.client_order_id,
            "conditional_stop_price": (
                None if order.conditional_stop_price is None else str(order.conditional_stop_price)
            ),
            "parent_order_id": order.parent_order_id,
        }

    @classmethod
    def from_state(
        cls,
        state: Mapping[str, Any],
        *,
        fill_sink: FillSink | None = None,
    ) -> PaperExchange:
        """Restore an exact checkpoint; historical fills remain in the external ledger."""
        if state.get("version") != cls.STATE_VERSION:
            msg = f"unsupported paper exchange state version {state.get('version')!r}"
            raise ValueError(msg)
        config = state["config"]
        balances = state["balances"]
        exchange = cls(
            initial_balances=balances["total"],
            markets=state["markets"],
            maker_fee_rate=config["maker_fee_rate"],
            taker_fee_rate=config["taker_fee_rate"],
            stop_slippage=config["stop_slippage"],
            max_open_orders=config["max_open_orders"],
            max_closed_orders=config["max_closed_orders"],
            max_fill_events=config["max_fill_events"],
            max_markets=config["max_markets"],
            start_timestamp=state["sequences"]["clock"],
            fill_sink=fill_sink,
        )
        exchange._reserved = {
            currency: _decimal(value, f"reserved[{currency}]")
            for currency, value in balances["reserved"].items()
        }
        exchange._latest_markets = {
            symbol: _MarketSnapshot(
                bid=_decimal(snapshot["bid"], "bid"),
                ask=_decimal(snapshot["ask"], "ask"),
                last=_decimal(snapshot["last"], "last"),
                event_sequence=snapshot["event_sequence"],
                timestamp=snapshot["timestamp"],
            )
            for symbol, snapshot in state["latest_markets"].items()
        }
        exchange._open_orders = {
            order_state["id"]: exchange._order_from_state(order_state)
            for order_state in state["open_orders"]
        }
        exchange._closed_orders = {
            order_state["id"]: exchange._order_from_state(order_state)
            for order_state in state["closed_orders"]
        }
        if len(exchange._open_orders) > exchange._max_open_orders:
            msg = "checkpoint exceeds max_open_orders"
            raise ValueError(msg)
        if len(exchange._closed_orders) > exchange._max_closed_orders:
            msg = "checkpoint exceeds max_closed_orders"
            raise ValueError(msg)
        sequences = state["sequences"]
        exchange._event_sequence = sequences["event"]
        exchange._order_sequence = sequences["order"]
        exchange._fill_sequence = sequences["fill"]
        exchange._fills = []
        exchange._assert_restored_orders()
        exchange._assert_accounting()
        return exchange

    @staticmethod
    def _order_from_state(state: Mapping[str, Any]) -> _Order:
        """Deserialize one aggregate order record."""
        return _Order(
            id=state["id"],
            timestamp=state["timestamp"],
            updated_timestamp=state["updated_timestamp"],
            symbol=state["symbol"],
            type=state["type"],
            side=state["side"],
            amount=_decimal(state["amount"], "amount"),
            price=None if state["price"] is None else _decimal(state["price"], "price"),
            stop_price=(
                None if state["stop_price"] is None else _decimal(state["stop_price"], "stop_price")
            ),
            post_only=state["post_only"],
            status=state["status"],
            filled=_decimal(state["filled"], "filled"),
            remaining=_decimal(state["remaining"], "remaining"),
            cost=_decimal(state["cost"], "cost"),
            fee_paid=_decimal(state["fee_paid"], "fee_paid"),
            fee_currency=state["fee_currency"],
            created_event_sequence=state["created_event_sequence"],
            triggered=state["triggered"],
            reserved_currency=state["reserved_currency"],
            reserved_remaining=_decimal(state["reserved_remaining"], "reserved_remaining"),
            reject_reason=state["reject_reason"],
            last_trade_timestamp=state["last_trade_timestamp"],
            last_liquidity=state["last_liquidity"],
            client_order_id=state["client_order_id"],
            conditional_stop_price=(
                None
                if state["conditional_stop_price"] is None
                else _decimal(state["conditional_stop_price"], "conditional_stop_price")
            ),
            parent_order_id=state["parent_order_id"],
        )

    def _assert_restored_orders(self) -> None:
        """Reject malformed checkpoint order placement and terminal states."""
        for order in self._open_orders.values():
            if order.symbol not in self._markets or order.status not in _OPEN_STATUSES:
                msg = f"invalid restored open order {order.id}"
                raise ValueError(msg)
        for order in self._closed_orders.values():
            if order.symbol not in self._markets or order.status not in _CLOSED_STATUSES:
                msg = f"invalid restored closed order {order.id}"
                raise ValueError(msg)


__all__ = [
    "MarketRules",
    "OrderNotFoundError",
    "PaperExchange",
    "PaperExchangeError",
]
