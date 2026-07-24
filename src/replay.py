"""Adapter-driven, production-parity market replay foundation.

The engine owns time, event ordering, scoring boundaries, and measurement.  It
does not own strategy or matching decisions: a supplied cycle callback drives a
supplied strategy adapter against a supplied exchange adapter.  This keeps a
later extraction of the production main cycle mechanical rather than creating a
second strategy implementation inside the backtester.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any, Generic, Protocol, TypeAlias, TypeVar, cast

from market_clock import (
    AsOfSnapshot,
    ClosedCandlePolicy,
    MarketClock,
    Milliseconds,
    Observation,
    ObservationStatus,
    ObservationStore,
    observation_order_key,
)

DecimalInput: TypeAlias = Decimal | str | int | float
_ZERO = Decimal(0)
_ONE = Decimal(1)
_BASIS_POINTS = Decimal(10_000)
_DAY_MS = 86_400_000


def _decimal(value: DecimalInput, name: str) -> Decimal:
    """Convert financial inputs without inheriting binary float arithmetic."""
    if isinstance(value, bool):
        message = f"{name} must be numeric"
        raise TypeError(message)
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        message = f"{name} must be a finite decimal"
        raise ValueError(message) from exc
    if not result.is_finite():
        message = f"{name} must be a finite decimal"
        raise ValueError(message)
    return result


def _require_non_negative_ms(value: int, name: str) -> None:
    """Validate millisecond parameters independently of market_clock internals."""
    if isinstance(value, bool) or not isinstance(value, int):
        message = f"{name} must be an integer number of milliseconds"
        raise TypeError(message)
    if value < 0:
        message = f"{name} must be non-negative"
        raise ValueError(message)


@dataclass(frozen=True, slots=True)
class TimeWindow:
    """Half-open event-time interval."""

    start_ms: Milliseconds
    end_ms: Milliseconds

    def __post_init__(self) -> None:
        """Require a non-empty, forward interval."""
        _require_non_negative_ms(self.start_ms, "start_ms")
        _require_non_negative_ms(self.end_ms, "end_ms")
        if self.end_ms <= self.start_ms:
            message = "end_ms must be greater than start_ms"
            raise ValueError(message)

    def contains(self, at_ms: Milliseconds) -> bool:
        """Return whether the instant lies in this half-open interval."""
        return self.start_ms <= at_ms < self.end_ms


@dataclass(frozen=True, slots=True)
class ReplayWindow:
    """Warmup plus scored periods for one replay run."""

    warmup_start_ms: Milliseconds
    scored_start_ms: Milliseconds
    scored_end_ms: Milliseconds

    def __post_init__(self) -> None:
        """Require ordered, non-empty scored boundaries."""
        _require_non_negative_ms(self.warmup_start_ms, "warmup_start_ms")
        _require_non_negative_ms(self.scored_start_ms, "scored_start_ms")
        _require_non_negative_ms(self.scored_end_ms, "scored_end_ms")
        if self.scored_start_ms < self.warmup_start_ms:
            message = "scored_start_ms must not precede warmup_start_ms"
            raise ValueError(message)
        if self.scored_end_ms <= self.scored_start_ms:
            message = "scored_end_ms must be greater than scored_start_ms"
            raise ValueError(message)

    @property
    def warmup(self) -> TimeWindow | None:
        """Return the warmup interval, or None when scoring starts immediately."""
        if self.warmup_start_ms == self.scored_start_ms:
            return None
        return TimeWindow(self.warmup_start_ms, self.scored_start_ms)

    @property
    def scored(self) -> TimeWindow:
        """Return the half-open scored interval."""
        return TimeWindow(self.scored_start_ms, self.scored_end_ms)

    def is_scored(self, at_ms: Milliseconds) -> bool:
        """Return whether an event contributes to reported performance."""
        return self.scored.contains(at_ms)


def purge_overlapping_labels(
    observations: Iterable[Observation],
    *,
    scored_start_ms: Milliseconds,
    label_horizon_ms: Milliseconds,
) -> tuple[Observation, ...]:
    """Remove training rows whose forward labels touch the scored period."""
    _require_non_negative_ms(scored_start_ms, "scored_start_ms")
    _require_non_negative_ms(label_horizon_ms, "label_horizon_ms")
    return tuple(
        observation
        for observation in observations
        if observation.event_at_ms + label_horizon_ms < scored_start_ms
    )


def embargo_after_window(
    observations: Iterable[Observation],
    *,
    scored_end_ms: Milliseconds,
    embargo_ms: Milliseconds,
) -> tuple[Observation, ...]:
    """Remove rows immediately after a scored interval from a later train set."""
    _require_non_negative_ms(scored_end_ms, "scored_end_ms")
    _require_non_negative_ms(embargo_ms, "embargo_ms")
    embargo_end_ms = scored_end_ms + embargo_ms
    return tuple(
        observation
        for observation in observations
        if not scored_end_ms <= observation.event_at_ms < embargo_end_ms
    )


def purge_and_embargo(
    observations: Iterable[Observation],
    *,
    scored_window: TimeWindow,
    label_horizon_ms: Milliseconds,
    embargo_ms: Milliseconds,
) -> tuple[Observation, ...]:
    """Exclude both label-overlap and post-score leakage zones."""
    _require_non_negative_ms(label_horizon_ms, "label_horizon_ms")
    _require_non_negative_ms(embargo_ms, "embargo_ms")
    purge_start_ms = max(0, scored_window.start_ms - label_horizon_ms)
    embargo_end_ms = scored_window.end_ms + embargo_ms
    return tuple(
        observation
        for observation in observations
        if not purge_start_ms <= observation.event_at_ms < embargo_end_ms
    )


@dataclass(frozen=True, slots=True)
class ReplayEvent:
    """Observation release with stable queue and strictly increasing logical time."""

    observation: Observation
    replay_at_ms: Milliseconds
    logical_at_ms: Milliseconds
    sequence: int


@dataclass(frozen=True, slots=True)
class ReplayFill:
    """Normalized economic result of one exchange fill."""

    fill_id: str
    order_id: str
    at_ms: Milliseconds
    symbol: str
    side: str
    quantity: Decimal
    price: Decimal
    fee: Decimal = _ZERO
    gross_pnl: Decimal = _ZERO
    slippage_cost: Decimal = _ZERO

    def __post_init__(self) -> None:
        """Normalize financial values and reject malformed fills."""
        _require_non_negative_ms(self.at_ms, "at_ms")
        if not self.fill_id.strip():
            message = "fill_id must be non-empty"
            raise ValueError(message)
        if not self.order_id.strip():
            message = "order_id must be non-empty"
            raise ValueError(message)
        if not self.symbol.strip():
            message = "symbol must be non-empty"
            raise ValueError(message)
        side = self.side.strip().lower()
        if side not in {"buy", "sell"}:
            message = f"unsupported fill side {self.side!r}"
            raise ValueError(message)
        quantity = _decimal(self.quantity, "quantity")
        price = _decimal(self.price, "price")
        fee = _decimal(self.fee, "fee")
        gross_pnl = _decimal(self.gross_pnl, "gross_pnl")
        slippage_cost = _decimal(self.slippage_cost, "slippage_cost")
        if quantity <= 0:
            message = "quantity must be positive"
            raise ValueError(message)
        if price <= 0:
            message = "price must be positive"
            raise ValueError(message)
        if fee < 0:
            message = "fee must be non-negative"
            raise ValueError(message)
        if slippage_cost < 0:
            message = "slippage_cost must be non-negative"
            raise ValueError(message)
        object.__setattr__(self, "fill_id", self.fill_id.strip())
        object.__setattr__(self, "order_id", self.order_id.strip())
        object.__setattr__(self, "symbol", self.symbol.strip().upper())
        object.__setattr__(self, "side", side)
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "price", price)
        object.__setattr__(self, "fee", fee)
        object.__setattr__(self, "gross_pnl", gross_pnl)
        object.__setattr__(self, "slippage_cost", slippage_cost)

    @property
    def notional(self) -> Decimal:
        """Return exact quote notional."""
        return self.quantity * self.price

    @property
    def net_pnl(self) -> Decimal:
        """Return fee- and slippage-adjusted economic PnL."""
        return self.gross_pnl - self.fee - self.slippage_cost

    @classmethod
    def from_paper_fill(
        cls,
        fill: Mapping[str, Any],
        *,
        gross_pnl: DecimalInput = _ZERO,
        reference_price: DecimalInput | None = None,
    ) -> ReplayFill:
        """Normalize the public fill shape emitted by ``PaperExchange``."""
        price = _decimal(cast("DecimalInput", fill["price"]), "price")
        quantity = _decimal(cast("DecimalInput", fill["amount"]), "amount")
        side = str(fill["side"]).lower()
        slippage = _ZERO
        if reference_price is not None:
            reference = _decimal(reference_price, "reference_price")
            price_delta = price - reference if side == "buy" else reference - price
            slippage = max(_ZERO, price_delta * quantity)
        fee_value = cast("Mapping[str, Any]", fill.get("fee", {})).get("cost", _ZERO)
        return cls(
            fill_id=str(fill["id"]),
            order_id=str(fill["order"]),
            at_ms=int(fill["timestamp"]),
            symbol=str(fill["symbol"]),
            side=side,
            quantity=quantity,
            price=price,
            fee=_decimal(cast("DecimalInput", fee_value), "fee"),
            gross_pnl=_decimal(gross_pnl, "gross_pnl"),
            slippage_cost=slippage,
        )


@dataclass(frozen=True, slots=True)
class SignalEvent:
    """One strategy signal sample, before episode de-duplication."""

    at_ms: Milliseconds
    symbol: str
    action: str
    strength: Decimal = _ZERO

    def __post_init__(self) -> None:
        """Normalize signal identity and strength."""
        _require_non_negative_ms(self.at_ms, "at_ms")
        symbol = self.symbol.strip().upper()
        action = self.action.strip().upper()
        if not symbol:
            message = "signal symbol must be non-empty"
            raise ValueError(message)
        if not action:
            message = "signal action must be non-empty"
            raise ValueError(message)
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "strength", _decimal(self.strength, "strength"))


@dataclass(frozen=True, slots=True)
class SignalEpisode:
    """Consecutive samples of one actionable signal counted as one decision."""

    symbol: str
    action: str
    started_at_ms: Milliseconds
    ended_at_ms: Milliseconds
    sample_count: int
    peak_strength: Decimal


def deduplicate_signal_episodes(
    signals: Iterable[SignalEvent],
    *,
    max_gap_ms: Milliseconds | None = None,
    neutral_actions: frozenset[str] = frozenset({"HOLD", "FLAT", "NONE"}),
) -> tuple[SignalEpisode, ...]:
    """Collapse repeated cycle signals until neutral, reversal, or a long gap."""
    if max_gap_ms is not None:
        _require_non_negative_ms(max_gap_ms, "max_gap_ms")
    normalized_neutral = frozenset(action.strip().upper() for action in neutral_actions)
    ordered = sorted(signals, key=lambda signal: (signal.at_ms, signal.symbol, signal.action))
    active: dict[str, list[SignalEvent]] = {}
    episodes: list[SignalEpisode] = []

    def close_episode(samples: list[SignalEvent]) -> None:
        first = samples[0]
        episodes.append(
            SignalEpisode(
                symbol=first.symbol,
                action=first.action,
                started_at_ms=first.at_ms,
                ended_at_ms=samples[-1].at_ms,
                sample_count=len(samples),
                peak_strength=max((abs(sample.strength) for sample in samples), default=_ZERO),
            ),
        )

    for signal in ordered:
        current = active.get(signal.symbol)
        if signal.action in normalized_neutral:
            if current is not None:
                close_episode(current)
                del active[signal.symbol]
            continue
        gap_exceeded = (
            current is not None
            and max_gap_ms is not None
            and signal.at_ms - current[-1].at_ms > max_gap_ms
        )
        action_changed = current is not None and signal.action != current[-1].action
        if current is not None and (gap_exceeded or action_changed):
            close_episode(current)
            current = None
        if current is None:
            active[signal.symbol] = [signal]
        else:
            current.append(signal)

    for samples in active.values():
        close_episode(samples)
    episodes.sort(key=lambda episode: (episode.started_at_ms, episode.symbol, episode.action))
    return tuple(episodes)


def signal_autocorrelation(values: Sequence[DecimalInput], lag: int = 1) -> float:
    """Return finite sample autocorrelation for numeric signal scores."""
    if isinstance(lag, bool) or not isinstance(lag, int) or lag <= 0:
        message = "lag must be a positive integer"
        raise ValueError(message)
    numbers = [float(_decimal(value, "signal value")) for value in values]
    if len(numbers) <= lag:
        return 0.0
    mean = sum(numbers) / len(numbers)
    denominator = sum((value - mean) ** 2 for value in numbers)
    if denominator == 0:
        return 0.0
    numerator = sum(
        (numbers[index] - mean) * (numbers[index - lag] - mean)
        for index in range(lag, len(numbers))
    )
    return max(-1.0, min(1.0, numerator / denominator))


def effective_sample_size(
    values: Sequence[DecimalInput],
    *,
    max_lag: int | None = None,
) -> float:
    """Estimate independent signal count using the initial-positive ACF sum."""
    sample_count = len(values)
    if sample_count < 2:  # noqa: PLR2004
        return float(sample_count)
    if max_lag is None:
        max_lag = min(sample_count - 1, max(1, int(math.sqrt(sample_count))))
    if isinstance(max_lag, bool) or not isinstance(max_lag, int) or max_lag <= 0:
        message = "max_lag must be a positive integer"
        raise ValueError(message)
    max_lag = min(max_lag, sample_count - 1)
    positive_sum = 0.0
    for lag in range(1, max_lag + 1):
        correlation = signal_autocorrelation(values, lag)
        if correlation <= 0:
            break
        positive_sum += correlation
    estimate = sample_count / (1.0 + 2.0 * positive_sum)
    return max(1.0, min(float(sample_count), estimate))


@dataclass(frozen=True, slots=True)
class CycleResult:
    """Optional metadata returned by the production cycle callback."""

    signals: tuple[SignalEvent, ...] = ()


@dataclass(frozen=True, slots=True)
class ReplayCycleContext:
    """Future-proof cycle input containing only data visible at this instant."""

    event: ReplayEvent | None
    snapshot: AsOfSnapshot
    window: ReplayWindow
    candle_policy: ClosedCandlePolicy | None = None

    @property
    def now_ms(self) -> Milliseconds:
        """Return real replay time rather than tie-breaking logical time."""
        return self.snapshot.as_of_ms

    @property
    def is_scored(self) -> bool:
        """Return whether this cycle belongs to the scored interval."""
        return self.window.is_scored(self.now_ms)

    def closed_candles(self, source: str) -> tuple[Observation, ...]:
        """Return policy-approved candles available to the cycle."""
        if self.candle_policy is None:
            message = "closed_candles requires a ClosedCandlePolicy"
            raise RuntimeError(message)
        return self.candle_policy.select(self.snapshot, source)


class ReplayExchangeAdapter(Protocol):
    """Minimum boundary an exchange simulator exposes to replay."""

    def process_event(self, event: ReplayEvent) -> Sequence[ReplayFill]:
        """Apply market data and return fills generated before this cycle."""
        ...

    def mark_to_market(self, context: ReplayCycleContext) -> DecimalInput:
        """Return total account equity in one quote currency."""
        ...


PaperExchangeT = TypeVar("PaperExchangeT")
StrategyT = TypeVar("StrategyT")
ExchangeAdapterT = TypeVar("ExchangeAdapterT", bound=ReplayExchangeAdapter)
CycleCallback: TypeAlias = Callable[
    [ReplayCycleContext, ExchangeAdapterT, StrategyT],
    CycleResult | None,
]


@dataclass(frozen=True, slots=True)
class PaperMarketEvent:
    """Normalized quote passed to an existing ``PaperExchange`` instance."""

    symbol: str
    bid: Decimal
    ask: Decimal
    last: Decimal
    available_liquidity: Decimal | None = None

    @classmethod
    def from_observation(cls, observation: Observation) -> PaperMarketEvent | None:
        """Read a conventional quote payload, ignoring non-market observations."""
        if observation.status is not ObservationStatus.VALID:
            return None
        if not isinstance(observation.payload, Mapping):
            return None
        required = {"symbol", "bid", "ask", "last"}
        if not required.issubset(observation.payload):
            return None
        liquidity = observation.payload.get("available_liquidity")
        return cls(
            symbol=str(observation.payload["symbol"]),
            bid=_decimal(cast("DecimalInput", observation.payload["bid"]), "bid"),
            ask=_decimal(cast("DecimalInput", observation.payload["ask"]), "ask"),
            last=_decimal(cast("DecimalInput", observation.payload["last"]), "last"),
            available_liquidity=(
                None
                if liquidity is None
                else _decimal(cast("DecimalInput", liquidity), "available_liquidity")
            ),
        )


PaperEventMapper: TypeAlias = Callable[[Observation], PaperMarketEvent | None]
PaperFillMapper: TypeAlias = Callable[[Mapping[str, Any]], ReplayFill]
EquityCallback: TypeAlias = Callable[[PaperExchangeT, ReplayCycleContext], DecimalInput]


class PaperExchangeReplayAdapter(Generic[PaperExchangeT]):
    """Thin adapter that delegates matching to a supplied ``PaperExchange``."""

    def __init__(
        self,
        exchange: PaperExchangeT,
        *,
        equity_callback: EquityCallback[PaperExchangeT],
        event_mapper: PaperEventMapper = PaperMarketEvent.from_observation,
        fill_mapper: PaperFillMapper = ReplayFill.from_paper_fill,
    ) -> None:
        """Bind data conversion and valuation without embedding strategy rules."""
        self.exchange = exchange
        self._equity_callback = equity_callback
        self._event_mapper = event_mapper
        self._fill_mapper = fill_mapper

    def process_event(self, event: ReplayEvent) -> tuple[ReplayFill, ...]:
        """Publish a quote to the wrapped exchange using strict logical time."""
        market_event = self._event_mapper(event.observation)
        if market_event is None:
            return ()
        process_market_event = getattr(self.exchange, "process_market_event", None)
        if not callable(process_market_event):
            message = "wrapped exchange must provide process_market_event"
            raise TypeError(message)
        exchange_timestamp = self._exchange_timestamp(event.logical_at_ms)
        fills = process_market_event(
            market_event.symbol,
            bid=market_event.bid,
            ask=market_event.ask,
            last=market_event.last,
            available_liquidity=market_event.available_liquidity,
            timestamp=exchange_timestamp,
        )
        return tuple(replace(self._fill_mapper(fill), at_ms=event.replay_at_ms) for fill in fills)

    def _exchange_timestamp(self, proposed_at_ms: Milliseconds) -> Milliseconds:
        """Advance beyond order timestamps created inside the previous cycle."""
        export_state = getattr(self.exchange, "to_state", None)
        if not callable(export_state):
            return proposed_at_ms
        state = cast("Mapping[str, Any]", export_state())
        sequences = cast("Mapping[str, Any]", state.get("sequences", {}))
        current_at_ms = sequences.get("clock")
        if isinstance(current_at_ms, bool) or not isinstance(current_at_ms, int):
            message = "wrapped exchange state must expose integer sequences.clock"
            raise TypeError(message)
        return max(proposed_at_ms, current_at_ms + 1)

    def mark_to_market(self, context: ReplayCycleContext) -> Decimal:
        """Delegate account valuation to the supplied production-aware callback."""
        return _decimal(self._equity_callback(self.exchange, context), "equity")


@dataclass(frozen=True, slots=True)
class EquityPoint:
    """Exact account equity at a scored replay instant."""

    at_ms: Milliseconds
    equity: Decimal


@dataclass(frozen=True, slots=True)
class DailyTradeCount:
    """UTC calendar-day fill count, including explicit zero-trade days."""

    date: str
    trades: int


@dataclass(frozen=True, slots=True)
class StressScenario:
    """Counterfactual execution-cost multipliers applied after replay."""

    fee_multiplier: Decimal = _ONE
    additional_slippage_bps: Decimal = _ZERO

    def __post_init__(self) -> None:
        """Normalize and validate non-negative stress parameters."""
        fee_multiplier = _decimal(self.fee_multiplier, "fee_multiplier")
        slippage_bps = _decimal(self.additional_slippage_bps, "additional_slippage_bps")
        if fee_multiplier < 0:
            message = "fee_multiplier must be non-negative"
            raise ValueError(message)
        if slippage_bps < 0:
            message = "additional_slippage_bps must be non-negative"
            raise ValueError(message)
        object.__setattr__(self, "fee_multiplier", fee_multiplier)
        object.__setattr__(self, "additional_slippage_bps", slippage_bps)


@dataclass(frozen=True, slots=True)
class StressMetrics:
    """Economic metrics under one fee/slippage scenario."""

    fees: Decimal
    slippage_cost: Decimal
    net_pnl: Decimal


@dataclass(frozen=True, slots=True)
class ReplayMetrics:
    """Fee-aware scored-window performance summary."""

    trade_count: int
    traded_notional: Decimal
    gross_pnl: Decimal
    fees: Decimal
    slippage_cost: Decimal
    net_pnl: Decimal
    starting_equity: Decimal
    ending_equity: Decimal
    equity_change: Decimal
    return_pct: Decimal
    max_drawdown: Decimal
    max_drawdown_pct: Decimal
    total_days: int
    zero_trade_days: int

    def stress(self, fills: Sequence[ReplayFill], scenario: StressScenario) -> StressMetrics:
        """Reprice execution costs while leaving supplied gross economics fixed."""
        stressed_fees = sum((fill.fee for fill in fills), start=_ZERO) * scenario.fee_multiplier
        actual_slippage = sum((fill.slippage_cost for fill in fills), start=_ZERO)
        notional = sum((fill.notional for fill in fills), start=_ZERO)
        stressed_slippage = (
            actual_slippage + notional * scenario.additional_slippage_bps / _BASIS_POINTS
        )
        gross_pnl = sum((fill.gross_pnl for fill in fills), start=_ZERO)
        return StressMetrics(
            fees=stressed_fees,
            slippage_cost=stressed_slippage,
            net_pnl=gross_pnl - stressed_fees - stressed_slippage,
        )


@dataclass(frozen=True, slots=True)
class ReplayReport:
    """Complete immutable replay output."""

    window: ReplayWindow
    fills: tuple[ReplayFill, ...]
    signals: tuple[SignalEvent, ...]
    signal_episodes: tuple[SignalEpisode, ...]
    equity_curve: tuple[EquityPoint, ...]
    daily_trade_counts: tuple[DailyTradeCount, ...]
    metrics: ReplayMetrics

    def stress(self, scenario: StressScenario) -> StressMetrics:
        """Return fee/slippage sensitivity for the scored fills."""
        return self.metrics.stress(self.fills, scenario)


def _utc_date(at_ms: Milliseconds) -> str:
    """Convert Unix milliseconds to a stable UTC date key."""
    return datetime.fromtimestamp(at_ms / 1000, tz=UTC).date().isoformat()


def _daily_trade_counts(
    window: ReplayWindow,
    fills: Sequence[ReplayFill],
) -> tuple[DailyTradeCount, ...]:
    """Build a dense UTC day series over the scored interval."""
    first_date = datetime.fromtimestamp(window.scored_start_ms / 1000, tz=UTC).date()
    last_instant_ms = window.scored_end_ms - 1
    last_date = datetime.fromtimestamp(last_instant_ms / 1000, tz=UTC).date()
    counts: dict[str, int] = defaultdict(int)
    for fill in fills:
        counts[_utc_date(fill.at_ms)] += 1
    days: list[DailyTradeCount] = []
    current = first_date
    while current <= last_date:
        date_key = current.isoformat()
        days.append(DailyTradeCount(date=date_key, trades=counts[date_key]))
        current += timedelta(days=1)
    return tuple(days)


def _drawdown(equity_curve: Sequence[EquityPoint]) -> tuple[Decimal, Decimal]:
    """Return maximum absolute and percentage peak-to-trough drawdown."""
    if not equity_curve:
        return _ZERO, _ZERO
    peak = equity_curve[0].equity
    maximum_amount = _ZERO
    maximum_pct = _ZERO
    for point in equity_curve:
        peak = max(peak, point.equity)
        amount = peak - point.equity
        percentage = amount / peak * 100 if peak > 0 else _ZERO
        maximum_amount = max(maximum_amount, amount)
        maximum_pct = max(maximum_pct, percentage)
    return maximum_amount, maximum_pct


def _build_metrics(
    fills: Sequence[ReplayFill],
    equity_curve: Sequence[EquityPoint],
    daily_counts: Sequence[DailyTradeCount],
) -> ReplayMetrics:
    """Aggregate exact economic and mark-to-market measurements."""
    starting_equity = equity_curve[0].equity
    ending_equity = equity_curve[-1].equity
    equity_change = ending_equity - starting_equity
    return_pct = equity_change / starting_equity * 100 if starting_equity != 0 else _ZERO
    gross_pnl = sum((fill.gross_pnl for fill in fills), start=_ZERO)
    fees = sum((fill.fee for fill in fills), start=_ZERO)
    slippage_cost = sum((fill.slippage_cost for fill in fills), start=_ZERO)
    maximum_drawdown, maximum_drawdown_pct = _drawdown(equity_curve)
    return ReplayMetrics(
        trade_count=len(fills),
        traded_notional=sum((fill.notional for fill in fills), start=_ZERO),
        gross_pnl=gross_pnl,
        fees=fees,
        slippage_cost=slippage_cost,
        net_pnl=gross_pnl - fees - slippage_cost,
        starting_equity=starting_equity,
        ending_equity=ending_equity,
        equity_change=equity_change,
        return_pct=return_pct,
        max_drawdown=maximum_drawdown,
        max_drawdown_pct=maximum_drawdown_pct,
        total_days=len(daily_counts),
        zero_trade_days=sum(day.trades == 0 for day in daily_counts),
    )


class ReplayEngine(Generic[ExchangeAdapterT, StrategyT]):
    """Drive deterministic observations through supplied production adapters."""

    def __init__(  # noqa: PLR0913
        self,
        observations: Iterable[Observation],
        *,
        exchange_adapter: ExchangeAdapterT,
        strategy_adapter: StrategyT,
        cycle_callback: CycleCallback[ExchangeAdapterT, StrategyT],
        candle_policy: ClosedCandlePolicy | None = None,
        cycle_trigger: Callable[[ReplayCycleContext], bool] | None = None,
        signal_episode_gap_ms: Milliseconds | None = None,
    ) -> None:
        """Bind fixed data and injected behavior; no strategy logic is copied here."""
        if signal_episode_gap_ms is not None:
            _require_non_negative_ms(signal_episode_gap_ms, "signal_episode_gap_ms")
        self._store = ObservationStore(observations)
        self._exchange_adapter = exchange_adapter
        self._strategy_adapter = strategy_adapter
        self._cycle_callback = cycle_callback
        self._candle_policy = candle_policy
        self._cycle_trigger = cycle_trigger or (lambda _context: True)
        self._signal_episode_gap_ms = signal_episode_gap_ms

    def _release_at_ms(self, observation: Observation) -> Milliseconds:
        """Apply candle-close gating when the policy covers this source."""
        if self._candle_policy is None:
            return observation.known_at_ms
        return self._candle_policy.replay_at_ms(observation)

    def _ordered_events(self, window: ReplayWindow) -> tuple[ReplayEvent, ...]:
        """Build a stable queue and a strict exchange-safe logical timestamp."""
        queued = sorted(
            (
                (self._release_at_ms(observation), observation)
                for observation in self._store.observations
                if window.warmup_start_ms <= self._release_at_ms(observation) < window.scored_end_ms
            ),
            key=lambda item: (item[0], observation_order_key(item[1])),
        )
        events: list[ReplayEvent] = []
        logical_at_ms = max(0, window.warmup_start_ms - 1)
        for sequence, (replay_at_ms, observation) in enumerate(queued, start=1):
            logical_at_ms = max(replay_at_ms, logical_at_ms + 1)
            events.append(
                ReplayEvent(
                    observation=observation,
                    replay_at_ms=replay_at_ms,
                    logical_at_ms=logical_at_ms,
                    sequence=sequence,
                ),
            )
        return tuple(events)

    def run(self, window: ReplayWindow) -> ReplayReport:
        """Execute warmup and scored events, returning exact immutable metrics."""
        clock = MarketClock(self._store, start_at_ms=window.warmup_start_ms)
        scored_fills: list[ReplayFill] = []
        scored_signals: list[SignalEvent] = []
        equity_curve: list[EquityPoint] = []
        seen_fill_ids: set[str] = set()
        score_boundary_recorded = False

        for event in self._ordered_events(window):
            if not score_boundary_recorded and event.replay_at_ms >= window.scored_start_ms:
                boundary_context = self._context_at_boundary(clock, window)
                self._append_equity(equity_curve, window.scored_start_ms, boundary_context)
                score_boundary_recorded = True

            snapshot = self._safe_snapshot(clock.advance_to(event.replay_at_ms))
            observation_is_usable = event.observation.is_usable(event.replay_at_ms)
            context_event = event if observation_is_usable else None
            context = ReplayCycleContext(context_event, snapshot, window, self._candle_policy)
            fills = (
                tuple(self._exchange_adapter.process_event(event)) if observation_is_usable else ()
            )
            self._validate_fills(fills, event, seen_fill_ids)
            scored_fills.extend(fill for fill in fills if window.is_scored(fill.at_ms))

            if self._cycle_trigger(context):
                result = self._cycle_callback(
                    context,
                    self._exchange_adapter,
                    self._strategy_adapter,
                )
                if result is not None:
                    self._validate_signals(result.signals, context)
                    scored_signals.extend(
                        signal for signal in result.signals if window.is_scored(signal.at_ms)
                    )
            if context.is_scored:
                self._append_equity(equity_curve, event.replay_at_ms, context)

        final_context = self._final_context(clock, window)
        if not score_boundary_recorded:
            self._append_equity(equity_curve, window.scored_start_ms, final_context)
        self._append_equity(equity_curve, window.scored_end_ms, final_context)
        daily_counts = _daily_trade_counts(window, scored_fills)
        metrics = _build_metrics(scored_fills, equity_curve, daily_counts)
        signals = tuple(sorted(scored_signals, key=lambda item: (item.at_ms, item.symbol)))
        return ReplayReport(
            window=window,
            fills=tuple(scored_fills),
            signals=signals,
            signal_episodes=deduplicate_signal_episodes(
                signals,
                max_gap_ms=self._signal_episode_gap_ms,
            ),
            equity_curve=tuple(equity_curve),
            daily_trade_counts=daily_counts,
            metrics=metrics,
        )

    def _context_at_boundary(
        self,
        clock: MarketClock,
        window: ReplayWindow,
    ) -> ReplayCycleContext:
        """Value pre-event state at the exact scored boundary."""
        snapshot = self._safe_snapshot(clock.advance_to(window.scored_start_ms))
        return ReplayCycleContext(None, snapshot, window, self._candle_policy)

    def _final_context(self, clock: MarketClock, window: ReplayWindow) -> ReplayCycleContext:
        """Create a terminal valuation context without publishing a fake event."""
        # The scored interval is half-open. A release exactly at end belongs to
        # the next window even though this terminal point is labelled at end.
        snapshot = self._safe_snapshot(clock.advance_to(window.scored_end_ms - 1))
        return ReplayCycleContext(None, snapshot, window, self._candle_policy)

    def _safe_snapshot(self, snapshot: AsOfSnapshot) -> AsOfSnapshot:
        """Apply closed-candle filtering before any adapter or callback sees data."""
        if self._candle_policy is None:
            return snapshot
        return self._candle_policy.filter_snapshot(snapshot)

    def _append_equity(
        self,
        curve: list[EquityPoint],
        at_ms: Milliseconds,
        context: ReplayCycleContext,
    ) -> None:
        """Append an exact valuation, retaining pre/post-event boundary points."""
        equity = _decimal(self._exchange_adapter.mark_to_market(context), "equity")
        if equity < 0:
            message = "mark_to_market returned negative equity"
            raise ValueError(message)
        curve.append(EquityPoint(at_ms=at_ms, equity=equity))

    @staticmethod
    def _validate_fills(
        fills: Sequence[ReplayFill],
        event: ReplayEvent,
        seen_fill_ids: set[str],
    ) -> None:
        """Reject duplicate, future, or pre-event fills at the adapter boundary."""
        fill_ids = {fill.fill_id for fill in fills}
        if len(fill_ids) != len(fills):
            message = f"exchange adapter returned duplicate fills for event {event.sequence}"
            raise ValueError(message)
        repeated_fill_ids = fill_ids & seen_fill_ids
        if repeated_fill_ids:
            repeated = min(repeated_fill_ids)
            message = f"exchange adapter repeated fill {repeated!r} across events"
            raise ValueError(message)
        seen_fill_ids.update(fill_ids)
        for fill in fills:
            if fill.at_ms < event.replay_at_ms:
                message = f"fill {fill.fill_id} predates replay event {event.sequence}"
                raise ValueError(message)
            if fill.at_ms > event.logical_at_ms:
                message = f"fill {fill.fill_id} is in the future of replay event {event.sequence}"
                raise ValueError(message)

    @staticmethod
    def _validate_signals(
        signals: Sequence[SignalEvent],
        context: ReplayCycleContext,
    ) -> None:
        """Reject signal metadata timestamped after its producing cycle."""
        for signal in signals:
            if signal.at_ms > context.now_ms:
                message = f"signal {signal.symbol}/{signal.action} is future-dated"
                raise ValueError(message)


__all__ = [
    "CycleResult",
    "DailyTradeCount",
    "EquityPoint",
    "PaperExchangeReplayAdapter",
    "PaperMarketEvent",
    "ReplayCycleContext",
    "ReplayEngine",
    "ReplayEvent",
    "ReplayExchangeAdapter",
    "ReplayFill",
    "ReplayMetrics",
    "ReplayReport",
    "ReplayWindow",
    "SignalEpisode",
    "SignalEvent",
    "StressMetrics",
    "StressScenario",
    "TimeWindow",
    "deduplicate_signal_episodes",
    "effective_sample_size",
    "embargo_after_window",
    "purge_and_embargo",
    "purge_overlapping_labels",
    "signal_autocorrelation",
]
