"""Production-parity contracts for the standalone v10 replay foundation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from decimal import Decimal
from typing import TYPE_CHECKING, cast

import pytest

from market_clock import (
    ClosedCandlePolicy,
    Observation,
    ObservationStatus,
    ObservationStore,
    select_as_of,
)
from paper_exchange import MarketRules, PaperExchange
from replay import (
    CycleResult,
    PaperExchangeReplayAdapter,
    ReplayCycleContext,
    ReplayEngine,
    ReplayEvent,
    ReplayFill,
    ReplayWindow,
    SignalEvent,
    StressScenario,
    TimeWindow,
    deduplicate_signal_episodes,
    effective_sample_size,
    purge_and_embargo,
    signal_autocorrelation,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

D = Decimal
MINUTE_MS = 60_000
DAY_MS = 86_400_000
SOURCE = "candles:BTC/USDT:1m"


def observation(  # noqa: PLR0913
    event_at_ms: int,
    price: str,
    *,
    available_at_ms: int | None = None,
    ingested_at_ms: int | None = None,
    expires_at_ms: int | None = None,
    status: ObservationStatus = ObservationStatus.VALID,
    source: str = SOURCE,
) -> Observation:
    """Build one open-time-stamped candle observation."""
    available = available_at_ms if available_at_ms is not None else event_at_ms + MINUTE_MS
    ingested = ingested_at_ms if ingested_at_ms is not None else available
    expires = expires_at_ms if expires_at_ms is not None else event_at_ms + 10 * DAY_MS
    return Observation.from_payload(
        event_at_ms=event_at_ms,
        available_at_ms=available,
        ingested_at_ms=ingested,
        expires_at_ms=expires,
        source=source,
        payload={"close": price},
        status=status,
    )


def close_price(item: Observation) -> Decimal:
    """Read the immutable conventional test payload."""
    payload = cast("Mapping[str, object]", item.payload)
    return D(str(payload["close"]))


class RecordingExchange:
    """Small deterministic adapter; strategy decisions remain in the callback."""

    def __init__(self) -> None:
        """Initialize balances, event trace, and deterministic fill schedule."""
        self.cash = D("1000")
        self.events: list[tuple[int, str]] = []
        self.pending_fills: dict[int, tuple[ReplayFill, ...]] = {}
        self.equity_values: Iterator[Decimal] | None = None

    def process_event(self, event: ReplayEvent) -> Sequence[ReplayFill]:
        """Record deterministic order and return configured adapter fills."""
        self.events.append((event.replay_at_ms, event.observation.payload_hash))
        return self.pending_fills.get(event.sequence, ())

    def mark_to_market(self, _context: ReplayCycleContext) -> Decimal:
        """Value cash using only adapter state."""
        if self.equity_values is not None:
            return next(self.equity_values)
        return self.cash


def test_observation_is_deeply_immutable_and_content_addressed() -> None:
    """Caller mutation cannot alter an archived observation or its identity."""
    payload = {"close": "100", "levels": ["99", "101"]}
    item = Observation.from_payload(
        event_at_ms=0,
        available_at_ms=1,
        ingested_at_ms=2,
        expires_at_ms=100,
        source="quote",
        payload=payload,
    )
    payload["close"] = "999"

    assert cast("Mapping[str, object]", item.payload)["close"] == "100"
    with pytest.raises(TypeError):
        cast("dict[str, object]", item.payload)["close"] = "101"
    with pytest.raises(FrozenInstanceError):
        item.source = "changed"  # type: ignore[misc]


def test_as_of_selection_rejects_future_stale_and_invalid_latest_revisions() -> None:
    """No source value may leak before ingestion or survive expiry/retraction."""
    first = observation(0, "100", available_at_ms=10, ingested_at_ms=20, expires_at_ms=50)
    future_revision = observation(
        0,
        "101",
        available_at_ms=10,
        ingested_at_ms=40,
        expires_at_ms=80,
    )
    invalidation = observation(
        0,
        "0",
        available_at_ms=10,
        ingested_at_ms=60,
        expires_at_ms=90,
        status=ObservationStatus.INVALID,
    )
    observations = [invalidation, future_revision, first]

    assert select_as_of(observations, 19, source=SOURCE) is None
    assert select_as_of(observations, 30, source=SOURCE) == first
    assert select_as_of(observations, 45, source=SOURCE) == future_revision
    assert select_as_of(observations, 55, source=SOURCE) == future_revision
    assert select_as_of(observations, 65, source=SOURCE) is None
    assert select_as_of(observations, 100, source=SOURCE) is None


def test_snapshot_never_contains_future_or_stale_history() -> None:
    """A cycle receives a fixed safe view rather than the mutable data store."""
    current = observation(0, "100", available_at_ms=10, ingested_at_ms=10, expires_at_ms=30)
    future = observation(20, "200", available_at_ms=40, ingested_at_ms=40)
    store = ObservationStore([future, current])

    snapshot = store.snapshot(20)

    assert snapshot.latest(SOURCE) == current
    assert snapshot.history(SOURCE) == (current,)
    assert store.snapshot(35).latest(SOURCE) is None


def test_closed_candle_policy_and_deterministic_tie_order() -> None:
    """Open candles wait for close and same-time releases have a stable order."""
    items = [
        observation(0, "101", source="b"),
        observation(0, "100", source="a"),
    ]
    exchange = RecordingExchange()
    seen: list[tuple[int, tuple[str, ...]]] = []
    policy = ClosedCandlePolicy(interval_ms=MINUTE_MS)

    def cycle(
        context: ReplayCycleContext,
        _exchange: RecordingExchange,
        _strategy: object,
    ) -> None:
        seen.append(
            (
                context.now_ms,
                tuple(item.source for item in context.snapshot.observations),
            ),
        )

    ReplayEngine(
        reversed(items),
        exchange_adapter=exchange,
        strategy_adapter=object(),
        cycle_callback=cycle,
        candle_policy=policy,
    ).run(ReplayWindow(0, MINUTE_MS, 2 * MINUTE_MS))

    assert [source_hash for _, source_hash in exchange.events] == [
        items[1].payload_hash,
        items[0].payload_hash,
    ]
    assert seen[0][0] == MINUTE_MS
    assert seen[0][1] == ("a", "b")


def test_expired_observation_never_reaches_exchange_or_cycle_payload() -> None:
    """Stale queue entries can trigger a missing-data cycle but cannot be consumed."""
    stale = observation(0, "100", expires_at_ms=MINUTE_MS)
    exchange = RecordingExchange()
    seen: list[tuple[ReplayEvent | None, tuple[Observation, ...]]] = []

    def cycle(
        context: ReplayCycleContext,
        _exchange: RecordingExchange,
        _strategy: object,
    ) -> None:
        seen.append((context.event, context.snapshot.observations))

    ReplayEngine(
        [stale],
        exchange_adapter=exchange,
        strategy_adapter=object(),
        cycle_callback=cycle,
        candle_policy=ClosedCandlePolicy(MINUTE_MS),
    ).run(ReplayWindow(0, MINUTE_MS, 2 * MINUTE_MS))

    assert exchange.events == []
    assert seen == [(None, ())]


def _run_price_replay(items: list[Observation]) -> list[tuple[int, tuple[Decimal, ...]]]:
    """Run a strategy-shaped callback and capture exactly what each cycle sees."""
    trace: list[tuple[int, tuple[Decimal, ...]]] = []

    def production_cycle(
        context: ReplayCycleContext,
        _exchange: RecordingExchange,
        _strategy: object,
    ) -> CycleResult:
        prices = tuple(close_price(item) for item in context.closed_candles(SOURCE))
        trace.append((context.now_ms, prices))
        return CycleResult(
            signals=(SignalEvent(context.now_ms, "BTC/USDT", "BUY", prices[-1]),),
        )

    ReplayEngine(
        items,
        exchange_adapter=RecordingExchange(),
        strategy_adapter=object(),
        cycle_callback=production_cycle,
        candle_policy=ClosedCandlePolicy(interval_ms=MINUTE_MS, sources=frozenset({SOURCE})),
    ).run(ReplayWindow(0, MINUTE_MS, 6 * MINUTE_MS))
    return trace


def test_future_perturbation_cannot_change_replay_prefix() -> None:
    """Changing unseen future prices leaves every earlier cycle byte-for-byte equal."""
    baseline = [observation(index * MINUTE_MS, str(100 + index)) for index in range(5)]
    perturbed = [
        item if index < 3 else observation(index * MINUTE_MS, str(9_000 + index))  # noqa: PLR2004
        for index, item in enumerate(baseline)
    ]

    baseline_trace = _run_price_replay(baseline)
    perturbed_trace = _run_price_replay(perturbed)

    cutoff_ms = 4 * MINUTE_MS
    assert [row for row in baseline_trace if row[0] < cutoff_ms] == [
        row for row in perturbed_trace if row[0] < cutoff_ms
    ]
    assert baseline_trace[-1] != perturbed_trace[-1]


def test_late_ingestion_cannot_rewrite_an_earlier_cycle() -> None:
    """A corrected old candle becomes visible only when its revision is ingested."""
    original = observation(0, "100", available_at_ms=MINUTE_MS, ingested_at_ms=MINUTE_MS)
    corrected = observation(
        0,
        "999",
        available_at_ms=MINUTE_MS,
        ingested_at_ms=3 * MINUTE_MS,
    )
    next_candle = observation(MINUTE_MS, "101")
    trace = _run_price_replay([corrected, next_candle, original])

    first_cycle = next(prices for at_ms, prices in trace if at_ms == MINUTE_MS)
    correction_cycle = next(prices for at_ms, prices in trace if at_ms == 3 * MINUTE_MS)
    assert first_cycle == (D("100"),)
    assert correction_cycle[0] == D("999")


def test_warmup_is_executed_but_excluded_from_scored_signals() -> None:
    """Warmup mutates adapters and strategy state without flattering metrics."""
    items = [observation(index * MINUTE_MS, str(100 + index)) for index in range(3)]
    strategy_state: list[int] = []

    def cycle(
        context: ReplayCycleContext,
        _exchange: RecordingExchange,
        state: list[int],
    ) -> CycleResult:
        state.append(context.now_ms)
        return CycleResult((SignalEvent(context.now_ms, "BTC/USDT", "BUY"),))

    report = ReplayEngine(
        items,
        exchange_adapter=RecordingExchange(),
        strategy_adapter=strategy_state,
        cycle_callback=cycle,
        candle_policy=ClosedCandlePolicy(MINUTE_MS),
    ).run(ReplayWindow(0, 2 * MINUTE_MS, 4 * MINUTE_MS))

    assert strategy_state == [MINUTE_MS, 2 * MINUTE_MS, 3 * MINUTE_MS]
    assert [signal.at_ms for signal in report.signals] == [2 * MINUTE_MS, 3 * MINUTE_MS]
    assert len(report.signal_episodes) == 1


def test_engine_delegates_matching_to_supplied_paper_exchange() -> None:
    """A callback-created order receives only a later-event fill and exact fee."""
    quotes = [
        Observation.from_payload(
            event_at_ms=at_ms,
            available_at_ms=at_ms,
            ingested_at_ms=at_ms,
            expires_at_ms=100,
            source="quotes",
            payload={
                "symbol": "BTC/USDT",
                "bid": bid,
                "ask": ask,
                "last": last,
            },
        )
        for at_ms, bid, ask, last in [
            (1, "99", "101", "100"),
            (2, "99", "100", "100"),
        ]
    ]
    paper = PaperExchange(
        initial_balances={"USDT": "1000", "BTC": "0"},
        markets={
            "BTC/USDT": MarketRules(
                "BTC",
                "USDT",
                price_increment="0.01",
                amount_increment="0.1",
                min_notional="10",
            ),
        },
        maker_fee_rate="0.001",
    )

    def equity(exchange: PaperExchange, context: ReplayCycleContext) -> Decimal:
        latest = context.snapshot.latest("quotes")
        assert latest is not None
        payload = cast("Mapping[str, object]", latest.payload)
        return exchange.get_total_balance("USDT") + exchange.get_total_balance("BTC") * D(
            str(payload["last"]),
        )

    adapter = PaperExchangeReplayAdapter(paper, equity_callback=equity)
    strategy_state = {"placed": False}

    def production_cycle(
        _context: ReplayCycleContext,
        exchange: PaperExchangeReplayAdapter[PaperExchange],
        state: dict[str, bool],
    ) -> None:
        if not state["placed"]:
            exchange.exchange.place_maker_buy("BTC/USDT", price="100", amount="1")
            state["placed"] = True

    report = ReplayEngine(
        quotes,
        exchange_adapter=adapter,
        strategy_adapter=strategy_state,
        cycle_callback=production_cycle,
    ).run(ReplayWindow(1, 1, 3))

    assert len(report.fills) == 1
    assert report.fills[0].at_ms == 2  # noqa: PLR2004
    assert report.metrics.fees == D("0.100")
    assert report.metrics.net_pnl == D("-0.100")
    assert report.metrics.equity_change == D("-0.100")


def test_purge_and_embargo_excludes_both_leakage_zones() -> None:
    """Training rows near either side of a scored window are unavailable."""
    items = [observation(index * MINUTE_MS, str(index)) for index in range(10)]

    clean = purge_and_embargo(
        items,
        scored_window=TimeWindow(4 * MINUTE_MS, 7 * MINUTE_MS),
        label_horizon_ms=2 * MINUTE_MS,
        embargo_ms=MINUTE_MS,
    )

    assert [item.event_at_ms // MINUTE_MS for item in clean] == [0, 1, 8, 9]


def test_signal_episode_dedup_and_effective_sample_size() -> None:
    """Repeated cycle signals count once and correlated samples lose weight."""
    signals = [
        SignalEvent(0, "BTC/USDT", "BUY", D("0.5")),
        SignalEvent(1, "BTC/USDT", "BUY", D("0.8")),
        SignalEvent(2, "BTC/USDT", "HOLD"),
        SignalEvent(3, "BTC/USDT", "BUY", D("0.6")),
        SignalEvent(4, "BTC/USDT", "SELL", D("0.9")),
    ]

    episodes = deduplicate_signal_episodes(signals)

    assert [(episode.action, episode.sample_count) for episode in episodes] == [
        ("BUY", 2),
        ("BUY", 1),
        ("SELL", 1),
    ]
    repeated = [D("1"), D("2"), D("3"), D("4"), D("5"), D("6")]
    assert signal_autocorrelation(repeated) > 0
    assert 1 <= effective_sample_size(repeated) < len(repeated)
    assert effective_sample_size([1, -1, 1, -1, 1, -1]) == 6  # noqa: PLR2004


def test_fee_net_metrics_zero_days_drawdown_and_stress() -> None:
    """Scored reports retain gross, costs, dense days, drawdown, and sensitivity."""
    start = 20_000 * DAY_MS
    items = [observation(start + index * DAY_MS, str(100 + index)) for index in range(3)]
    exchange = RecordingExchange()
    exchange.pending_fills[2] = (
        ReplayFill(
            fill_id="fill-1",
            order_id="order-1",
            at_ms=start + DAY_MS + MINUTE_MS,
            symbol="BTC/USDT",
            side="sell",
            quantity=D("2"),
            price=D("100"),
            fee=D("1"),
            gross_pnl=D("20"),
            slippage_cost=D("2"),
        ),
    )
    exchange.equity_values = iter([D("1000"), D("1000"), D("900"), D("1100"), D("1100")])
    report = ReplayEngine(
        items,
        exchange_adapter=exchange,
        strategy_adapter=object(),
        cycle_callback=lambda _context, _exchange, _strategy: None,
        candle_policy=ClosedCandlePolicy(MINUTE_MS),
    ).run(ReplayWindow(start, start + MINUTE_MS, start + 3 * DAY_MS))

    assert report.metrics.gross_pnl == D("20")
    assert report.metrics.fees == D("1")
    assert report.metrics.net_pnl == D("17")
    assert report.metrics.max_drawdown == D("100")
    assert report.metrics.max_drawdown_pct == D("10.0")
    assert report.metrics.total_days == 3  # noqa: PLR2004
    assert report.metrics.zero_trade_days == 2  # noqa: PLR2004
    stressed = report.stress(StressScenario(D("2"), D("50")))
    assert stressed.fees == D("2")
    assert stressed.slippage_cost == D("3")
    assert stressed.net_pnl == D("15")
