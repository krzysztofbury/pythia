"""Bitemporal market observations and a deterministic replay clock.

``event_at_ms`` describes when a fact happened, ``available_at_ms`` when its
producer could first have published it, and ``ingested_at_ms`` when this system
actually received it.  Replay visibility requires all three times to be no
later than the as-of time.  This distinction is the core no-lookahead invariant.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from decimal import Decimal
from enum import StrEnum
from types import MappingProxyType
from typing import TypeAlias, cast

Milliseconds: TypeAlias = int
FrozenPayload: TypeAlias = (
    None
    | bool
    | int
    | float
    | str
    | Decimal
    | tuple["FrozenPayload", ...]
    | Mapping[str, "FrozenPayload"]
)


class ObservationStatus(StrEnum):
    """Data-quality state attached to an observation revision."""

    VALID = "valid"
    INVALID = "invalid"
    QUARANTINED = "quarantined"
    SUPERSEDED = "superseded"


def _require_timestamp(value: int, name: str) -> None:
    """Reject ambiguous booleans and negative Unix timestamps."""
    if isinstance(value, bool) or not isinstance(value, int):
        message = f"{name} must be an integer number of milliseconds"
        raise TypeError(message)
    if value < 0:
        message = f"{name} must be non-negative"
        raise ValueError(message)


def _freeze_payload(value: object) -> FrozenPayload:
    """Recursively detach mutable containers from an immutable observation."""
    if isinstance(value, float) and not math.isfinite(value):
        message = "floating-point payload values must be finite"
        raise ValueError(message)
    if isinstance(value, Decimal) and not value.is_finite():
        message = "Decimal payload values must be finite"
        raise ValueError(message)
    if value is None or isinstance(value, (bool, int, float, str, Decimal)):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return _freeze_payload(asdict(value))
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            message = "payload mapping keys must be strings"
            raise TypeError(message)
        frozen = {key: _freeze_payload(item) for key, item in value.items()}
        return MappingProxyType(frozen)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_payload(item) for item in value)
    message = f"payload value of type {type(value).__name__} is not replay-serializable"
    raise TypeError(message)


def _json_value(value: FrozenPayload) -> object:
    """Convert a frozen payload to a stable, type-preserving JSON value."""
    if isinstance(value, Decimal):
        return {"__decimal__": str(value)}
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


def hash_payload(payload: object) -> str:
    """Return a deterministic SHA-256 hash for a supported payload."""
    frozen = _freeze_payload(payload)
    encoded = json.dumps(
        _json_value(frozen),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class Observation:
    """One immutable revision of a market fact."""

    event_at_ms: Milliseconds
    available_at_ms: Milliseconds
    ingested_at_ms: Milliseconds
    expires_at_ms: Milliseconds
    source: str
    payload_hash: str
    status: ObservationStatus
    payload: FrozenPayload = field(default=None, compare=False, hash=False, repr=False)

    def __post_init__(self) -> None:
        """Validate time ordering and detach the payload from caller mutation."""
        for name in ("event_at_ms", "available_at_ms", "ingested_at_ms", "expires_at_ms"):
            _require_timestamp(cast("int", getattr(self, name)), name)
        if self.available_at_ms < self.event_at_ms:
            message = "available_at_ms must not precede event_at_ms"
            raise ValueError(message)
        if self.ingested_at_ms < self.available_at_ms:
            message = "ingested_at_ms must not precede available_at_ms"
            raise ValueError(message)
        normalized_source = self.source.strip()
        if not normalized_source:
            message = "source must be non-empty"
            raise ValueError(message)
        normalized_hash = self.payload_hash.strip().lower()
        if not normalized_hash:
            message = "payload_hash must be non-empty"
            raise ValueError(message)
        try:
            normalized_status = ObservationStatus(self.status)
        except ValueError as exc:
            message = f"unsupported observation status {self.status!r}"
            raise ValueError(message) from exc

        object.__setattr__(self, "source", normalized_source)
        object.__setattr__(self, "payload_hash", normalized_hash)
        object.__setattr__(self, "status", normalized_status)
        object.__setattr__(self, "payload", _freeze_payload(self.payload))

    @classmethod
    def from_payload(  # noqa: PLR0913
        cls,
        *,
        event_at_ms: Milliseconds,
        available_at_ms: Milliseconds,
        ingested_at_ms: Milliseconds,
        expires_at_ms: Milliseconds,
        source: str,
        payload: object,
        status: ObservationStatus | str = ObservationStatus.VALID,
    ) -> Observation:
        """Build an observation and derive its content-addressed payload hash."""
        return cls(
            event_at_ms=event_at_ms,
            available_at_ms=available_at_ms,
            ingested_at_ms=ingested_at_ms,
            expires_at_ms=expires_at_ms,
            source=source,
            payload_hash=hash_payload(payload),
            status=ObservationStatus(status),
            payload=_freeze_payload(payload),
        )

    @property
    def known_at_ms(self) -> Milliseconds:
        """Return the first instant at which replay may know this revision."""
        return max(self.event_at_ms, self.available_at_ms, self.ingested_at_ms)

    def is_known(self, as_of_ms: Milliseconds) -> bool:
        """Return whether every temporal dimension is visible as of the instant."""
        _require_timestamp(as_of_ms, "as_of_ms")
        return self.known_at_ms <= as_of_ms

    def is_stale(self, as_of_ms: Milliseconds) -> bool:
        """Return whether the half-open validity interval has ended."""
        _require_timestamp(as_of_ms, "as_of_ms")
        return as_of_ms >= self.expires_at_ms

    def is_usable(self, as_of_ms: Milliseconds) -> bool:
        """Return whether this revision is known, valid, and not expired."""
        return (
            self.is_known(as_of_ms)
            and self.status is ObservationStatus.VALID
            and not self.is_stale(as_of_ms)
        )


def observation_order_key(observation: Observation) -> tuple[int, int, int, str, str]:
    """Return the stable total ordering used by replay event queues."""
    return (
        observation.known_at_ms,
        observation.event_at_ms,
        observation.available_at_ms,
        observation.source,
        observation.payload_hash,
    )


def _revision_key(observation: Observation) -> tuple[int, int, str]:
    """Order revisions of one source/event without depending on input order."""
    return observation.ingested_at_ms, observation.available_at_ms, observation.payload_hash


def _latest_known_revision(
    observations: Iterable[Observation],
    as_of_ms: Milliseconds,
) -> Observation | None:
    """Select the latest revision that physically existed at the as-of instant."""
    known = [observation for observation in observations if observation.is_known(as_of_ms)]
    return max(known, key=_revision_key, default=None)


def select_as_of(
    observations: Iterable[Observation],
    as_of_ms: Milliseconds,
    *,
    source: str | None = None,
) -> Observation | None:
    """Select the latest source fact without falling back past stale/retracted data.

    Revision selection occurs before validity filtering.  Consequently, a newly
    ingested invalidation or an expired latest fact cannot accidentally uncover
    an older value and make it appear current.
    """
    _require_timestamp(as_of_ms, "as_of_ms")
    normalized_source = source.strip() if source is not None else None
    candidates = [
        observation
        for observation in observations
        if normalized_source is None or observation.source == normalized_source
    ]
    if not candidates:
        return None

    latest_event_at_ms = max(
        (observation.event_at_ms for observation in candidates if observation.is_known(as_of_ms)),
        default=None,
    )
    if latest_event_at_ms is None:
        return None
    revision = _latest_known_revision(
        (
            observation
            for observation in candidates
            if observation.event_at_ms == latest_event_at_ms
        ),
        as_of_ms,
    )
    return revision if revision is not None and revision.is_usable(as_of_ms) else None


@dataclass(frozen=True, slots=True)
class AsOfSnapshot:
    """Immutable data view that cannot be advanced by a cycle callback."""

    as_of_ms: Milliseconds
    observations: tuple[Observation, ...]
    _latest_by_source: Mapping[str, Observation]

    def latest(self, source: str) -> Observation | None:
        """Return the current usable fact for a source."""
        return self._latest_by_source.get(source.strip())

    def history(self, source: str) -> tuple[Observation, ...]:
        """Return known, usable source events in event-time order."""
        normalized_source = source.strip()
        return tuple(
            observation
            for observation in self.observations
            if observation.source == normalized_source
        )


class ObservationStore:
    """Read-only indexed collection supporting safe bitemporal snapshots."""

    def __init__(self, observations: Iterable[Observation]) -> None:
        """Materialize observations in a stable order and reject exact duplicates."""
        ordered = tuple(sorted(observations, key=observation_order_key))
        identities = {
            (
                observation.event_at_ms,
                observation.available_at_ms,
                observation.ingested_at_ms,
                observation.source,
                observation.payload_hash,
            )
            for observation in ordered
        }
        if len(identities) != len(ordered):
            message = "observations contain a duplicate revision"
            raise ValueError(message)
        self._observations = ordered
        grouped: dict[str, list[Observation]] = defaultdict(list)
        for observation in ordered:
            grouped[observation.source].append(observation)
        self._by_source = {source: tuple(items) for source, items in grouped.items()}

    @property
    def observations(self) -> tuple[Observation, ...]:
        """Return the deterministic complete event stream."""
        return self._observations

    def as_of(self, source: str, as_of_ms: Milliseconds) -> Observation | None:
        """Return the latest usable fact for one source."""
        return select_as_of(self._by_source.get(source.strip(), ()), as_of_ms)

    def snapshot(self, as_of_ms: Milliseconds) -> AsOfSnapshot:
        """Build a revision-correct immutable snapshot for a replay cycle."""
        _require_timestamp(as_of_ms, "as_of_ms")
        selected_history: list[Observation] = []
        latest_by_source: dict[str, Observation] = {}
        for source, source_observations in self._by_source.items():
            by_event: dict[int, list[Observation]] = defaultdict(list)
            for observation in source_observations:
                if observation.is_known(as_of_ms):
                    by_event[observation.event_at_ms].append(observation)
            for event_observations in by_event.values():
                revision = _latest_known_revision(event_observations, as_of_ms)
                if revision is not None and revision.is_usable(as_of_ms):
                    selected_history.append(revision)
            latest = select_as_of(source_observations, as_of_ms)
            if latest is not None:
                latest_by_source[source] = latest

        selected_history.sort(key=lambda item: (item.event_at_ms, item.source, item.payload_hash))
        return AsOfSnapshot(
            as_of_ms=as_of_ms,
            observations=tuple(selected_history),
            _latest_by_source=MappingProxyType(latest_by_source),
        )


@dataclass(frozen=True, slots=True)
class ClosedCandlePolicy:
    """Release open-time-stamped candles only after their complete interval."""

    interval_ms: Milliseconds
    close_lag_ms: Milliseconds = 0
    sources: frozenset[str] | None = None

    def __post_init__(self) -> None:
        """Validate interval and normalize an optional source allowlist."""
        _require_timestamp(self.interval_ms, "interval_ms")
        _require_timestamp(self.close_lag_ms, "close_lag_ms")
        if self.interval_ms == 0:
            message = "interval_ms must be positive"
            raise ValueError(message)
        if self.sources is not None:
            normalized = frozenset(source.strip() for source in self.sources if source.strip())
            if not normalized:
                message = "sources must contain at least one non-empty source"
                raise ValueError(message)
            object.__setattr__(self, "sources", normalized)

    def applies_to(self, observation: Observation) -> bool:
        """Return whether the observation is governed by this candle policy."""
        return self.sources is None or observation.source in self.sources

    def closed_at_ms(self, observation: Observation) -> Milliseconds:
        """Return the earliest instant at which the whole candle is observable."""
        return observation.event_at_ms + self.interval_ms + self.close_lag_ms

    def replay_at_ms(self, observation: Observation) -> Milliseconds:
        """Combine data arrival and candle close into one safe release time."""
        if not self.applies_to(observation):
            return observation.known_at_ms
        return max(observation.known_at_ms, self.closed_at_ms(observation))

    def is_closed(self, observation: Observation, as_of_ms: Milliseconds) -> bool:
        """Return whether a usable observation is a completed candle as of time."""
        if not self.applies_to(observation):
            return False
        return observation.is_usable(as_of_ms) and self.closed_at_ms(observation) <= as_of_ms

    def select(self, snapshot: AsOfSnapshot, source: str) -> tuple[Observation, ...]:
        """Return only closed candles from an already-safe snapshot."""
        return tuple(
            observation
            for observation in snapshot.history(source)
            if self.is_closed(observation, snapshot.as_of_ms)
        )

    def filter_snapshot(self, snapshot: AsOfSnapshot) -> AsOfSnapshot:
        """Remove open candles so callbacks cannot bypass the policy accidentally."""
        observations = tuple(
            observation
            for observation in snapshot.observations
            if not self.applies_to(observation)
            or self.closed_at_ms(observation) <= snapshot.as_of_ms
        )
        latest_by_source: dict[str, Observation] = {}
        for observation in observations:
            current = latest_by_source.get(observation.source)
            if current is None or (observation.event_at_ms, _revision_key(observation)) > (
                current.event_at_ms,
                _revision_key(current),
            ):
                latest_by_source[observation.source] = observation
        return AsOfSnapshot(
            as_of_ms=snapshot.as_of_ms,
            observations=observations,
            _latest_by_source=MappingProxyType(latest_by_source),
        )


class MarketClock:
    """Monotonic replay clock with immutable as-of views."""

    def __init__(self, store: ObservationStore, start_at_ms: Milliseconds = 0) -> None:
        """Start a clock over a fixed observation store."""
        _require_timestamp(start_at_ms, "start_at_ms")
        self._store = store
        self._now_ms = start_at_ms

    @property
    def now_ms(self) -> Milliseconds:
        """Return current replay time."""
        return self._now_ms

    def advance_to(self, next_at_ms: Milliseconds) -> AsOfSnapshot:
        """Advance monotonically and return the resulting safe snapshot."""
        _require_timestamp(next_at_ms, "next_at_ms")
        if next_at_ms < self._now_ms:
            message = f"market clock cannot move backwards from {self._now_ms} to {next_at_ms}"
            raise ValueError(message)
        self._now_ms = next_at_ms
        return self._store.snapshot(self._now_ms)

    def snapshot(self) -> AsOfSnapshot:
        """Return a safe view at current replay time."""
        return self._store.snapshot(self._now_ms)


__all__ = [
    "AsOfSnapshot",
    "ClosedCandlePolicy",
    "MarketClock",
    "Milliseconds",
    "Observation",
    "ObservationStatus",
    "ObservationStore",
    "hash_payload",
    "observation_order_key",
    "select_as_of",
]
