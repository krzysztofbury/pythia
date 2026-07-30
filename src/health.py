"""Atomic structured health publication and command-line health checking."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import math
import os
import re
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Sequence

HEALTH_SCHEMA_VERSION = 1
# 3x the 900 s entry cycle: a healthy process heartbeats every cycle, so a
# document older than three cycles means the process is wedged. Deployments
# with a different cadence override this via HEALTHCHECK_MAX_AGE_SECONDS.
DEFAULT_STALE_AFTER_SECONDS = 2_700.0
_ERROR_CODE_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_.:-]{0,63}")
_TIMESTAMP_FIELDS = (
    "last_cycle_started_at",
    "last_cycle_completed_at",
    "last_reconciliation_at",
    "last_migration_at",
    "last_backup_at",
    "current_cycle_deadline",
)


class HealthState(StrEnum):
    """Operator-visible application lifecycle state."""

    STARTING = "STARTING"
    RECOVERING = "RECOVERING"
    READY = "READY"
    DEGRADED = "DEGRADED"
    DRAINING = "DRAINING"
    STOPPED = "STOPPED"


@dataclass(frozen=True)
class HealthSnapshot:
    """Complete health document written as one atomic JSON value."""

    schema_version: int
    app_version: str
    build_version: str
    instance_id: str
    boot_id: str
    pid: int
    state: HealthState
    updated_at: str
    updated_monotonic: float
    last_cycle_started_at: str | None = None
    last_cycle_completed_at: str | None = None
    last_reconciliation_at: str | None = None
    last_migration_at: str | None = None
    last_backup_at: str | None = None
    current_cycle_deadline: str | None = None
    current_cycle_deadline_monotonic: float | None = None
    error_codes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        data: dict[str, object] = dataclasses.asdict(self)
        data["state"] = self.state.value
        data["error_codes"] = list(self.error_codes)
        return data


class HealthReporter:
    """Maintain and atomically publish one process's health snapshot."""

    def __init__(  # noqa: PLR0913
        self,
        path: str | Path,
        *,
        app_version: str,
        build_version: str,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], dt.datetime] | None = None,
        boot_id: str | None = None,
    ) -> None:
        """Publish a fresh STARTING document, replacing any persisted old instance."""
        if not app_version or not build_version:
            message = "app_version and build_version must be non-empty"
            raise ValueError(message)
        self.path = Path(path)
        self._clock = clock
        self._wall_clock = wall_clock or (lambda: dt.datetime.now(tz=dt.UTC))
        self._lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

        now_at, now_monotonic = self._now()
        self._snapshot = HealthSnapshot(
            schema_version=HEALTH_SCHEMA_VERSION,
            app_version=app_version,
            build_version=build_version,
            instance_id=uuid.uuid4().hex,
            boot_id=boot_id or _BOOT_ID,
            pid=os.getpid(),
            state=HealthState.STARTING,
            updated_at=now_at,
            updated_monotonic=now_monotonic,
        )
        self._write_locked(self._snapshot)

    @property
    def snapshot(self) -> HealthSnapshot:
        """Return the latest immutable in-memory snapshot."""
        with self._lock:
            return self._snapshot

    def set_state(
        self,
        state: HealthState | str,
        *,
        error_codes: Collection[str] = (),
    ) -> HealthSnapshot:
        """Publish a lifecycle state and validated non-secret error identifiers."""
        target_state = HealthState(state)
        codes = _validate_error_codes(error_codes)
        with self._lock:
            now_at, now_monotonic = self._now()
            snapshot = dataclasses.replace(
                self._snapshot,
                state=target_state,
                error_codes=codes,
                updated_at=now_at,
                updated_monotonic=now_monotonic,
            )
            return self._write_locked(snapshot)

    def heartbeat(self) -> HealthSnapshot:
        """Refresh process freshness without changing lifecycle data."""
        with self._lock:
            now_at, now_monotonic = self._now()
            snapshot = dataclasses.replace(
                self._snapshot,
                updated_at=now_at,
                updated_monotonic=now_monotonic,
            )
            return self._write_locked(snapshot)

    def mark_cycle_started(self, timeout_seconds: float) -> HealthSnapshot:
        """Record a cycle start and its wall and monotonic deadlines."""
        _validate_timeout(timeout_seconds)
        with self._lock:
            wall_now = self._wall_now()
            now_monotonic = self._clock()
            snapshot = dataclasses.replace(
                self._snapshot,
                updated_at=_format_timestamp(wall_now),
                updated_monotonic=now_monotonic,
                last_cycle_started_at=_format_timestamp(wall_now),
                current_cycle_deadline=_format_timestamp(
                    wall_now + dt.timedelta(seconds=timeout_seconds),
                ),
                current_cycle_deadline_monotonic=now_monotonic + timeout_seconds,
            )
            return self._write_locked(snapshot)

    def mark_cycle_completed(self) -> HealthSnapshot:
        """Record successful cycle completion and clear its active deadline."""
        with self._lock:
            now_at, now_monotonic = self._now()
            snapshot = dataclasses.replace(
                self._snapshot,
                updated_at=now_at,
                updated_monotonic=now_monotonic,
                last_cycle_completed_at=now_at,
                current_cycle_deadline=None,
                current_cycle_deadline_monotonic=None,
            )
            return self._write_locked(snapshot)

    def mark_reconciliation(self) -> HealthSnapshot:
        """Record successful exchange reconciliation."""
        return self._mark_timestamp("last_reconciliation_at")

    def mark_migration(self) -> HealthSnapshot:
        """Record successful schema migration."""
        return self._mark_timestamp("last_migration_at")

    def mark_backup(self) -> HealthSnapshot:
        """Record successful state backup."""
        return self._mark_timestamp("last_backup_at")

    def is_cycle_stuck(self, *, now_monotonic: float | None = None) -> bool:
        """Determine cycle overrun exclusively from a monotonic deadline."""
        with self._lock:
            deadline = self._snapshot.current_cycle_deadline_monotonic
        now = self._clock() if now_monotonic is None else now_monotonic
        return deadline is not None and now > deadline

    def _mark_timestamp(self, field: str) -> HealthSnapshot:
        if field not in _TIMESTAMP_FIELDS:
            message = f"unsupported health timestamp field: {field}"
            raise ValueError(message)
        with self._lock:
            now_at, now_monotonic = self._now()
            values = self._snapshot.as_dict()
            values[field] = now_at
            values["state"] = self._snapshot.state
            values["error_codes"] = self._snapshot.error_codes
            values["updated_at"] = now_at
            values["updated_monotonic"] = now_monotonic
            snapshot = HealthSnapshot(**values)  # type: ignore[arg-type]
            return self._write_locked(snapshot)

    def _now(self) -> tuple[str, float]:
        return _format_timestamp(self._wall_now()), self._clock()

    def _wall_now(self) -> dt.datetime:
        moment = self._wall_clock()
        if moment.tzinfo is None or moment.utcoffset() is None:
            message = "wall_clock must return a timezone-aware datetime"
            raise ValueError(message)
        return moment.astimezone(dt.UTC)

    def _write_locked(self, snapshot: HealthSnapshot) -> HealthSnapshot:
        temporary_path = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary_path.open("x", encoding="utf-8") as health_file:
                json.dump(
                    snapshot.as_dict(),
                    health_file,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                health_file.write("\n")
                health_file.flush()
                os.fsync(health_file.fileno())
            temporary_path.replace(self.path)
        finally:
            temporary_path.unlink(missing_ok=True)
        self._snapshot = snapshot
        return snapshot


def check_health(
    path: str | Path,
    *,
    stale_after_seconds: float = DEFAULT_STALE_AFTER_SECONDS,
    accepted_states: Collection[HealthState | str] = (HealthState.READY,),
    now_monotonic: float | None = None,
    boot_id: str | None = None,
) -> bool:
    """Return whether a complete health file is fresh, ready, and not cycle-stuck."""
    _validate_timeout(stale_after_seconds, allow_zero=True)
    expected_states = {HealthState(state).value for state in accepted_states}
    now = time.monotonic() if now_monotonic is None else now_monotonic
    expected_boot_id = _BOOT_ID if boot_id is None else boot_id
    raw = _read_document(path)
    if raw is None:
        return False

    required = {
        "schema_version",
        "app_version",
        "build_version",
        "instance_id",
        "boot_id",
        "pid",
        "state",
        "updated_at",
        "updated_monotonic",
        "current_cycle_deadline_monotonic",
        "error_codes",
        *_TIMESTAMP_FIELDS,
    }
    if not required.issubset(raw):
        return False
    if not _valid_identity_and_state(raw, expected_boot_id, expected_states):
        return False

    updated = _finite_number(raw["updated_monotonic"])
    deadline_value = raw["current_cycle_deadline_monotonic"]
    deadline = None if deadline_value is None else _finite_number(deadline_value)
    fresh = updated is not None and updated <= now and now - updated <= stale_after_seconds
    deadline_valid = deadline_value is None or (deadline is not None and now <= deadline)
    return fresh and deadline_valid


def _read_document(path: str | Path) -> dict[str, object] | None:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return None
    return raw if isinstance(raw, dict) else None


def _valid_identity_and_state(
    raw: dict[str, object],
    expected_boot_id: str,
    expected_states: set[str],
) -> bool:
    return (
        raw["schema_version"] == HEALTH_SCHEMA_VERSION
        and raw["boot_id"] == expected_boot_id
        and isinstance(raw["state"], str)
        and raw["state"] in expected_states
        and isinstance(raw["app_version"], str)
        and bool(raw["app_version"])
        and isinstance(raw["build_version"], str)
        and bool(raw["build_version"])
        and isinstance(raw["instance_id"], str)
        and bool(raw["instance_id"])
        and isinstance(raw["error_codes"], list)
        and len(raw["error_codes"]) <= 32  # noqa: PLR2004
        and all(
            isinstance(code, str) and _ERROR_CODE_PATTERN.fullmatch(code) is not None
            for code in raw["error_codes"]
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run a Docker/Kubernetes-compatible health check and return an exit code."""
    parser = argparse.ArgumentParser(description="Check Oracle Pythia structured health")
    parser.add_argument("path", type=Path)
    parser.add_argument(
        "--max-age-seconds",
        type=float,
        default=DEFAULT_STALE_AFTER_SECONDS,
    )
    arguments = parser.parse_args(argv)
    return 0 if check_health(arguments.path, stale_after_seconds=arguments.max_age_seconds) else 1


def _validate_error_codes(error_codes: Collection[str]) -> tuple[str, ...]:
    codes = tuple(dict.fromkeys(error_codes))
    if len(codes) > 32:  # noqa: PLR2004
        message = "at most 32 health error codes may be published"
        raise ValueError(message)
    for code in codes:
        if not isinstance(code, str) or _ERROR_CODE_PATTERN.fullmatch(code) is None:
            message = "health error codes must be short identifiers, not exception text"
            raise ValueError(message)
    return codes


def _validate_timeout(value: float, *, allow_zero: bool = False) -> None:
    minimum_valid = value >= 0 if allow_zero else value > 0
    if isinstance(value, bool) or not math.isfinite(value) or not minimum_valid:
        message = (
            "timeout must be a finite non-negative number"
            if allow_zero
            else "timeout must be positive and finite"
        )
        raise ValueError(message)


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _format_timestamp(moment: dt.datetime) -> str:
    return moment.astimezone(dt.UTC).isoformat(timespec="microseconds")


def _read_boot_id() -> str:
    try:
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    except OSError:
        boot_id = f"boot-epoch-{round((time.time() - time.monotonic()) / 10.0):.0f}"
    return boot_id


_BOOT_ID = _read_boot_id()


if __name__ == "__main__":
    sys.exit(main())
