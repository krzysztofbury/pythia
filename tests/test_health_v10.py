"""Tests for atomic structured health and monotonic checks."""

from __future__ import annotations

import datetime as dt
import json
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from health import HealthReporter, HealthState, check_health, main

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class FakeClock:
    """Controllable wall and monotonic clocks for lifecycle tests."""

    monotonic_value: float = 100.0
    wall_value: dt.datetime = dt.datetime(2026, 7, 24, tzinfo=dt.UTC)

    def monotonic(self) -> float:
        """Return the controlled monotonic time."""
        return self.monotonic_value

    def wall(self) -> dt.datetime:
        """Return the controlled wall time."""
        return self.wall_value

    def advance(self, seconds: float) -> None:
        """Advance both clocks by the same duration."""
        self.monotonic_value += seconds
        self.wall_value += dt.timedelta(seconds=seconds)


def test_health_document_contains_versions_state_and_operational_timestamps(
    tmp_path: Path,
) -> None:
    """Every required lifecycle and operation field is represented in JSON."""
    clock = FakeClock()
    path = tmp_path / "health.json"
    reporter = HealthReporter(
        path,
        app_version="10.0.0",
        build_version="sha-123",
        clock=clock.monotonic,
        wall_clock=clock.wall,
        boot_id="test-boot",
    )
    reporter.set_state(HealthState.RECOVERING)
    reporter.mark_migration()
    reporter.mark_backup()
    reporter.mark_reconciliation()
    reporter.mark_cycle_started(60.0)
    reporter.set_state(HealthState.READY)

    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["schema_version"] == 1
    assert document["app_version"] == "10.0.0"
    assert document["build_version"] == "sha-123"
    assert document["state"] == "READY"
    assert document["last_cycle_started_at"] is not None
    assert document["last_reconciliation_at"] is not None
    assert document["last_migration_at"] is not None
    assert document["last_backup_at"] is not None
    assert document["current_cycle_deadline"] is not None
    assert check_health(path, now_monotonic=clock.monotonic(), boot_id="test-boot")


def test_stuck_detection_uses_monotonic_deadline_not_wall_time(tmp_path: Path) -> None:
    """Wall-clock changes cannot hide or manufacture a stuck cycle."""
    clock = FakeClock()
    path = tmp_path / "health.json"
    reporter = HealthReporter(
        path,
        app_version="10.0.0",
        build_version="build",
        clock=clock.monotonic,
        wall_clock=clock.wall,
        boot_id="test-boot",
    )
    reporter.set_state(HealthState.READY)
    reporter.mark_cycle_started(10.0)
    clock.wall_value -= dt.timedelta(days=30)
    clock.monotonic_value += 11.0

    assert reporter.is_cycle_stuck()
    assert not check_health(path, now_monotonic=clock.monotonic(), boot_id="test-boot")


def test_new_reporter_replaces_persisted_ready_file_with_starting(tmp_path: Path) -> None:
    """A previous instance's READY file cannot remain ready during a new startup."""
    path = tmp_path / "health.json"
    path.write_text('{"state":"READY"}', encoding="utf-8")

    HealthReporter(path, app_version="10.0.0", build_version="new-build")

    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["state"] == "STARTING"
    assert not check_health(path)


def test_stale_or_previous_boot_documents_are_unhealthy(tmp_path: Path) -> None:
    """Monotonic age and boot identity reject persisted health false positives."""
    clock = FakeClock()
    path = tmp_path / "health.json"
    reporter = HealthReporter(
        path,
        app_version="10.0.0",
        build_version="build",
        clock=clock.monotonic,
        wall_clock=clock.wall,
        boot_id="old-boot",
    )
    reporter.set_state(HealthState.READY)

    assert not check_health(path, now_monotonic=clock.monotonic(), boot_id="new-boot")
    assert not check_health(
        path,
        stale_after_seconds=5.0,
        now_monotonic=clock.monotonic() + 6.0,
        boot_id="old-boot",
    )


def test_only_identifier_error_codes_can_be_published(tmp_path: Path) -> None:
    """Raw exceptions and secret-bearing prose cannot enter structured health."""
    reporter = HealthReporter(
        tmp_path / "health.json",
        app_version="10.0.0",
        build_version="build",
    )

    reporter.set_state(HealthState.DEGRADED, error_codes=["EXCHANGE_UNAVAILABLE"])
    with pytest.raises(ValueError, match="identifiers"):
        reporter.set_state(
            HealthState.DEGRADED,
            error_codes=["authorization failed: secret=value"],
        )


def test_atomic_writes_never_expose_partial_json(tmp_path: Path) -> None:
    """Concurrent readers see either complete old JSON or complete new JSON."""
    path = tmp_path / "health.json"
    reporter = HealthReporter(path, app_version="10.0.0", build_version="build")
    failures: list[Exception] = []

    def read_repeatedly() -> None:
        for _ in range(100):
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except Exception as error:  # noqa: BLE001, PERF203
                failures.append(error)

    reader = threading.Thread(target=read_repeatedly)
    reader.start()
    for _ in range(100):
        reporter.heartbeat()
    reader.join()

    assert failures == []


def test_cli_main_returns_probe_exit_codes(tmp_path: Path) -> None:
    """The check entry point maps readiness to conventional process exit codes."""
    path = tmp_path / "health.json"
    reporter = HealthReporter(path, app_version="10.0.0", build_version="build")
    assert main([str(path)]) == 1
    reporter.set_state(HealthState.READY)
    assert main([str(path)]) == 0
