"""Tests for process-lifetime singleton ownership."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from singleton import ProcessSingleton, SingletonAlreadyRunningError, read_lease

if TYPE_CHECKING:
    from pathlib import Path


def test_lock_is_exclusive_and_lease_is_observable(tmp_path: Path) -> None:
    """The flock is exclusive and ownership metadata remains observable."""
    path = tmp_path / "oracle.lock"
    first = ProcessSingleton(path)
    lease = first.acquire()

    observed = read_lease(path)
    assert observed == lease
    assert observed.active is True

    second = ProcessSingleton(path)
    with pytest.raises(SingletonAlreadyRunningError) as error:
        second.acquire()
    assert error.value.lease is not None
    assert error.value.lease.token == lease.token

    refreshed = first.refresh()
    assert refreshed.token == lease.token
    assert refreshed.heartbeat_at >= lease.heartbeat_at
    first.release()
    assert read_lease(path).active is False  # type: ignore[union-attr]

    second_lease = second.acquire()
    assert second_lease.token != lease.token
    second.release()


def test_context_manager_releases_lock(tmp_path: Path) -> None:
    """Context exit marks the lease inactive and gives up the descriptor."""
    path = tmp_path / "oracle.lock"
    with ProcessSingleton(path) as singleton:
        assert singleton.acquired is True
    assert singleton.acquired is False
    assert singleton.lease is not None
    assert singleton.lease.active is False
