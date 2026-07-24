"""Tests for drain-safe runtime mutation state."""

from __future__ import annotations

import signal
import threading

import pytest

from runtime_state import MutationRejectedError, RuntimeState


def test_drain_closes_gate_before_waiting_for_in_flight_mutation() -> None:
    """Drain rejects new work while allowing admitted work to finish."""
    state = RuntimeState()
    entered = threading.Event()
    release = threading.Event()

    def mutate() -> None:
        with state.mutation_section():
            entered.set()
            release.wait(timeout=1.0)

    worker = threading.Thread(target=mutate)
    worker.start()
    assert entered.wait(timeout=1.0)

    state.request_drain()
    assert not state.entry_gate_open
    assert state.in_flight_mutations == 1
    with pytest.raises(MutationRejectedError), state.mutation_section():
        pass

    assert not state.wait_for_mutations(0.01)
    release.set()
    worker.join(timeout=1.0)
    assert state.wait_for_mutations(0.1)
    assert state.in_flight_mutations == 0


def test_sigterm_handler_withdraws_readiness_and_interrupts_waits() -> None:
    """The signal path performs only immediate, non-blocking drain actions."""
    state = RuntimeState()
    assert state.mark_ready()
    assert state.readiness

    state.handle_signal(signal.SIGTERM, None)

    assert state.draining
    assert not state.readiness
    assert state.shutdown_event.is_set()
    assert not state.mark_ready()


def test_stop_is_bounded_and_marks_stopped_only_after_mutations_finish() -> None:
    """A timed-out stop stays draining rather than claiming shutdown completed."""
    state = RuntimeState()
    entered = threading.Event()
    release = threading.Event()

    def mutate() -> None:
        with state.mutation_section():
            entered.set()
            release.wait(timeout=1.0)

    worker = threading.Thread(target=mutate)
    worker.start()
    assert entered.wait(timeout=1.0)

    assert not state.stop(0.0)
    assert state.draining
    assert not state.stopped
    release.set()
    worker.join(timeout=1.0)
    assert state.stop(0.1)
    assert state.stopped
    assert not state.draining


@pytest.mark.parametrize("timeout", [-1.0, float("inf"), float("nan")])
def test_invalid_drain_timeout_is_rejected(timeout: float) -> None:
    """Unbounded or nonsensical drain waits are forbidden."""
    state = RuntimeState()
    with pytest.raises(ValueError, match="finite and non-negative"):
        state.drain(timeout)
    assert state.entry_gate_open
