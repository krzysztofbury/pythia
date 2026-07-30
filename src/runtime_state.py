"""Thread-safe readiness, drain signaling, and mutation admission control."""

from __future__ import annotations

import contextlib
import math
import signal
import threading
import time
from collections.abc import Callable, Iterator
from types import FrameType


class MutationRejectedError(RuntimeError):
    """Raised when a mutation tries to enter after draining has begun."""


SignalHandler = Callable[[int, FrameType | None], object] | int | None


class RuntimeState:
    """Coordinate readiness and bounded shutdown around mutation sections."""

    def __init__(self) -> None:
        """Create a starting runtime with an open gate and no mutations."""
        self._condition = threading.Condition(threading.RLock())
        self._entry_gate_open = True
        self._ready = False
        self._stopped = False
        self._in_flight_mutations = 0
        self._shutdown_event = threading.Event()

    @property
    def shutdown_event(self) -> threading.Event:
        """Expose the event used to interrupt main-loop sleeps."""
        return self._shutdown_event

    @property
    def entry_gate_open(self) -> bool:
        """Whether a new mutation may enter."""
        with self._condition:
            return self._entry_gate_open

    @property
    def accepting_mutations(self) -> bool:
        """Whether a new mutation may enter."""
        return self.entry_gate_open

    @property
    def in_flight_mutations(self) -> int:
        """Return the number of admitted mutation sections not yet exited."""
        with self._condition:
            return self._in_flight_mutations

    @property
    def draining(self) -> bool:
        """Whether drain has closed the admission gate."""
        with self._condition:
            return not self._entry_gate_open and not self._stopped

    @property
    def stopped(self) -> bool:
        """Whether the runtime has completed shutdown."""
        with self._condition:
            return self._stopped

    @property
    def ready(self) -> bool:
        """Whether the runtime is ready to receive work."""
        with self._condition:
            return self._ready and self._entry_gate_open and not self._stopped

    @property
    def readiness(self) -> bool:
        """Expose readiness under a probe-friendly name."""
        return self.ready

    def mark_ready(self) -> bool:
        """Become ready unless drain has already won the lifecycle race."""
        with self._condition:
            if not self._entry_gate_open or self._stopped:
                return False
            self._ready = True
            return True

    def mark_not_ready(self) -> None:
        """Withdraw readiness without closing mutation admission."""
        with self._condition:
            self._ready = False

    @contextlib.contextmanager
    def mutation_section(self) -> Iterator[None]:
        """Admit one mutation or reject it when the drain gate is closed."""
        with self._condition:
            if not self._entry_gate_open or self._stopped:
                message = "runtime is draining; new mutations are rejected"
                raise MutationRejectedError(message)
            self._in_flight_mutations += 1
        try:
            yield
        finally:
            with self._condition:
                self._in_flight_mutations -= 1
                if self._in_flight_mutations == 0:
                    self._condition.notify_all()

    def request_drain(self) -> None:
        """Atomically close admission, withdraw readiness, and wake the main loop."""
        with self._condition:
            self._entry_gate_open = False
            self._ready = False
            self._shutdown_event.set()
            self._condition.notify_all()

    def wait_for_mutations(self, timeout_seconds: float) -> bool:
        """Wait at most timeout_seconds for all admitted mutations to finish."""
        _validate_timeout(timeout_seconds)
        deadline = time.monotonic() + timeout_seconds
        with self._condition:
            while self._in_flight_mutations:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(timeout=remaining)
            return True

    def drain(self, timeout_seconds: float) -> bool:
        """Close admission first, then wait a bounded time for current mutations."""
        _validate_timeout(timeout_seconds)
        self.request_drain()
        return self.wait_for_mutations(timeout_seconds)

    def stop(self, timeout_seconds: float) -> bool:
        """Drain and mark stopped only after every admitted mutation exits."""
        drained = self.drain(timeout_seconds)
        if drained:
            with self._condition:
                self._stopped = True
                self._condition.notify_all()
        return drained

    def handle_signal(self, _signum: int, _frame: FrameType | None) -> None:
        """Close the gate immediately from a SIGTERM-compatible handler."""
        self.request_drain()

    def install_sigterm_handler(self) -> SignalHandler:
        """Install the non-blocking drain handler and return the previous handler."""
        previous = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, self.handle_signal)
        return previous


def _validate_timeout(timeout_seconds: float) -> None:
    if (
        isinstance(timeout_seconds, bool)
        or not math.isfinite(timeout_seconds)
        or timeout_seconds < 0
    ):
        message = "timeout_seconds must be finite and non-negative"
        raise ValueError(message)
