"""Process-lifetime advisory lock with an observable lease record."""

from __future__ import annotations

import atexit
import dataclasses
import datetime as dt
import fcntl
import json
import os
import socket
import uuid
from pathlib import Path
from typing import Any, Self


@dataclasses.dataclass(frozen=True)
class ProcessLease:
    """Human-readable ownership metadata; the flock remains authoritative."""

    token: str
    pid: int
    hostname: str
    acquired_at: str
    heartbeat_at: str
    active: bool
    released_at: str | None = None


class SingletonAlreadyRunningError(RuntimeError):
    """Raised when another process owns the singleton lock."""

    def __init__(self, lock_path: Path, lease: ProcessLease | None) -> None:
        """Describe the lock path and last observable owner."""
        self.lock_path = lock_path
        self.lease = lease
        owner = f"pid={lease.pid} host={lease.hostname}" if lease else "unknown owner"
        super().__init__(f"singleton lock {lock_path} is already held ({owner})")


class ProcessSingleton:
    """Own an exclusive ``flock`` until explicit release or process exit."""

    def __init__(self, lock_path: str | Path) -> None:
        """Prepare a singleton for a lock path without acquiring it yet."""
        self.lock_path = Path(lock_path)
        self.lease_path = self.lock_path.with_name(f"{self.lock_path.name}.lease.json")
        self._descriptor: int | None = None
        self._lease: ProcessLease | None = None

    @property
    def lease(self) -> ProcessLease | None:
        """Return this instance's lease, if acquired."""
        return self._lease

    @property
    def acquired(self) -> bool:
        """Whether this instance currently owns the lock descriptor."""
        return self._descriptor is not None

    def acquire(self) -> ProcessLease:
        """Acquire the lock without waiting and publish ownership metadata."""
        if self._descriptor is not None:
            message = "singleton is already acquired by this instance"
            raise RuntimeError(message)

        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        descriptor = os.open(self.lock_path, flags, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            os.close(descriptor)
            raise SingletonAlreadyRunningError(
                self.lock_path,
                read_lease(self.lock_path),
            ) from error

        now = _utc_now()
        lease = ProcessLease(
            token=uuid.uuid4().hex,
            pid=os.getpid(),
            hostname=socket.gethostname(),
            acquired_at=now,
            heartbeat_at=now,
            active=True,
        )
        self._descriptor = descriptor
        self._lease = lease
        try:
            self._publish(lease)
            atexit.register(self.release)
        except Exception:
            self._lease = None
            self._descriptor = None
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
            raise
        return lease

    def refresh(self) -> ProcessLease:
        """Refresh the observable heartbeat without changing lock ownership."""
        lease = self._require_lease()
        refreshed = dataclasses.replace(lease, heartbeat_at=_utc_now())
        self._publish(refreshed)
        self._lease = refreshed
        return refreshed

    def release(self) -> None:
        """Mark the lease inactive, then release and close the flock descriptor."""
        descriptor = self._descriptor
        lease = self._lease
        if descriptor is None:
            return

        try:
            if lease is not None:
                now = _utc_now()
                released = dataclasses.replace(
                    lease,
                    heartbeat_at=now,
                    active=False,
                    released_at=now,
                )
                self._lease = released
                self._publish(released)
        finally:
            self._descriptor = None
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
            atexit.unregister(self.release)

    def __enter__(self) -> Self:
        """Acquire and return this singleton."""
        self.acquire()
        return self

    def __exit__(self, *_exc: object) -> None:
        """Release lock ownership on context exit."""
        self.release()

    def _require_lease(self) -> ProcessLease:
        if self._descriptor is None or self._lease is None:
            message = "singleton has not been acquired"
            raise RuntimeError(message)
        return self._lease

    def _publish(self, lease: ProcessLease) -> None:
        data: dict[str, Any] = dataclasses.asdict(lease)
        temporary_path = self.lease_path.with_name(
            f".{self.lease_path.name}.{lease.token}.tmp",
        )
        try:
            with temporary_path.open("w", encoding="utf-8") as lease_file:
                json.dump(data, lease_file, sort_keys=True, separators=(",", ":"))
                lease_file.flush()
                os.fsync(lease_file.fileno())
            temporary_path.replace(self.lease_path)
        finally:
            temporary_path.unlink(missing_ok=True)


def read_lease(lock_path: str | Path) -> ProcessLease | None:
    """Read the latest complete lease record, if one has been published."""
    path = Path(lock_path)
    lease_path = path.with_name(f"{path.name}.lease.json")
    try:
        raw = json.loads(lease_path.read_text(encoding="utf-8"))
        return ProcessLease(**raw)
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError):
        return None


def _utc_now() -> str:
    return dt.datetime.now(tz=dt.UTC).isoformat(timespec="microseconds")
