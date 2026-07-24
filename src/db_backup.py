"""Consistent SQLite backups and pre-migration snapshots."""

from __future__ import annotations

import datetime as dt
import os
import sqlite3
import uuid
from pathlib import Path


class DatabaseIntegrityError(RuntimeError):
    """Raised when SQLite's full integrity check does not return ``ok``."""


class BackupInTransactionError(RuntimeError):
    """Raised rather than risking a self-blocking backup of an open transaction."""


def check_integrity(database: str | Path | sqlite3.Connection) -> None:
    """Run ``PRAGMA integrity_check`` and raise when any problem is reported."""
    connection, should_close = _open_source(database)
    try:
        rows = connection.execute("PRAGMA integrity_check").fetchall()
    finally:
        if should_close:
            connection.close()

    messages = [str(row[0]) for row in rows]
    if messages != ["ok"]:
        detail = "; ".join(messages) if messages else "integrity_check returned no rows"
        raise DatabaseIntegrityError(detail)


def backup_database(
    source: str | Path | sqlite3.Connection,
    destination: str | Path,
) -> Path:
    """Create a transactionally consistent, integrity-checked SQLite backup.

    SQLite's online backup API includes committed WAL content. The destination
    is built under a temporary name and atomically installed only after it
    passes a full integrity check.
    """
    destination_path = Path(destination)
    if destination_path.exists():
        raise FileExistsError(destination_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    source_connection, should_close = _open_source(source)
    temporary_path = destination_path.with_name(
        f".{destination_path.name}.{uuid.uuid4().hex}.tmp",
    )
    destination_connection: sqlite3.Connection | None = None
    try:
        source_path = _main_database_path(source_connection)
        if source_path is not None and source_path.resolve() == destination_path.resolve():
            message = "source and destination must be different files"
            raise ValueError(message)
        if source_connection.in_transaction:
            message = "commit or roll back the source transaction before backup"
            raise BackupInTransactionError(message)
        check_integrity(source_connection)
        destination_connection = sqlite3.connect(temporary_path)
        source_connection.backup(destination_connection)
        destination_connection.close()
        destination_connection = None
        check_integrity(temporary_path)
        temporary_path.replace(destination_path)
        _fsync_directory(destination_path.parent)
    finally:
        if destination_connection is not None:
            destination_connection.close()
        if should_close:
            source_connection.close()
        temporary_path.unlink(missing_ok=True)

    return destination_path


def create_pre_migration_snapshot(
    source: str | Path | sqlite3.Connection,
    backup_directory: str | Path,
    *,
    database_name: str,
    from_version: int,
    to_version: int,
) -> Path:
    """Create a uniquely named snapshot immediately before a migration batch."""
    if from_version < 0:
        message = "from_version must not be negative"
        raise ValueError(message)
    if to_version <= from_version:
        message = "to_version must be greater than from_version"
        raise ValueError(message)

    timestamp = dt.datetime.now(tz=dt.UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    safe_name = Path(database_name).name
    snapshot_name = f"{safe_name}.v{from_version}-to-v{to_version}.{timestamp}.sqlite3"
    return backup_database(source, Path(backup_directory) / snapshot_name)


def _open_source(
    database: str | Path | sqlite3.Connection,
) -> tuple[sqlite3.Connection, bool]:
    if isinstance(database, sqlite3.Connection):
        return database, False

    path = Path(database)
    if not path.is_file():
        raise FileNotFoundError(path)
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    return connection, True


def _main_database_path(connection: sqlite3.Connection) -> Path | None:
    rows = connection.execute("PRAGMA database_list").fetchall()
    main = next((row for row in rows if row[1] == "main"), None)
    if main is None or not main[2]:
        return None
    return Path(str(main[2]))


def _fsync_directory(directory: Path) -> None:
    """Make an atomic rename durable: the entry itself lives in the directory."""
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
