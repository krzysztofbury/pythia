"""Tests for migration snapshots, bookkeeping, and rollback."""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

import pytest

from migrations.runner import Migration, MigrationDriftError, MigrationRunner

if TYPE_CHECKING:
    from pathlib import Path


def test_runner_snapshots_before_migration_and_is_idempotent(tmp_path: Path) -> None:
    """The sole snapshot precedes changes and rerunning applies nothing."""
    path = tmp_path / "store.db"
    backup_directory = tmp_path / "snapshots"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE original(value TEXT)")
    connection.execute("INSERT INTO original VALUES ('preserved')")
    connection.commit()
    connection.close()

    migration = Migration(
        1,
        "add_execution_table",
        lambda database: database.execute("CREATE TABLE execution(id INTEGER PRIMARY KEY)"),
    )
    runner = MigrationRunner(path, [migration], backup_directory=backup_directory)
    assert runner.run() == [1]
    assert runner.run() == []

    snapshots = list(backup_directory.glob("*.sqlite3"))
    assert len(snapshots) == 1
    snapshot = sqlite3.connect(snapshots[0])
    assert snapshot.execute("SELECT value FROM original").fetchone()[0] == "preserved"
    assert (
        snapshot.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'execution'",
        ).fetchone()[0]
        == 0
    )
    assert (
        snapshot.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' "
            "AND name = 'schema_migrations'",
        ).fetchone()[0]
        == 0
    )
    snapshot.close()


def test_failed_migration_rolls_back_schema_and_marker(tmp_path: Path) -> None:
    """A failed callback leaves neither its schema nor its version marker."""
    path = tmp_path / "store.db"

    def fail(database: sqlite3.Connection) -> None:
        database.execute("CREATE TABLE should_rollback(id INTEGER)")
        message = "migration failed"
        raise RuntimeError(message)

    runner = MigrationRunner(
        path,
        [Migration(1, "failing", fail)],
        backup_directory=tmp_path / "snapshots",
    )
    with pytest.raises(RuntimeError, match="migration failed"):
        runner.run()

    connection = sqlite3.connect(path)
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'should_rollback'",
        ).fetchone()[0]
        == 0
    )
    assert connection.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0] == 0
    connection.close()


def test_runner_refuses_recorded_registry_drift(tmp_path: Path) -> None:
    """Recorded names cannot be silently redefined by a new registry."""
    path = tmp_path / "store.db"
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE schema_migrations("
        "version INTEGER PRIMARY KEY, name TEXT NOT NULL, applied_at TEXT NOT NULL) STRICT",
    )
    connection.execute("INSERT INTO schema_migrations VALUES (1, 'old_name', 'now')")
    connection.commit()
    connection.close()

    runner = MigrationRunner(
        path,
        [Migration(1, "new_name", lambda _database: None)],
        backup_directory=tmp_path / "snapshots",
    )
    with pytest.raises(MigrationDriftError):
        runner.run()
