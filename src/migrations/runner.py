"""Small, transactional SQLite migration runner with mandatory snapshots."""

from __future__ import annotations

import dataclasses
import datetime as dt
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from db_backup import check_integrity, create_pre_migration_snapshot

if TYPE_CHECKING:
    from collections.abc import Sequence


MigrationCallable = Callable[[sqlite3.Connection], None]


@dataclasses.dataclass(frozen=True, order=True)
class Migration:
    """One numbered, forward-only schema migration."""

    version: int
    name: str
    apply: MigrationCallable = dataclasses.field(compare=False, repr=False)

    def __post_init__(self) -> None:
        """Validate registry identity fields early."""
        if self.version <= 0:
            message = "migration version must be positive"
            raise ValueError(message)
        if not self.name or self.name.strip() != self.name:
            message = "migration name must be non-empty and trimmed"
            raise ValueError(message)


class MigrationDriftError(RuntimeError):
    """Raised when recorded migrations do not match the supplied registry."""


class MigrationRunner:
    """Apply an ordered migration registry, snapshotting before any change."""

    def __init__(
        self,
        database_path: str | Path,
        migrations: Sequence[Migration],
        *,
        backup_directory: str | Path,
        busy_timeout_ms: int = 5_000,
    ) -> None:
        """Validate and retain a complete migration registry."""
        if busy_timeout_ms <= 0:
            message = "busy_timeout_ms must be positive"
            raise ValueError(message)
        versions = [migration.version for migration in migrations]
        if len(versions) != len(set(versions)):
            message = "migration versions must be unique"
            raise ValueError(message)
        self.database_path = Path(database_path)
        self.migrations = tuple(sorted(migrations))
        self.backup_directory = Path(backup_directory)
        self.busy_timeout_ms = busy_timeout_ms

    def run(self) -> list[int]:
        """Apply pending migrations and return their version numbers."""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path, isolation_level=None)
        try:
            connection.execute(f"PRAGMA busy_timeout = {self.busy_timeout_ms}")
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA journal_mode = WAL")
            applied = self._read_applied(connection)
            self._validate_registry(applied)
            pending = [item for item in self.migrations if item.version not in applied]
            if pending:
                create_pre_migration_snapshot(
                    connection,
                    self.backup_directory,
                    database_name=self.database_path.name,
                    from_version=max(applied, default=0),
                    to_version=pending[-1].version,
                )
            self._ensure_bookkeeping(connection)
            for migration in pending:
                self._apply_one(connection, migration)
            check_integrity(connection)
            return [migration.version for migration in pending]
        finally:
            connection.close()

    @staticmethod
    def _read_applied(connection: sqlite3.Connection) -> dict[int, str]:
        table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_migrations'",
        ).fetchone()
        if table is None:
            return {}
        rows = connection.execute(
            "SELECT version, name FROM schema_migrations ORDER BY version",
        ).fetchall()
        return {int(version): str(name) for version, name in rows}

    def _validate_registry(self, applied: dict[int, str]) -> None:
        registry = {migration.version: migration.name for migration in self.migrations}
        for version, recorded_name in applied.items():
            if registry.get(version) != recorded_name:
                message = f"recorded migration {version}:{recorded_name!r} is not in the registry"
                raise MigrationDriftError(message)
        applied_versions = sorted(applied)
        expected_prefix = [item.version for item in self.migrations[: len(applied_versions)]]
        if applied_versions != expected_prefix:
            message = "recorded migrations are not a contiguous registry prefix"
            raise MigrationDriftError(message)

    @staticmethod
    def _ensure_bookkeeping(connection: sqlite3.Connection) -> None:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "version INTEGER PRIMARY KEY, "
            "name TEXT NOT NULL UNIQUE, "
            "applied_at TEXT NOT NULL"
            ") STRICT",
        )

    @staticmethod
    def _apply_one(connection: sqlite3.Connection, migration: Migration) -> None:
        connection.execute("BEGIN IMMEDIATE")
        try:
            migration.apply(connection)
            connection.execute(
                "INSERT INTO schema_migrations(version, name, applied_at) VALUES (?, ?, ?)",
                (migration.version, migration.name, dt.datetime.now(tz=dt.UTC).isoformat()),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
