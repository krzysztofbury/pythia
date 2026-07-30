"""Tests for consistent SQLite backup handling."""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

import pytest

from db_backup import BackupInTransactionError, backup_database, check_integrity

if TYPE_CHECKING:
    from pathlib import Path


def test_backup_captures_committed_wal_content(tmp_path: Path) -> None:
    """Online backup includes committed rows still represented in WAL."""
    source_path = tmp_path / "source.db"
    destination_path = tmp_path / "backups" / "copy.db"
    source = sqlite3.connect(source_path)
    source.execute("PRAGMA journal_mode = WAL")
    source.execute("CREATE TABLE values_table(value TEXT NOT NULL)")
    source.execute("INSERT INTO values_table VALUES ('from-wal')")
    source.commit()

    assert backup_database(source, destination_path) == destination_path
    check_integrity(destination_path)
    backup = sqlite3.connect(destination_path)
    assert backup.execute("SELECT value FROM values_table").fetchone()[0] == "from-wal"
    backup.close()
    source.close()


def test_backup_never_overwrites_existing_destination(tmp_path: Path) -> None:
    """An existing backup path is preserved byte-for-byte."""
    source_path = tmp_path / "source.db"
    destination_path = tmp_path / "copy.db"
    source = sqlite3.connect(source_path)
    source.execute("CREATE TABLE sample(id INTEGER)")
    source.commit()
    source.close()
    destination_path.write_bytes(b"keep-me")

    with pytest.raises(FileExistsError):
        backup_database(source_path, destination_path)
    assert destination_path.read_bytes() == b"keep-me"


def test_backup_rejects_missing_source(tmp_path: Path) -> None:
    """A source path must identify an existing regular file."""
    with pytest.raises(FileNotFoundError):
        backup_database(tmp_path / "missing.db", tmp_path / "copy.db")


def test_backup_rejects_an_open_source_transaction(tmp_path: Path) -> None:
    """A connection cannot deadlock itself by backing up uncommitted work."""
    source = sqlite3.connect(tmp_path / "source.db")
    source.execute("CREATE TABLE sample(id INTEGER)")
    source.commit()
    source.execute("INSERT INTO sample VALUES (1)")

    with pytest.raises(BackupInTransactionError):
        backup_database(source, tmp_path / "copy.db")
    source.rollback()
    source.close()
