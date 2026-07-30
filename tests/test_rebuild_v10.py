"""Offline v9 archive and exact Kraken rebuild tests."""
# ruff: noqa: D102, D103, D107

from __future__ import annotations

import json
import sqlite3
import stat
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pytest

from execution_store import ExecutionStore
from rebuild_v10 import RebuildError, rebuild_v10_store

if TYPE_CHECKING:
    from pathlib import Path


class RebuildExchange:
    """Narrow exact-history fixture for an offline rebuild."""

    def __init__(self) -> None:
        self.fills = [
            {
                "id": "fill-1",
                "order": "entry-1",
                "timestamp": 2_000,
                "symbol": "BTC/USD",
                "side": "buy",
                "price": "100.004",
                "amount": "1",
                "cost": "100.01",
                "fee": {"cost": "0.01", "currency": "USD"},
                "info": {
                    "cost": "100.01",
                    "fee": "0.01",
                    "price": "100.004",
                    "vol": "1",
                },
            },
        ]
        self.open_orders: list[dict[str, Any]] = [
            {
                "id": "stop-1",
                "symbol": "BTC/USD",
                "side": "sell",
                "type": "stop-loss",
                "amount": "1",
                "remaining": "1",
                "stopLossPrice": "90",
                "status": "open",
            },
        ]
        self.balances = {"BTC": Decimal(1)}

    def fetch_account_fills(
        self,
        since_ms: int,
        end_ms: int,
        *,
        page_size: int,
        max_pages: int,
    ) -> list[dict[str, Any]]:
        _ = page_size, max_pages
        return [fill for fill in self.fills if since_ms <= fill["timestamp"] <= end_ms]

    def fetch_open_orders(self) -> list[dict[str, Any]]:
        return list(self.open_orders)

    def get_total_balance(self, currency: str) -> Decimal:
        return self.balances.get(currency, Decimal(0))


def _legacy_database(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE trades(id INTEGER PRIMARY KEY, note TEXT NOT NULL)")
    connection.execute("INSERT INTO trades(note) VALUES ('v9 evidence')")
    connection.commit()
    connection.close()


def test_rebuild_archives_v9_and_installs_only_audited_v10(tmp_path: Path) -> None:
    legacy = tmp_path / "trades.db"
    output = tmp_path / "live-execution-v10.db"
    archive_dir = tmp_path / "archive"
    _legacy_database(legacy)

    result = rebuild_v10_store(
        RebuildExchange(),  # type: ignore[arg-type]
        legacy_path=legacy,
        output_path=output,
        archive_dir=archive_dir,
        scope="live:kraken:test",
        symbols=("BTC/USD",),
        since_ms=0,
        end_ms=10_000,
        balance_tolerance=Decimal(0),
    )

    assert result.output_path == output
    assert result.fill_count == 1
    assert result.protection_count == 1
    assert result.archive_path.stat().st_mode & stat.S_IWUSR == 0
    manifest = json.loads(result.manifest_path.read_text())
    assert len(manifest["sha256"]) == 64  # noqa: PLR2004
    report = json.loads(result.report_path.read_text())
    assert report["status"] == "READY_FOR_RECOVERY_ONLY_LIVE"

    store = ExecutionStore(output, "live:kraken:test")
    position = store.get_position("BTC/USD")
    assert position is not None
    assert position.cost_basis == Decimal("100.02")
    protections = store.list_protection("BTC/USD", active_only=True)
    assert protections[0]["exchange_order_id"] == "stop-1"
    assert store.get_sync_cursor("fills:account") == "10000"


def test_rebuild_refuses_non_protective_open_order_and_does_not_install(tmp_path: Path) -> None:
    legacy = tmp_path / "trades.db"
    output = tmp_path / "live-execution-v10.db"
    archive_dir = tmp_path / "archive"
    _legacy_database(legacy)
    exchange = RebuildExchange()
    exchange.open_orders[0]["type"] = "limit"

    with pytest.raises(RebuildError, match="not an adoptable sell stop"):
        rebuild_v10_store(
            exchange,  # type: ignore[arg-type]
            legacy_path=legacy,
            output_path=output,
            archive_dir=archive_dir,
            scope="live:kraken:test",
            symbols=("BTC/USD",),
            since_ms=0,
            end_ms=10_000,
            balance_tolerance=Decimal(0),
        )

    assert output.exists() is False
    assert output.with_name("live-execution-v10.rebuild-report.json").exists()
