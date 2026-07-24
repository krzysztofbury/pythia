"""Offline, fail-closed v9 archive and Kraken-to-v10 execution rebuild."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import os
import stat
import sys
from collections.abc import Iterable, Mapping
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from config import Settings
from db_backup import backup_database
from exchange import KRAKEN_FILL_PAGE_SIZE, MAX_PAGINATION_PAGES, KrakenExchange
from execution_store import ExecutionStore, OrderState, Side, deterministic_client_id

_DAY_MS = 86_400_000
_OVERLAP_MS = 1_000
_MAX_REBUILD_WINDOWS = 3_660


class RebuildError(RuntimeError):
    """Raised when exact evidence is incomplete or inconsistent."""


@dataclasses.dataclass(frozen=True)
class RebuildResult:
    """Installed rebuild paths and audited counts."""

    output_path: Path
    archive_path: Path
    manifest_path: Path
    report_path: Path
    fill_count: int
    protection_count: int


def rebuild_v10_store(  # noqa: PLR0913
    exchange: KrakenExchange,
    *,
    legacy_path: str | Path,
    output_path: str | Path,
    archive_dir: str | Path,
    scope: str,
    symbols: Iterable[str],
    since_ms: int,
    end_ms: int,
    balance_tolerance: Decimal,
) -> RebuildResult:
    """Archive v9 and atomically install a v10 store only after exact audits pass."""
    legacy = Path(legacy_path)
    output = Path(output_path)
    archive_directory = Path(archive_dir)
    exact_symbols = _validate_inputs(
        legacy,
        output,
        archive_directory,
        scope,
        symbols,
        since_ms,
        end_ms,
        balance_tolerance,
    )
    archive_path, manifest_path = _archive_legacy(legacy, archive_directory)
    temporary = output.with_name(f".{output.name}.rebuild.tmp")
    report_path = output.with_name(f"{output.stem}.rebuild-report.json")
    if temporary.exists():
        message = f"stale rebuild temporary exists: {temporary}"
        raise RebuildError(message)

    report: dict[str, Any] = {
        "archive_path": str(archive_path),
        "end_ms": end_ms,
        "fills": 0,
        "output_path": str(output),
        "protections": 0,
        "scope": scope,
        "since_ms": since_ms,
        "status": "FAILED",
        "symbols": list(exact_symbols),
    }
    store: ExecutionStore | None = None
    try:
        store = ExecutionStore(temporary, scope)
        fills = _fetch_history(exchange, since_ms, end_ms)
        for fill in fills:
            _apply_rebuild_fill(store, fill)
        report["fills"] = len(fills)
        protections = _adopt_open_protections(exchange, store, exact_symbols)
        report["protections"] = protections
        report["balances"] = _audit_balances(
            exchange,
            store,
            exact_symbols,
            balance_tolerance,
        )
        store.set_sync_cursor("fills:account", str(end_ms // _OVERLAP_MS * _OVERLAP_MS))
        _validate_rebuilt_store(store)
        report["status"] = "READY_FOR_RECOVERY_ONLY_LIVE"
        store.close()
        store = None
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(output)
        _fsync_directory(output.parent)
        _write_json_atomic(report_path, report)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        _write_json_atomic(report_path, report)
        raise
    finally:
        if store is not None:
            store.close()
        temporary.unlink(missing_ok=True)
        temporary.with_name(f"{temporary.name}-wal").unlink(missing_ok=True)
        temporary.with_name(f"{temporary.name}-shm").unlink(missing_ok=True)
    return RebuildResult(
        output,
        archive_path,
        manifest_path,
        report_path,
        int(report["fills"]),
        int(report["protections"]),
    )


def _validate_rebuilt_store(store: ExecutionStore) -> None:
    integrity = str(store.connection.execute("PRAGMA integrity_check").fetchone()[0])
    if integrity != "ok":
        message = f"rebuilt database failed integrity_check: {integrity}"
        raise RebuildError(message)
    if store.list_reconciliation_issues():
        message = "rebuilt database contains unresolved reconciliation issues"
        raise RebuildError(message)


def _validate_inputs(  # noqa: PLR0913
    legacy: Path,
    output: Path,
    archive_directory: Path,
    scope: str,
    symbols: Iterable[str],
    since_ms: int,
    end_ms: int,
    balance_tolerance: Decimal,
) -> tuple[str, ...]:
    if not legacy.is_file():
        message = f"legacy database does not exist: {legacy}"
        raise RebuildError(message)
    if output.exists():
        message = f"refusing to overwrite v10 output: {output}"
        raise RebuildError(message)
    if output.resolve() == legacy.resolve():
        message = "legacy and v10 output paths must differ"
        raise RebuildError(message)
    if not scope.strip():
        message = "scope must not be empty"
        raise RebuildError(message)
    exact_symbols = tuple(dict.fromkeys(symbol.strip() for symbol in symbols if symbol.strip()))
    if not exact_symbols or any("/" not in symbol for symbol in exact_symbols):
        message = "at least one exact BASE/QUOTE symbol is required"
        raise RebuildError(message)
    if isinstance(since_ms, bool) or isinstance(end_ms, bool) or since_ms < 0 or end_ms <= since_ms:
        message = "rebuild time bounds must be increasing non-negative milliseconds"
        raise RebuildError(message)
    windows = (end_ms - since_ms + _DAY_MS - 1) // _DAY_MS
    if windows > _MAX_REBUILD_WINDOWS:
        message = f"rebuild exceeds {_MAX_REBUILD_WINDOWS} bounded daily windows"
        raise RebuildError(message)
    if not balance_tolerance.is_finite() or balance_tolerance < 0:
        message = "balance_tolerance must be a finite non-negative Decimal"
        raise RebuildError(message)
    if not archive_directory.parent.exists():
        message = f"archive parent does not exist: {archive_directory.parent}"
        raise RebuildError(message)
    return exact_symbols


def _archive_legacy(legacy: Path, archive_directory: Path) -> tuple[Path, Path]:
    archive_directory.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now(tz=dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    archive = archive_directory / f"v9-trades-{timestamp}.db"
    manifest = archive.with_suffix(".manifest.json")
    if archive.exists() or manifest.exists():
        message = f"archive collision at {archive}"
        raise RebuildError(message)
    backup_database(legacy, archive)
    digest = _sha256_file(archive)
    archive.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    _write_json_atomic(
        manifest,
        {
            "archive": archive.name,
            "created_at": dt.datetime.now(tz=dt.UTC).isoformat(),
            "sha256": digest,
            "source": str(legacy.resolve()),
        },
    )
    return archive, manifest


def _fetch_history(exchange: KrakenExchange, since_ms: int, end_ms: int) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    start = since_ms
    for _window in range(_MAX_REBUILD_WINDOWS):
        window_end = min(start + _DAY_MS, end_ms)
        page = exchange.fetch_account_fills(
            start,
            window_end,
            page_size=KRAKEN_FILL_PAGE_SIZE,
            max_pages=MAX_PAGINATION_PAGES,
        )
        for fill in page:
            fill_id = fill.get("id")
            if not isinstance(fill_id, str) or not fill_id:
                message = "Kraken returned a fill without an exact ID"
                raise RebuildError(message)
            previous = by_id.get(fill_id)
            if previous is not None and _canonical_json(previous) != _canonical_json(fill):
                message = f"Kraken fill {fill_id!r} changed across overlapping windows"
                raise RebuildError(message)
            by_id[fill_id] = dict(fill)
        if window_end == end_ms:
            break
        start = max(window_end - _OVERLAP_MS, 0)
    else:
        message = "Kraken rebuild exceeded its bounded window count"
        raise RebuildError(message)
    return sorted(by_id.values(), key=_fill_sort_key)


def _apply_rebuild_fill(store: ExecutionStore, fill: Mapping[str, Any]) -> None:
    fill_id = _required_text("fill id", fill.get("id"))
    exchange_order_id = _required_text("fill order id", fill.get("order"))
    symbol = _required_text("fill symbol", fill.get("symbol"))
    side = Side(str(fill.get("side") or "").upper())
    info = fill.get("info")
    raw = info if isinstance(info, Mapping) else {}
    price = _decimal("fill price", raw.get("price", fill.get("price")), positive=True)
    quantity = _decimal("fill amount", raw.get("vol", fill.get("amount")), positive=True)
    cost = _decimal(
        "fill cost",
        raw.get("cost", fill.get("cost", price * quantity)),
        positive=True,
    )
    fee_data = fill.get("fee")
    fee_mapping = fee_data if isinstance(fee_data, Mapping) else {}
    fee = _decimal("fill fee", raw.get("fee", fee_mapping.get("cost", 0)), positive=False)
    fee_currency = fee_mapping.get("currency")
    if fee_currency is not None:
        fee_currency = _required_text("fee currency", fee_currency)
    timestamp = fill.get("timestamp")
    if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp < 0:
        message = f"fill {fill_id!r} has an invalid timestamp"
        raise RebuildError(message)
    store.apply_fill(
        fill_id,
        symbol,
        side,
        price,
        quantity,
        cost=cost,
        fee=fee,
        fee_currency=fee_currency,
        executed_at_ms=timestamp,
        payload={"exchange_order_id": exchange_order_id, "source": "KRAKEN_V10_REBUILD"},
    )


def _adopt_open_protections(
    exchange: KrakenExchange,
    store: ExecutionStore,
    symbols: tuple[str, ...],
) -> int:
    allowed_symbols = set(symbols)
    orders = exchange.fetch_open_orders()
    coverage: dict[str, Decimal] = {}
    adopted = 0
    for order in orders:
        order_id = _required_text("open order id", order.get("id"))
        symbol = _required_text("open order symbol", order.get("symbol"))
        if symbol not in allowed_symbols:
            message = f"open order {order_id!r} is outside the rebuild universe: {symbol}"
            raise RebuildError(message)
        if str(order.get("side") or "").lower() != "sell" or not _is_stop_order(order):
            message = f"open order {order_id!r} is not an adoptable sell stop"
            raise RebuildError(message)
        position = store.get_position(symbol)
        if position is None:
            message = f"open stop {order_id!r} has no reconstructed position"
            raise RebuildError(message)
        quantity = _remaining_quantity(order)
        trigger = _trigger_price(order)
        protection_id = deterministic_client_id(store.scope, f"rebuild-stop:{order_id}")
        store.record_protection(
            protection_id,
            symbol,
            "STOP_LOSS",
            quantity,
            trigger,
            epoch_id=position.epoch_id,
            exchange_order_id=order_id,
            state=OrderState.OPEN,
            payload={"source": "KRAKEN_V10_REBUILD"},
        )
        coverage[symbol] = coverage.get(symbol, Decimal(0)) + quantity
        adopted += 1
    for position in store.list_open_positions():
        if coverage.get(position.symbol, Decimal(0)) != position.quantity:
            message = f"open stop coverage does not equal {position.symbol} inventory"
            raise RebuildError(message)
    return adopted


def _audit_balances(
    exchange: KrakenExchange,
    store: ExecutionStore,
    symbols: tuple[str, ...],
    tolerance: Decimal,
) -> list[dict[str, str]]:
    audited: list[dict[str, str]] = []
    for symbol in symbols:
        base = symbol.split("/", maxsplit=1)[0]
        position = store.get_position(symbol)
        expected = Decimal(0) if position is None else position.quantity
        actual = Decimal(str(exchange.get_total_balance(base)))
        difference = actual - expected
        if abs(difference) > tolerance:
            message = (
                f"balance mismatch for {symbol}: expected={expected} actual={actual} "
                f"tolerance={tolerance}"
            )
            raise RebuildError(message)
        audited.append(
            {
                "actual": str(actual),
                "difference": str(difference),
                "expected": str(expected),
                "symbol": symbol,
            },
        )
    return audited


def _remaining_quantity(order: Mapping[str, Any]) -> Decimal:
    if order.get("remaining") is not None:
        return _decimal("open order remaining", order["remaining"], positive=True)
    amount = _decimal("open order amount", order.get("amount"), positive=True)
    filled = _decimal("open order filled", order.get("filled", 0), positive=False)
    remaining = amount - filled
    if remaining <= 0:
        message = "open order has no remaining quantity"
        raise RebuildError(message)
    return remaining


def _trigger_price(order: Mapping[str, Any]) -> Decimal:
    for value in (
        order.get("stopLossPrice"),
        order.get("triggerPrice"),
        order.get("stopPrice"),
        order.get("price"),
    ):
        if value is not None:
            return _decimal("open stop trigger", value, positive=True)
    message = "open stop has no trigger price"
    raise RebuildError(message)


def _is_stop_order(order: Mapping[str, Any]) -> bool:
    order_type = str(order.get("type") or "").lower().replace("_", "-")
    return order_type in {"stop", "stop-loss", "stop-loss-limit", "stop-market"}


def _fill_sort_key(fill: Mapping[str, Any]) -> tuple[int, str]:
    timestamp = fill.get("timestamp")
    fill_id = fill.get("id")
    if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp < 0:
        message = "Kraken returned a fill with an invalid timestamp"
        raise RebuildError(message)
    return timestamp, _required_text("fill id", fill_id)


def _decimal(name: str, value: object, *, positive: bool) -> Decimal:
    if isinstance(value, bool):
        message = f"{name} must be numeric"
        raise RebuildError(message)
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        message = f"{name} must be a finite Decimal"
        raise RebuildError(message) from error
    if not parsed.is_finite() or (positive and parsed <= 0) or (not positive and parsed < 0):
        message = f"{name} is outside its accepted range"
        raise RebuildError(message)
    return parsed


def _required_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        message = f"{name} must be an exact non-empty string"
        raise RebuildError(message)
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as target:
            json.dump(value, target, indent=2, sort_keys=True)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        temporary.replace(path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _parse_utc_date(value: str) -> int:
    try:
        parsed = dt.datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=dt.UTC)
    except ValueError as error:
        message = "--since must use YYYY-MM-DD"
        raise argparse.ArgumentTypeError(message) from error
    return int(parsed.timestamp() * 1000)


def _scope(account_id: str) -> str:
    digest = hashlib.sha256(account_id.encode()).hexdigest()[:16]
    return f"live:kraken:{digest}"


def _arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-db", required=True, type=Path)
    parser.add_argument("--output-db", required=True, type=Path)
    parser.add_argument("--archive-dir", required=True, type=Path)
    parser.add_argument("--since", required=True, type=_parse_utc_date)
    parser.add_argument(
        "--confirm-services-stopped",
        required=True,
        choices=("I_HAVE_STOPPED_V9_AND_V10",),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the explicit offline cutover command."""
    args = _arguments(sys.argv[1:] if argv is None else argv)
    settings = Settings()
    if settings.trading_account_id is None:
        message = "TRADING_ACCOUNT_ID is required for a LIVE rebuild"
        raise RebuildError(message)
    end_ms = int(dt.datetime.now(tz=dt.UTC).timestamp() * 1000)
    exchange = KrakenExchange(settings.kraken_api_key, settings.kraken_private_key)
    result = rebuild_v10_store(
        exchange,
        legacy_path=args.legacy_db,
        output_path=args.output_db,
        archive_dir=args.archive_dir,
        scope=_scope(settings.trading_account_id.get_secret_value()),
        symbols=settings.universe_list,
        since_ms=args.since,
        end_ms=end_ms,
        balance_tolerance=Decimal(str(settings.reconciliation_balance_tolerance)),
    )
    print(result.report_path)  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
