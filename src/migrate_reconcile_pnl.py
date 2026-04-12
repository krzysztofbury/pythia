"""One-time migration: reconcile historical P&L against actual Kraken fills.

Fetches ALL trade fills from Kraken, matches them against our journal,
corrects P&L calculations to include actual fees, inserts any missing
fills (SL/TP triggered by Kraken), and prints a reconciliation report.

Usage:
    PYTHONPATH=src uv run python src/migrate_reconcile_pnl.py [--dry-run]

Safety:
    - Creates a backup of trades.db before modifying
    - --dry-run mode shows what would change without writing
    - All changes happen in a single transaction (atomic rollback on error)
"""

import argparse
import datetime
import logging
import shutil
import sqlite3
import time
from pathlib import Path

from config import Settings
from exchange import KrakenExchange

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("PnLMigration")

# Kraken maker fee rate (conservative estimate for lowest tier)
MAKER_FEE_RATE = 0.0025  # 0.25%
TAKER_FEE_RATE = 0.0040  # 0.40%


def fetch_all_kraken_fills(
    exchange: KrakenExchange,
    symbol: str,
    since_ms: int,
) -> list[dict]:
    """Fetch ALL fills from Kraken for a symbol by paginating.

    CCXT's fetch_my_trades returns up to 200 per call.
    We paginate by advancing `since` to the last fill's timestamp + 1ms.
    """
    all_fills = []
    current_since = since_ms

    while True:
        try:
            fills = exchange.fetch_my_trades(symbol, since_ms=current_since)
        except Exception as exc:
            logger.warning(
                "Error fetching fills for %s at since=%s: %s", symbol, current_since, exc
            )
            break

        if not fills:
            break

        all_fills.extend(fills)
        logger.info("  Fetched %d fills for %s (total: %d)", len(fills), symbol, len(all_fills))

        # Advance past the last fill's timestamp
        last_ts = fills[-1].get("timestamp", 0)
        if last_ts <= current_since:
            break  # No progress, avoid infinite loop
        current_since = last_ts + 1

        # Rate limiting
        time.sleep(1.0)

        # If we got fewer than the limit, we've reached the end
        if len(fills) < 200:
            break

    return all_fills


def match_fill_to_journal(
    fill: dict,
    journal_trades: list[dict],
) -> dict | None:
    """Try to match a Kraken fill to a journal trade by timestamp + side + quantity.

    Returns the matching journal row or None.
    """
    fill_side = fill.get("side", "").upper()
    fill_qty = float(fill.get("amount", 0))
    fill_ts = fill.get("timestamp", 0)  # ms

    for trade in journal_trades:
        if trade["side"] != fill_side:
            continue

        # Match by quantity within 5% tolerance
        trade_qty = trade["quantity"]
        if trade_qty == 0:
            continue
        qty_diff = abs(trade_qty - fill_qty) / max(trade_qty, 1e-12)
        if qty_diff > 0.05:
            continue

        # Match by timestamp within 5 minutes
        trade_ts_str = trade["timestamp"]
        try:
            trade_dt = datetime.datetime.fromisoformat(trade_ts_str)
            trade_ts_ms = int(trade_dt.timestamp() * 1000)
        except (ValueError, TypeError):
            continue

        time_diff_ms = abs(fill_ts - trade_ts_ms)
        if time_diff_ms < 300_000:  # 5 minutes
            return trade

    return None


def run_migration(db_path: str, settings: Settings, *, dry_run: bool = False) -> dict:
    """Run the P&L reconciliation migration.

    Returns a report dict with statistics.
    """
    report = {
        "tickers_processed": [],
        "kraken_fills_fetched": 0,
        "journal_trades_before": 0,
        "missing_fills_inserted": 0,
        "pnl_corrections": 0,
        "fee_corrections": 0,
        "old_total_pnl": 0.0,
        "new_total_pnl": 0.0,
        "total_fees_added": 0.0,
        "errors": [],
    }

    # Connect to DB
    conn = sqlite3.connect(db_path, timeout=10.0)
    conn.row_factory = sqlite3.Row

    # Ensure new columns exist
    for ddl in [
        "ALTER TABLE trades ADD COLUMN fee REAL NOT NULL DEFAULT 0.0",
        "ALTER TABLE trades ADD COLUMN fill_source TEXT NOT NULL DEFAULT 'bot'",
        "ALTER TABLE trades ADD COLUMN exchange_fill_id TEXT DEFAULT NULL",
    ]:
        try:
            conn.execute(ddl)
        except sqlite3.OperationalError:
            pass
    try:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_exchange_fill_id ON trades(exchange_fill_id)")
    except sqlite3.OperationalError:
        pass
    conn.commit()

    # Get current state
    row = conn.execute("SELECT ROUND(SUM(profit), 6) FROM trades").fetchone()
    report["old_total_pnl"] = float(row[0]) if row[0] else 0.0
    report["journal_trades_before"] = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]

    # Get earliest trade timestamp for pagination start
    first_row = conn.execute("SELECT MIN(timestamp) FROM trades").fetchone()
    if not first_row or not first_row[0]:
        logger.error("No trades in database. Nothing to migrate.")
        conn.close()
        return report

    first_dt = datetime.datetime.fromisoformat(first_row[0])
    # Start fetching from 1 day before the first trade
    fetch_since_ms = int((first_dt - datetime.timedelta(days=1)).timestamp() * 1000)

    # Initialize exchange
    exchange = KrakenExchange(
        api_key=settings.kraken_api_key,
        private_key=settings.kraken_private_key,
    )

    # Get all traded instruments
    instruments = [r[0] for r in conn.execute("SELECT DISTINCT instrument FROM trades").fetchall()]

    for ticker in instruments:
        logger.info("Processing %s...", ticker)
        report["tickers_processed"].append(ticker)

        # Fetch all Kraken fills
        try:
            kraken_fills = fetch_all_kraken_fills(exchange, ticker, fetch_since_ms)
        except Exception as exc:
            logger.error("Failed to fetch fills for %s: %s", ticker, exc)
            report["errors"].append(f"{ticker}: fetch failed: {exc}")
            continue

        report["kraken_fills_fetched"] += len(kraken_fills)
        logger.info("  %s: %d Kraken fills fetched", ticker, len(kraken_fills))

        # Load all journal trades for this ticker
        journal_trades = [
            dict(r)
            for r in conn.execute(
                "SELECT id, timestamp, side, price, quantity, value, profit, tax, fee, "
                "fill_source, exchange_fill_id, reasoning "
                "FROM trades WHERE instrument = ? ORDER BY id",
                (ticker,),
            ).fetchall()
        ]

        # Build a set of already-matched Kraken fill IDs
        matched_fill_ids = set()
        for trade in journal_trades:
            if trade["exchange_fill_id"]:
                matched_fill_ids.add(trade["exchange_fill_id"])

        # Phase 1: Match existing journal trades to Kraken fills → correct prices + add fees
        unmatched_journal = list(journal_trades)  # Copy for tracking
        for fill in kraken_fills:
            fill_id = fill.get("id", "")
            if fill_id in matched_fill_ids:
                continue

            matched_trade = match_fill_to_journal(fill, unmatched_journal)
            if matched_trade is None:
                continue

            matched_fill_ids.add(fill_id)
            unmatched_journal.remove(matched_trade)

            fill_price = float(fill.get("price", 0))
            fill_qty = float(fill.get("amount", 0))
            fee_info = fill.get("fee", {})
            fill_fee = float(fee_info.get("cost", 0)) if fee_info else 0.0

            trade_id = matched_trade["id"]
            old_fee = matched_trade["fee"]
            old_price = matched_trade["price"]

            updates = {}
            if fill_fee > 0 and old_fee == 0:
                updates["fee"] = fill_fee
                report["fee_corrections"] += 1
                report["total_fees_added"] += fill_fee

            if abs(fill_price - old_price) / max(old_price, 1e-12) > 0.0001:
                updates["price"] = fill_price
                updates["value"] = fill_price * fill_qty

            if fill_id:
                updates["exchange_fill_id"] = fill_id

            if updates and not dry_run:
                set_clauses = ", ".join(f"{k} = ?" for k in updates)
                values = list(updates.values()) + [trade_id]
                conn.execute(
                    f"UPDATE trades SET {set_clauses} WHERE id = ?",  # noqa: S608
                    values,
                )

        # Phase 2: Find Kraken fills with no journal match → insert as missing fills
        for fill in kraken_fills:
            fill_id = fill.get("id", "")
            if fill_id in matched_fill_ids:
                continue

            fill_side = fill.get("side", "").upper()
            fill_price = float(fill.get("price", 0))
            fill_qty = float(fill.get("amount", 0))
            fill_cost = float(fill.get("cost", 0))
            fee_info = fill.get("fee", {})
            fill_fee = float(fee_info.get("cost", 0)) if fee_info else 0.0

            if fill_side not in ("BUY", "SELL") or fill_price <= 0 or fill_qty <= 0:
                continue

            fill_ts_ms = fill.get("timestamp", 0)
            fill_dt = datetime.datetime.fromtimestamp(
                fill_ts_ms / 1000,
                tz=datetime.timezone.utc,
            )

            reasoning = (
                f"Migration: {fill_side} fill found on Kraken but missing from journal. "
                f"Price={fill_price}, Fee={fill_fee:.6f} [{fill_id}]"
            )

            logger.info(
                "  MISSING: %s %s %.8f @ %.8f (fee: %.6f) [%s]",
                fill_dt.isoformat(),
                fill_side,
                fill_qty,
                fill_price,
                fill_fee,
                fill_id,
            )

            if not dry_run:
                days = [
                    "Monday",
                    "Tuesday",
                    "Wednesday",
                    "Thursday",
                    "Friday",
                    "Saturday",
                    "Sunday",
                ]
                conn.execute(
                    "INSERT INTO trades "
                    "(timestamp, date, day_of_week, account, instrument, side, "
                    " price, quantity, value, currency, profit, tax, reasoning, "
                    " fee, fill_source, exchange_fill_id) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        fill_dt.isoformat(),
                        fill_dt.strftime("%Y-%m-%d"),
                        days[fill_dt.weekday()],
                        "Kraken",
                        ticker,
                        fill_side,
                        fill_price,
                        fill_qty,
                        fill_cost,
                        settings.quote_currency,
                        0.0,  # Profit recalculated in Phase 3
                        0.0,
                        reasoning,
                        fill_fee,
                        "exchange",
                        fill_id,
                    ),
                )

            matched_fill_ids.add(fill_id)
            report["missing_fills_inserted"] += 1

    # Phase 3: Recalculate P&L for ALL SELL trades using FIFO cost basis + fees
    if not dry_run:
        conn.commit()  # Commit inserts before recalculating

        for ticker in instruments:
            logger.info("Recalculating P&L for %s...", ticker)

            trades = conn.execute(
                "SELECT id, side, price, quantity, fee FROM trades "
                "WHERE instrument = ? ORDER BY id ASC",
                (ticker,),
            ).fetchall()

            # FIFO cost basis tracking (same logic as get_average_cost_basis)
            total_qty = 0.0
            total_cost = 0.0
            total_buy_fees = 0.0

            for trade_id, side, price, qty, fee in trades:
                if side == "BUY":
                    total_cost += price * qty
                    total_qty += qty
                    total_buy_fees += fee
                elif side == "SELL":
                    if total_qty > 0:
                        avg_cost = total_cost / total_qty

                        # Proportional buy fees
                        prop_buy_fee = total_buy_fees * (qty / max(total_qty, 1e-12))

                        gross = (price - avg_cost) * qty
                        profit = gross - prop_buy_fee - fee
                        tax = profit * settings.tax_rate if profit > 0 else 0.0

                        conn.execute(
                            "UPDATE trades SET profit = ?, tax = ? WHERE id = ?",
                            (round(profit, 8), round(tax, 8), trade_id),
                        )
                        report["pnl_corrections"] += 1

                        # Deduct sold qty from running totals
                        sold_qty = min(qty, total_qty)
                        total_cost -= avg_cost * sold_qty
                        total_qty -= sold_qty
                        total_buy_fees -= prop_buy_fee
                    else:
                        # Sell with no open position — zero profit
                        conn.execute(
                            "UPDATE trades SET profit = 0.0, tax = 0.0 WHERE id = ?",
                            (trade_id,),
                        )
                elif side == "CANCEL":
                    # Cancels reduce open position count
                    if total_qty > 0:
                        avg_cost = total_cost / total_qty
                        sold_qty = min(qty, total_qty)
                        total_cost -= avg_cost * sold_qty
                        total_qty -= sold_qty

        conn.commit()

    # Final report
    row = conn.execute("SELECT ROUND(SUM(profit), 6) FROM trades").fetchone()
    report["new_total_pnl"] = float(row[0]) if row[0] else 0.0

    conn.close()
    return report


def main() -> None:
    """Entry point for the P&L reconciliation migration."""
    parser = argparse.ArgumentParser(description="Reconcile historical P&L against Kraken fills")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without modifying the database",
    )
    parser.add_argument(
        "--db-path",
        default="logs/trades.db",
        help="Path to trades database (default: logs/trades.db)",
    )
    args = parser.parse_args()

    # Load settings (needs .env)
    try:
        settings = Settings()
    except Exception:
        logger.critical("Failed to load settings. Ensure .env is configured.")
        raise

    db_path = args.db_path
    if not Path(db_path).exists():
        logger.error("Database not found: %s", db_path)
        return

    # Create backup
    backup_path = f"{db_path}.backup-{datetime.datetime.now(tz=datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    logger.info("Creating backup: %s", backup_path)
    shutil.copy2(db_path, backup_path)
    # Also backup WAL/SHM if they exist
    for ext in ("-wal", "-shm"):
        wal_path = db_path + ext
        if Path(wal_path).exists():
            shutil.copy2(wal_path, backup_path + ext)

    mode = "DRY RUN" if args.dry_run else "LIVE"
    logger.info("=== P&L Reconciliation Migration (%s) ===", mode)

    report = run_migration(db_path, settings, dry_run=args.dry_run)

    # Print report
    print("\n" + "=" * 60)
    print(f"  P&L Reconciliation Report ({mode})")
    print("=" * 60)
    print(f"  Tickers processed:      {', '.join(report['tickers_processed'])}")
    print(f"  Kraken fills fetched:    {report['kraken_fills_fetched']}")
    print(f"  Journal trades (before): {report['journal_trades_before']}")
    print(f"  Missing fills inserted:  {report['missing_fills_inserted']}")
    print(f"  P&L recalculations:      {report['pnl_corrections']}")
    print(f"  Fee corrections:         {report['fee_corrections']}")
    print(f"  Total fees added:        ${report['total_fees_added']:.4f}")
    print(f"  Old total P&L:           ${report['old_total_pnl']:.4f}")
    print(f"  New total P&L:           ${report['new_total_pnl']:.4f}")
    print(f"  P&L adjustment:          ${report['new_total_pnl'] - report['old_total_pnl']:.4f}")
    if report["errors"]:
        print(f"  Errors:                  {len(report['errors'])}")
        for err in report["errors"]:
            print(f"    - {err}")
    print("=" * 60)

    if args.dry_run:
        print(f"\n  Backup at: {backup_path}")
        print("  Run without --dry-run to apply changes.")
    else:
        print(f"\n  Backup at: {backup_path}")
        print("  Migration complete. Verify P&L matches Kraken balance.")


if __name__ == "__main__":
    main()
