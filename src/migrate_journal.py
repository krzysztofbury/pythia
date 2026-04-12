"""One-time migration: markdown trade journal -> SQLite."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from journal import TradeJournal

logger = logging.getLogger(__name__)


def migrate_md_to_sqlite(md_path: str, db_path: str) -> int:
    """Parse trades.md and insert all rows into a SQLite journal.

    Returns the number of migrated rows.
    """
    md = Path(md_path)
    if not md.exists():
        logger.warning("Markdown journal not found: %s", md_path)
        return 0

    lines = md.read_text(encoding="utf-8").strip().split("\n")
    # Skip header (row 0) and separator (row 1)
    data_lines = lines[2:]

    journal = TradeJournal(db_path=db_path)
    migrated = 0

    for line in data_lines:
        cols = [c.strip() for c in line.split("|")]
        cols = [c for c in cols if c]
        if len(cols) < 12:  # noqa: PLR2004
            continue
        try:
            # Manually insert with the original date preserved
            date_str = cols[0]
            day_name = cols[1]
            account = cols[2]
            instrument = cols[3]
            side = cols[4].upper()
            price = float(cols[5])
            quantity = float(cols[6])
            value = float(cols[7])
            currency = cols[8]
            profit = float(cols[9])
            tax = float(cols[10])
            reasoning = cols[11] if len(cols) > 11 else ""  # noqa: PLR2004

            journal._conn.execute(  # noqa: SLF001
                "INSERT INTO trades "
                "(timestamp, date, day_of_week, account, instrument, side, "
                " price, quantity, value, currency, profit, tax, reasoning) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (f"{date_str}T00:00:00+00:00", date_str, day_name, account,
                 instrument, side, price, quantity, value, currency,
                 profit, tax, reasoning),
            )
            migrated += 1
        except (ValueError, IndexError) as e:
            logger.warning("Skipping malformed line: %s (%s)", line[:80], e)
            continue

    journal._conn.commit()  # noqa: SLF001
    journal.close()
    return migrated


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    md_file = sys.argv[1] if len(sys.argv) > 1 else "logs/trades.md"
    db_file = sys.argv[2] if len(sys.argv) > 2 else "logs/trades.db"  # noqa: PLR2004

    count = migrate_md_to_sqlite(md_file, db_file)
    logger.info("Migrated %d trades from %s -> %s", count, md_file, db_file)
