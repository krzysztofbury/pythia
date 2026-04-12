"""Shared test fixtures for Oracle Pythia."""
import sqlite3

import numpy as np
import pytest


@pytest.fixture()
def sample_candles() -> list[list]:
    """Generate realistic OHLCV candles for testing (100 candles, 1h)."""
    np.random.seed(42)
    base_price = 100.0
    candles = []
    ts = 1_700_000_000_000  # arbitrary start ms

    for i in range(100):
        # Random walk
        change = np.random.normal(0, 0.5)
        open_p = base_price + change
        high_p = open_p + abs(np.random.normal(0, 0.3))
        low_p = open_p - abs(np.random.normal(0, 0.3))
        close_p = open_p + np.random.normal(0, 0.2)
        volume = np.random.uniform(100, 1000)

        candles.append([
            ts + i * 3_600_000,
            round(open_p, 4),
            round(high_p, 4),
            round(low_p, 4),
            round(close_p, 4),
            round(volume, 2),
        ])
        base_price = close_p

    return candles


@pytest.fixture()
def sample_candles_short() -> list[list]:
    """Only 10 candles — below minimum for most indicators."""
    np.random.seed(99)
    candles = []
    ts = 1_700_000_000_000
    price = 50.0

    for i in range(10):
        o = price + np.random.normal(0, 0.1)
        h = o + 0.2
        l_ = o - 0.2
        c = o + np.random.normal(0, 0.05)
        candles.append([ts + i * 3_600_000, round(o, 4), round(h, 4), round(l_, 4), round(c, 4), 500.0])
        price = c

    return candles


@pytest.fixture()
def sqlite_journal(tmp_path):
    """Create a SQLite journal with sample trades and return the TradeJournal instance."""
    from journal import TradeJournal

    db_path = str(tmp_path / "trades.db")
    journal = TradeJournal(db_path=db_path)

    trades = [
        ("2024-01-01", "Monday", "Kraken", "BTC/USDT", "BUY",
         42000.0, 0.001, 42.0, "USDT", 0.0, 0.0, "Bullish momentum"),
        ("2024-01-02", "Tuesday", "Kraken", "BTC/USDT", "SELL",
         43000.0, 0.001, 43.0, "USDT", 1.0, 0.19, "Take profit"),
        ("2024-01-03", "Wednesday", "Kraken", "ETH/USDT", "BUY",
         2200.0, 0.01, 22.0, "USDT", 0.0, 0.0, "DCA entry"),
        ("2024-01-04", "Thursday", "Kraken", "ETH/USDT", "SELL",
         2100.0, 0.01, 21.0, "USDT", -1.0, 0.0, "Stop loss"),
        ("2024-01-05", "Friday", "Kraken", "SOL/USDT", "BUY",
         100.0, 0.5, 50.0, "USDT", 0.0, 0.0, "Breakout"),
        ("2024-01-06", "Saturday", "Kraken", "SOL/USDT", "SELL",
         110.0, 0.5, 55.0, "USDT", 5.0, 0.95, "Profit target"),
    ]

    for t in trades:
        journal._conn.execute(  # noqa: SLF001
            "INSERT INTO trades "
            "(timestamp, date, day_of_week, account, instrument, side, "
            " price, quantity, value, currency, profit, tax, reasoning) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (f"{t[0]}T00:00:00+00:00", *t),
        )
    journal._conn.commit()  # noqa: SLF001
    return journal


@pytest.fixture()
def sample_journal_content() -> str:
    """Sample markdown trade journal content (for migration tests)."""
    header = (
        "| Date | Day of Week | Account | Instrument | Side | "
        "Price | Quantity | Value | Currency | Profit | Tax (19%) | Reasoning |\n"
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
    )
    rows = [
        "| 2024-01-01 | Monday | Kraken | BTC/USDT | BUY | 42000.00000000 | 0.00100000 | 42.00 | USDT | 0.00 | 0.00 | Bullish momentum |\n",
        "| 2024-01-02 | Tuesday | Kraken | BTC/USDT | SELL | 43000.00000000 | 0.00100000 | 43.00 | USDT | 1.00 | 0.19 | Take profit |\n",
        "| 2024-01-03 | Wednesday | Kraken | ETH/USDT | BUY | 2200.00000000 | 0.01000000 | 22.00 | USDT | 0.00 | 0.00 | DCA entry |\n",
        "| 2024-01-04 | Thursday | Kraken | ETH/USDT | SELL | 2100.00000000 | 0.01000000 | 21.00 | USDT | -1.00 | 0.00 | Stop loss |\n",
        "| 2024-01-05 | Friday | Kraken | SOL/USDT | BUY | 100.00000000 | 0.50000000 | 50.00 | USDT | 0.00 | 0.00 | Breakout |\n",
        "| 2024-01-06 | Saturday | Kraken | SOL/USDT | SELL | 110.00000000 | 0.50000000 | 55.00 | USDT | 5.00 | 0.95 | Profit target |\n",
    ]
    return header + "".join(rows)
