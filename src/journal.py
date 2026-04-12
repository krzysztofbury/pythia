"""Trade Journal — SQLite-backed persistent logging and strategy context."""

from __future__ import annotations

import datetime
import logging
import sqlite3
import threading
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models import PositionState

logger = logging.getLogger(__name__)

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS trades (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    date        TEXT    NOT NULL,
    day_of_week TEXT    NOT NULL,
    account     TEXT    NOT NULL,
    instrument  TEXT    NOT NULL,
    side        TEXT    NOT NULL,
    price       REAL    NOT NULL,
    quantity    REAL    NOT NULL,
    value       REAL    NOT NULL,
    currency    TEXT    NOT NULL,
    profit      REAL    NOT NULL DEFAULT 0.0,
    tax         REAL    NOT NULL DEFAULT 0.0,
    reasoning   TEXT    NOT NULL DEFAULT '',
    fee         REAL    NOT NULL DEFAULT 0.0,
    fill_source TEXT    NOT NULL DEFAULT 'bot'
);
"""

_POSITION_STATE_SCHEMA = """\
CREATE TABLE IF NOT EXISTS position_state (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT    NOT NULL UNIQUE,
    entry_price     REAL    NOT NULL,
    high_watermark  REAL    NOT NULL,
    stop_mode       TEXT    NOT NULL DEFAULT 'FIXED',
    current_sl      REAL    NOT NULL DEFAULT 0.0,
    current_tp      REAL    NOT NULL DEFAULT 0.0,
    partial_exits   INTEGER NOT NULL DEFAULT 0,
    regime_at_entry TEXT    NOT NULL DEFAULT 'RANGING',
    updated_at      TEXT    NOT NULL
);
"""

_MARKET_SNAPSHOT_SCHEMA = """\
CREATE TABLE IF NOT EXISTS market_snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    ticker          TEXT    NOT NULL,
    price           REAL    NOT NULL,
    sentiment       TEXT    NOT NULL DEFAULT '',
    consensus       TEXT    NOT NULL DEFAULT '',
    fear_greed      TEXT    NOT NULL DEFAULT '',
    polymarket_odds TEXT    NOT NULL DEFAULT '',
    indicators_json TEXT    NOT NULL DEFAULT '{}',
    regime          TEXT    NOT NULL DEFAULT 'RANGING'
);
"""


_SIGNALS_SCHEMA = """\
CREATE TABLE IF NOT EXISTS signals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    ticker          TEXT    NOT NULL,
    action          TEXT    NOT NULL,
    confidence      REAL    NOT NULL,
    composite       REAL    NOT NULL,
    vol_regime      TEXT    NOT NULL DEFAULT 'NORMAL',
    reasoning       TEXT    NOT NULL DEFAULT ''
);
"""

_OPTIMIZATION_RUNS_SCHEMA = """\
CREATE TABLE IF NOT EXISTS optimization_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    symbol          TEXT    NOT NULL,
    trials          INTEGER NOT NULL,
    best_sharpe     REAL    NOT NULL,
    params_json     TEXT    NOT NULL,
    validation_json TEXT    NOT NULL DEFAULT '{}'
);
"""


class TradeJournal:
    """SQLite-backed trade journal (thread-safe via self._lock)."""

    def __init__(self, db_path: str = "logs/trades.db") -> None:
        """Initialize the journal with a SQLite database."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: parallel data collection reads from threads.
        # All _conn access is serialized via self._lock.
        self._conn = sqlite3.connect(
            str(self.db_path),
            timeout=10.0,
            check_same_thread=False,
        )
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(_SCHEMA)
            self._conn.execute(_POSITION_STATE_SCHEMA)
            self._conn.execute(_MARKET_SNAPSHOT_SCHEMA)
            self._conn.execute(_SIGNALS_SCHEMA)
            self._conn.execute(_OPTIMIZATION_RUNS_SCHEMA)
            # Migrate existing DBs: add fee and fill_source columns if missing
            for col, ddl in [
                ("fee", "ALTER TABLE trades ADD COLUMN fee REAL NOT NULL DEFAULT 0.0"),
                (
                    "fill_source",
                    "ALTER TABLE trades ADD COLUMN fill_source TEXT NOT NULL DEFAULT 'bot'",
                ),
            ]:
                try:
                    self._conn.execute(ddl)
                except sqlite3.OperationalError:
                    pass  # column already exists
            self._conn.commit()

    def log_trade(  # noqa: PLR0913
        self,
        account: str,
        instrument: str,
        side: str,
        price: float,
        quantity: float,
        value: float,
        currency: str,
        reasoning: str,
        profit: float = 0.0,
        tax: float = 0.0,
        fee: float = 0.0,
        fill_source: str = "bot",
    ) -> None:
        """Insert a new trade record."""
        assert price > 0, f"Trade price must be positive, got {price}"  # noqa: S101
        assert quantity > 0, f"Trade quantity must be positive, got {quantity}"  # noqa: S101
        assert side in ("BUY", "SELL", "CANCEL"), f"Unknown trade side: {side!r}"  # noqa: S101

        now = datetime.datetime.now(tz=datetime.timezone.utc)
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_name = days[now.weekday()]
        date_str = now.strftime("%Y-%m-%d")
        ts_str = now.isoformat()

        with self._lock:
            self._conn.execute(
                "INSERT INTO trades "
                "(timestamp, date, day_of_week, account, instrument, side, "
                " price, quantity, value, currency, profit, tax, reasoning, "
                " fee, fill_source) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    ts_str,
                    date_str,
                    day_name,
                    account,
                    instrument,
                    side,
                    price,
                    quantity,
                    value,
                    currency,
                    profit,
                    tax,
                    reasoning,
                    fee,
                    fill_source,
                ),
            )
            self._conn.commit()

    def get_last_buy_price(self, ticker: str) -> float | None:
        """Find the most recent BUY price for a ticker."""
        with self._lock:
            row = self._conn.execute(
                "SELECT price FROM trades "
                "WHERE instrument = ? AND side = 'BUY' "
                "ORDER BY id DESC LIMIT 1",
                (ticker,),
            ).fetchone()
        return float(row[0]) if row else None

    def get_trade_count(self, ticker: str, side: str = "BUY") -> int:
        """Count net active trades for a ticker."""
        with self._lock:
            if side.upper() == "BUY":
                row = self._conn.execute(
                    "SELECT "
                    "  COALESCE(SUM(CASE WHEN side = 'BUY' THEN 1 ELSE 0 END), 0) - "
                    "  COALESCE(SUM(CASE WHEN side IN ('SELL', 'CANCEL') "
                    "    THEN 1 ELSE 0 END), 0) "
                    "FROM trades WHERE instrument = ?",
                    (ticker,),
                ).fetchone()
                return max(int(row[0]), 0) if row else 0

            row = self._conn.execute(
                "SELECT COUNT(*) FROM trades WHERE instrument = ? AND side = ?",
                (ticker, side.upper()),
            ).fetchone()
            return int(row[0]) if row else 0

    def get_recent_history(self, limit: int = 10, ticker: str | None = None) -> str:
        """Return recent trades as a readable string for strategy context."""
        with self._lock:
            if ticker:
                rows = self._conn.execute(
                    "SELECT date, side, price, quantity, reasoning "
                    "FROM trades WHERE instrument = ? ORDER BY id DESC LIMIT ?",
                    (ticker, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT date, instrument, side, price, quantity, reasoning "
                    "FROM trades ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()

        if not rows:
            return "No previous trades recorded."

        lines = []
        for row in reversed(rows):
            if ticker:
                lines.append(
                    f"{row[0]} {row[1]} {row[2]:.8f} x{row[3]:.8f} — {row[4]}",
                )
            else:
                lines.append(
                    f"{row[0]} {row[1]} {row[2]} {row[3]:.8f} x{row[4]:.8f} — {row[5]}",
                )
        return "\n".join(lines)

    def get_average_cost_basis(self, ticker: str) -> float | None:
        """Weighted average price of un-exited BUYs for a ticker."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT side, price, quantity FROM trades WHERE instrument = ? ORDER BY id ASC",
                (ticker,),
            ).fetchall()

        if not rows:
            return None

        total_qty = 0.0
        total_cost = 0.0

        for side, price, qty in rows:
            if side == "BUY":
                total_cost += price * qty
                total_qty += qty
            elif side in ("SELL", "CANCEL"):
                if total_qty > 0:
                    average_cost = total_cost / total_qty
                    sold_qty = min(qty, total_qty)
                    total_cost -= average_cost * sold_qty
                    total_qty -= sold_qty
            else:
                msg = f"Unknown trade side in DB: {side!r}"
                raise AssertionError(msg)

        if total_qty <= 0:
            return None

        assert total_qty > 0, "Division guard: total_qty must be positive"  # noqa: S101
        return total_cost / total_qty

    def get_last_sync_timestamp(self, ticker: str) -> int | None:
        """Get the ms timestamp of the most recent trade for this ticker."""
        with self._lock:
            row = self._conn.execute(
                "SELECT timestamp FROM trades WHERE instrument = ? ORDER BY id DESC LIMIT 1",
                (ticker,),
            ).fetchone()
        if not row:
            return None
        dt = datetime.datetime.fromisoformat(row[0])
        return int(dt.timestamp() * 1000)

    def is_fill_recorded(self, exchange_trade_id: str) -> bool:
        """Check if a Kraken fill ID is already in our journal.
        We embed the exchange trade ID in the reasoning field as [ID].
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM trades WHERE reasoning LIKE ?",
                (f"%[{exchange_trade_id}]%",),
            ).fetchone()
        return row is not None

    def compute_pnl_with_fees(
        self,
        ticker: str,
        sell_price: float,
        sell_qty: float,
        sell_fee: float,
    ) -> tuple[float, float]:
        """Calculate realized P&L including buy and sell fees.
        Returns (profit_after_fees, tax). Tax rate = 0.19.
        """
        avg_cost = self.get_average_cost_basis(ticker)
        if avg_cost is None:
            avg_cost = sell_price

        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(fee), 0.0) FROM trades WHERE instrument = ? AND side = 'BUY'",
                (ticker,),
            ).fetchone()
        total_buy_fees = float(row[0]) if row else 0.0

        with self._lock:
            qty_row = self._conn.execute(
                "SELECT COALESCE(SUM(quantity), 0.0) FROM trades "
                "WHERE instrument = ? AND side = 'BUY'",
                (ticker,),
            ).fetchone()
        total_buy_qty = float(qty_row[0]) if qty_row else 1.0

        proportional_buy_fee = total_buy_fees * (sell_qty / max(total_buy_qty, 1e-12))
        gross_profit = (sell_price - avg_cost) * sell_qty
        profit = gross_profit - proportional_buy_fee - sell_fee
        tax = profit * 0.19 if profit > 0 else 0.0
        return profit, tax

    def get_position_open_time(self, ticker: str) -> datetime.datetime | None:
        """Return the timestamp of the earliest un-exited BUY for a ticker."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT side, timestamp FROM trades WHERE instrument = ? ORDER BY id ASC",
                (ticker,),
            ).fetchall()

        if not rows:
            return None

        open_count = 0
        first_open_ts: str | None = None
        for side, ts in rows:
            if side == "BUY":
                if open_count == 0:
                    first_open_ts = ts
                open_count += 1
            elif side in ("SELL", "CANCEL"):
                open_count = max(open_count - 1, 0)
                if open_count == 0:
                    first_open_ts = None

        if open_count > 0 and first_open_ts:
            return datetime.datetime.fromisoformat(first_open_ts)
        return None

    def get_daily_realized_pnl(self) -> float:
        """Return total realized P&L for today (UTC)."""
        today = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y-%m-%d")
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(profit), 0.0) FROM trades WHERE date = ? AND side = 'SELL'",
                (today,),
            ).fetchone()
        return float(row[0]) if row else 0.0

    def upsert_position_state(self, state: PositionState) -> None:
        """Insert or update position tracking state."""
        now = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                "INSERT INTO position_state "
                "(ticker, entry_price, high_watermark, stop_mode, current_sl, current_tp, "
                " partial_exits, regime_at_entry, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(ticker) DO UPDATE SET "
                "entry_price=excluded.entry_price, "
                "high_watermark=excluded.high_watermark, stop_mode=excluded.stop_mode, "
                "current_sl=excluded.current_sl, current_tp=excluded.current_tp, "
                "partial_exits=excluded.partial_exits, updated_at=excluded.updated_at",
                (
                    state.ticker,
                    state.entry_price,
                    state.high_watermark,
                    state.stop_mode.value,
                    state.current_sl,
                    state.current_tp,
                    state.partial_exits,
                    state.regime_at_entry,
                    now,
                ),
            )
            self._conn.commit()

    def get_position_state(self, ticker: str) -> PositionState | None:
        """Get position tracking state for a ticker."""
        from models import PositionState, StopMode

        with self._lock:
            row = self._conn.execute(
                "SELECT ticker, entry_price, high_watermark, stop_mode, "
                "current_sl, current_tp, partial_exits, regime_at_entry "
                "FROM position_state WHERE ticker = ?",
                (ticker,),
            ).fetchone()
        if not row:
            return None
        return PositionState(
            ticker=row[0],
            entry_price=row[1],
            high_watermark=row[2],
            stop_mode=StopMode(row[3]),
            current_sl=row[4],
            current_tp=row[5],
            partial_exits=row[6],
            regime_at_entry=row[7],
        )

    def delete_position_state(self, ticker: str) -> None:
        """Remove position state (after full exit)."""
        with self._lock:
            self._conn.execute("DELETE FROM position_state WHERE ticker = ?", (ticker,))
            self._conn.commit()

    def update_high_watermark(self, ticker: str, price: float) -> None:
        """Update high watermark if price exceeds current."""
        with self._lock:
            self._conn.execute(
                "UPDATE position_state SET high_watermark = MAX(high_watermark, ?), "
                "updated_at = ? WHERE ticker = ?",
                (price, datetime.datetime.now(tz=datetime.timezone.utc).isoformat(), ticker),
            )
            self._conn.commit()

    def save_market_snapshot(  # noqa: PLR0913
        self,
        ticker: str,
        price: float,
        sentiment: str,
        consensus: str,
        fear_greed: str,
        polymarket_odds: str,
        indicators_json: str,
        regime: str,
    ) -> None:
        """Persist a market data snapshot for backtest replay."""
        now = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                "INSERT INTO market_snapshots "
                "(timestamp, ticker, price, sentiment, consensus, fear_greed, "
                " polymarket_odds, indicators_json, regime) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    now,
                    ticker,
                    price,
                    sentiment,
                    consensus,
                    fear_greed,
                    polymarket_odds,
                    indicators_json,
                    regime,
                ),
            )
            self._conn.commit()

    def get_snapshots_for_backtest(self, ticker: str, limit: int = 1000) -> list[dict]:
        """Load historical market snapshots for backtesting."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT timestamp, price, sentiment, consensus, fear_greed, "
                "polymarket_odds, indicators_json, regime "
                "FROM market_snapshots WHERE ticker = ? ORDER BY id DESC LIMIT ?",
                (ticker, limit),
            ).fetchall()
        return [
            {
                "timestamp": r[0],
                "price": r[1],
                "sentiment": r[2],
                "consensus": r[3],
                "fear_greed": r[4],
                "polymarket_odds": r[5],
                "indicators_json": r[6],
                "regime": r[7],
            }
            for r in reversed(rows)
        ]

    # ------------------------------------------------------------------
    # Signals (replaces signals.jsonl file)
    # ------------------------------------------------------------------

    def log_signal(
        self,
        ticker: str,
        action: str,
        confidence: float,
        composite: float,
        vol_regime: str,
        reasoning: str,
    ) -> None:
        """Log a trading signal to the database (replaces signals.jsonl)."""
        now = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                "INSERT INTO signals "
                "(timestamp, ticker, action, confidence, composite, vol_regime, reasoning) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    now,
                    ticker,
                    action,
                    round(confidence, 4),
                    round(composite, 4),
                    vol_regime,
                    reasoning[:300],
                ),
            )
            self._conn.commit()

    def get_recent_signals(self, ticker: str | None = None, limit: int = 50) -> list[dict]:
        """Retrieve recent signals from the database."""
        with self._lock:
            if ticker:
                rows = self._conn.execute(
                    "SELECT timestamp, ticker, action, confidence, composite, "
                    "vol_regime, reasoning FROM signals "
                    "WHERE ticker = ? ORDER BY id DESC LIMIT ?",
                    (ticker, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT timestamp, ticker, action, confidence, composite, "
                    "vol_regime, reasoning FROM signals ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [
            {
                "timestamp": r[0],
                "ticker": r[1],
                "action": r[2],
                "confidence": r[3],
                "composite": r[4],
                "vol_regime": r[5],
                "reasoning": r[6],
            }
            for r in reversed(rows)
        ]

    # ------------------------------------------------------------------
    # Optimization runs (replaces optimization.json file)
    # ------------------------------------------------------------------

    def save_optimization_run(
        self,
        symbol: str,
        trials: int,
        best_sharpe: float,
        params_json: str,
        validation_json: str = "{}",
    ) -> None:
        """Save an optimization run result to the database."""
        now = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                "INSERT INTO optimization_runs "
                "(timestamp, symbol, trials, best_sharpe, params_json, validation_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (now, symbol, trials, best_sharpe, params_json, validation_json),
            )
            self._conn.commit()

    def get_latest_optimization(self, symbol: str | None = None) -> dict | None:
        """Get the most recent optimization run."""
        with self._lock:
            if symbol:
                row = self._conn.execute(
                    "SELECT timestamp, symbol, trials, best_sharpe, "
                    "params_json, validation_json FROM optimization_runs "
                    "WHERE symbol = ? ORDER BY id DESC LIMIT 1",
                    (symbol,),
                ).fetchone()
            else:
                row = self._conn.execute(
                    "SELECT timestamp, symbol, trials, best_sharpe, "
                    "params_json, validation_json FROM optimization_runs "
                    "ORDER BY id DESC LIMIT 1",
                ).fetchone()
        if not row:
            return None
        return {
            "timestamp": row[0],
            "symbol": row[1],
            "trials": row[2],
            "best_sharpe": row[3],
            "params_json": row[4],
            "validation_json": row[5],
        }

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            self._conn.close()
