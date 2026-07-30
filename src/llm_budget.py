"""Persistent, process-safe budgets for paid LLM calls."""

from __future__ import annotations

import logging
import math
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

_MICRO_USD_PER_USD = 1_000_000

# Reservations older than this are crash orphans: a reserve() with no
# finalize() outlives every bounded LLM call (timeout <= 120 s) by far.
_STALE_RESERVATION_SECONDS = 3600

logger = logging.getLogger("OraclePythia.LLMBudget")


class BudgetExceededError(RuntimeError):
    """Report that an LLM call cannot fit within a configured budget."""

    def __init__(self, scope: str, limit_usd: float) -> None:
        """Build an error naming the exhausted cap and its limit."""
        self.scope = scope
        self.limit_usd = limit_usd
        super().__init__(f"LLM {scope} budget exhausted (limit ${limit_usd:.6f}).")


@dataclass(frozen=True, slots=True)
class BudgetReservation:
    """Identify an amount reserved atomically in the budget ledger."""

    reservation_id: str
    reserved_usd: float


@dataclass(frozen=True, slots=True)
class BudgetSnapshot:
    """Expose current UTC-period spend, including active reservations."""

    daily_usd: float
    monthly_usd: float
    shadow_monthly_usd: float


class LLMBudgetManager:
    """Coordinate daily, monthly, and shadow LLM caps through SQLite.

    ``None`` disables a cap. A zero cap rejects every positive reservation.
    Shadow reservations count against the normal caps and the additional
    monthly shadow cap.
    """

    def __init__(
        self,
        db_path: str | Path,
        *,
        daily_cap_usd: float | None = None,
        monthly_cap_usd: float | None = None,
        shadow_cap_usd: float | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Initialize caps and create the persistent ledger if needed."""
        self._db_path = str(db_path)
        self._daily_cap = _optional_microusd(daily_cap_usd, "daily_cap_usd")
        self._monthly_cap = _optional_microusd(monthly_cap_usd, "monthly_cap_usd")
        self._shadow_cap = _optional_microusd(shadow_cap_usd, "shadow_cap_usd")
        self._clock = clock or (lambda: datetime.now(UTC))
        self._initialize()

    def reserve(self, amount_usd: float, *, shadow: bool = False) -> BudgetReservation:
        """Atomically reserve ``amount_usd`` or raise when any cap would be exceeded."""
        amount = _microusd(amount_usd, "amount_usd")
        now = self._utc_now()
        day = now.date().isoformat()
        month = day[:7]
        reservation_id = uuid.uuid4().hex

        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            daily = self._period_total(connection, "day", day)
            monthly = self._period_total(connection, "month", month)
            shadow_monthly = self._period_total(connection, "month", month, shadow=True)
            self._check_cap("daily", self._daily_cap, daily, amount)
            self._check_cap("monthly", self._monthly_cap, monthly, amount)
            if shadow:
                self._check_cap("shadow", self._shadow_cap, shadow_monthly, amount)
            connection.execute(
                """
                INSERT INTO llm_budget_reservations (
                    reservation_id, created_at, day, month, shadow,
                    reserved_microusd, actual_microusd, status
                ) VALUES (?, ?, ?, ?, ?, ?, NULL, 'active')
                """,
                (reservation_id, now.isoformat(), day, month, int(shadow), amount),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

        return BudgetReservation(reservation_id, amount / _MICRO_USD_PER_USD)

    def finalize(
        self,
        reservation: BudgetReservation,
        actual_usd: float | None = None,
    ) -> None:
        """Finalize a reservation with actual cost, or its full value when unknown."""
        actual = None if actual_usd is None else _microusd(actual_usd, "actual_usd")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT reserved_microusd, actual_microusd, status
                FROM llm_budget_reservations WHERE reservation_id = ?
                """,
                (reservation.reservation_id,),
            ).fetchone()
            if row is None:
                msg = f"Unknown LLM budget reservation: {reservation.reservation_id}"
                raise ValueError(msg)  # noqa: TRY301
            if row[2] == "finalized":
                expected = row[1]
                candidate = row[0] if actual is None else actual
                if expected != candidate:
                    msg = "LLM budget reservation was already finalized with a different cost."
                    raise ValueError(msg)  # noqa: TRY301
                connection.commit()
                return
            final_cost = row[0] if actual is None else actual
            connection.execute(
                """
                UPDATE llm_budget_reservations
                SET actual_microusd = ?, status = 'finalized'
                WHERE reservation_id = ? AND status = 'active'
                """,
                (final_cost, reservation.reservation_id),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def snapshot(self) -> BudgetSnapshot:
        """Return spend plus active reservations for the current UTC periods."""
        now = self._utc_now()
        day = now.date().isoformat()
        month = day[:7]
        connection = self._connect()
        try:
            daily = self._period_total(connection, "day", day)
            monthly = self._period_total(connection, "month", month)
            shadow = self._period_total(connection, "month", month, shadow=True)
        finally:
            connection.close()
        return BudgetSnapshot(
            daily_usd=daily / _MICRO_USD_PER_USD,
            monthly_usd=monthly / _MICRO_USD_PER_USD,
            shadow_monthly_usd=shadow / _MICRO_USD_PER_USD,
        )

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_budget_reservations (
                    reservation_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    day TEXT NOT NULL,
                    month TEXT NOT NULL,
                    shadow INTEGER NOT NULL CHECK (shadow IN (0, 1)),
                    reserved_microusd INTEGER NOT NULL CHECK (reserved_microusd >= 0),
                    actual_microusd INTEGER CHECK (actual_microusd >= 0),
                    status TEXT NOT NULL
                        CHECK (status IN ('active', 'finalized', 'expired'))
                )
                """,
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_llm_budget_period
                ON llm_budget_reservations (day, month, shadow, status)
                """,
            )
            connection.commit()
        finally:
            connection.close()
        self._expire_stale_reservations()

    def _expire_stale_reservations(self) -> None:
        """Release crash-orphaned reservations so a SIGKILL cannot brick the caps.

        A reservation is released only after ``_STALE_RESERVATION_SECONDS`` —
        far beyond any bounded call — so a concurrent in-flight call can never
        have its budget expired underneath it. Expired rows stop counting
        toward every cap (their ``actual_microusd`` stays NULL).
        """
        cutoff = (self._utc_now() - timedelta(seconds=_STALE_RESERVATION_SECONDS)).isoformat()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE llm_budget_reservations
                SET status = 'expired'
                WHERE status = 'active' AND created_at < ?
                """,
                (cutoff,),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        if cursor.rowcount:
            logger.warning(
                "Released %d crash-orphaned LLM budget reservation(s) older than %ds.",
                cursor.rowcount,
                _STALE_RESERVATION_SECONDS,
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._db_path, timeout=30.0)
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _utc_now(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None:
            msg = "LLMBudgetManager clock must return a timezone-aware datetime."
            raise ValueError(msg)
        return now.astimezone(UTC)

    @staticmethod
    def _period_total(
        connection: sqlite3.Connection,
        column: str,
        value: str,
        *,
        shadow: bool = False,
    ) -> int:
        if column not in {"day", "month"}:
            msg = f"Unsupported budget period: {column}"
            raise ValueError(msg)
        queries = {
            ("day", False): """
                SELECT COALESCE(SUM(
                    CASE WHEN status = 'active' THEN reserved_microusd ELSE actual_microusd END
                ), 0)
                FROM llm_budget_reservations WHERE day = ?
            """,
            ("day", True): """
                SELECT COALESCE(SUM(
                    CASE WHEN status = 'active' THEN reserved_microusd ELSE actual_microusd END
                ), 0)
                FROM llm_budget_reservations WHERE day = ? AND shadow = 1
            """,
            ("month", False): """
                SELECT COALESCE(SUM(
                    CASE WHEN status = 'active' THEN reserved_microusd ELSE actual_microusd END
                ), 0)
                FROM llm_budget_reservations WHERE month = ?
            """,
            ("month", True): """
                SELECT COALESCE(SUM(
                    CASE WHEN status = 'active' THEN reserved_microusd ELSE actual_microusd END
                ), 0)
                FROM llm_budget_reservations WHERE month = ? AND shadow = 1
            """,
        }
        row = connection.execute(
            queries[(column, shadow)],
            (value,),
        ).fetchone()
        return int(row[0])

    @staticmethod
    def _check_cap(scope: str, cap: int | None, used: int, requested: int) -> None:
        if cap is not None and used + requested > cap:
            raise BudgetExceededError(scope, cap / _MICRO_USD_PER_USD)


def _optional_microusd(value: float | None, name: str) -> int | None:
    if value is None:
        return None
    return _microusd(value, name)


def _microusd(value: float, name: str) -> int:
    if not math.isfinite(value):
        msg = f"{name} must be finite."
        raise ValueError(msg)
    if value < 0:
        msg = f"{name} must be >= 0."
        raise ValueError(msg)
    return math.ceil(value * _MICRO_USD_PER_USD)
