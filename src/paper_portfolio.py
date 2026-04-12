"""Paper mode virtual portfolio tracker."""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PaperPosition:
    """A virtual position in paper mode."""

    ticker: str
    quantity: float
    avg_cost: float


class PaperPortfolio:
    """In-memory portfolio for paper trading, rebuilt from SQLite journal on startup.

    Tracks virtual positions so paper mode can evaluate exits, not just entries.
    """

    def __init__(
        self,
        initial_balance: float = 10_000.0,
        quote_currency: str = "USDT",
        db_path: str | None = None,
    ) -> None:
        """Initialize the paper portfolio."""
        self._positions: dict[str, PaperPosition] = {}
        self._balance = initial_balance
        self._quote_currency = quote_currency

        if db_path:
            self._rebuild_from_db(db_path)

    def _rebuild_from_db(self, db_path: str) -> None:
        """Rebuild positions from the SQLite journal."""
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT instrument, side, price, quantity, value "
                "FROM trades ORDER BY id ASC",
            ).fetchall()
            conn.close()

            for row in rows:
                ticker = row["instrument"]
                side = row["side"].upper()
                price = row["price"]
                qty = row["quantity"]
                value = row["value"]

                if side == "BUY":
                    self.record_buy(ticker, price, qty)
                    self._balance -= value
                elif side == "SELL":
                    self.record_sell(ticker, qty)
                    self._balance += value
                elif side == "CANCEL":
                    pass  # No position change on cancelled orders
                else:
                    msg = f"Unknown trade side: {side!r}"
                    raise AssertionError(msg)  # noqa: TRY301

            logger.info(
                "Paper portfolio rebuilt: %d positions, $%.2f balance",
                len(self._positions), self._balance,
            )
        except Exception:  # noqa: BLE001
            logger.warning("Could not rebuild paper portfolio from DB.", exc_info=True)

    def has_open_position(self, symbol: str) -> bool:
        """Check if we have a virtual position in this symbol."""
        pos = self._positions.get(symbol)
        return pos is not None and pos.quantity > 0

    def get_free_balance(self, currency: str) -> float:
        """Return virtual free balance."""
        if currency == self._quote_currency:
            return self._balance
        # For base currencies, return position quantity
        for ticker, pos in self._positions.items():
            if ticker.split("/")[0] == currency:
                return pos.quantity
        return 0.0

    def get_total_balance(self, currency: str) -> float:
        """Return total balance (same as free for paper mode)."""
        return self.get_free_balance(currency)

    def record_buy(self, ticker: str, price: float, quantity: float) -> None:
        """Record a virtual buy, updating average cost basis."""
        assert price > 0, f"Buy price must be positive, got {price}"  # noqa: S101
        assert quantity > 0, f"Buy quantity must be positive, got {quantity}"  # noqa: S101

        existing = self._positions.get(ticker)
        if existing and existing.quantity > 0:
            total_cost = existing.avg_cost * existing.quantity + price * quantity
            total_qty = existing.quantity + quantity
            assert total_qty > 0, "Total quantity must be positive after buy"  # noqa: S101
            existing.avg_cost = total_cost / total_qty
            existing.quantity = total_qty
        else:
            self._positions[ticker] = PaperPosition(
                ticker=ticker, quantity=quantity, avg_cost=price,
            )

    def record_sell(self, ticker: str, quantity: float) -> None:
        """Record a virtual sell."""
        assert quantity > 0, f"Sell quantity must be positive, got {quantity}"  # noqa: S101

        existing = self._positions.get(ticker)
        if existing:
            existing.quantity = max(existing.quantity - quantity, 0.0)
            if existing.quantity <= 0:
                del self._positions[ticker]

    def get_cost_basis(self, ticker: str) -> float | None:
        """Return the weighted average cost basis for a position."""
        pos = self._positions.get(ticker)
        if pos and pos.quantity > 0:
            return pos.avg_cost
        return None
