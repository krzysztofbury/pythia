"""Portfolio manager — ticker classification and portfolio-level context helpers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from execution_store import OrderPurpose, Side

if TYPE_CHECKING:
    from cache import CachedExchange
    from config import Settings
    from exchange import KrakenExchange
    from execution_store import ExecutionStore
    from paper_portfolio import PaperPortfolio

logger = logging.getLogger("OraclePythia")


def compute_available_slots(n_available: int, n_owned: int, max_total: int) -> int:
    """Compute the number of entry slots across which capital is divided.

    v9.6: previously this was just ``len(available_tickers)`` — with a
    7-ticker universe and a 5-position cap, capital was split across 7 slots
    that could never all be filled, shrinking every allocation by 29%.
    """
    assert n_available >= 0 and n_owned >= 0 and max_total >= 1, (  # noqa: S101, PT018
        f"invalid slot inputs: {(n_available, n_owned, max_total)}"
    )
    remaining_cap = max_total - n_owned
    return max(min(n_available, remaining_cap), 1)


def classify_tickers(
    settings: Settings,
    exchange: KrakenExchange,
    cached_exchange: CachedExchange | None = None,
    paper_portfolio: PaperPortfolio | None = None,
    *,
    execution_store: ExecutionStore | None = None,
) -> tuple[list[str], list[str]]:
    """Split the universe into owned and available tickers.

    Returns (owned_tickers, available_tickers).
    """
    owned_tickers: list[str] = []
    available_tickers: list[str] = []
    if execution_store is None and cached_exchange is None and paper_portfolio is None:
        msg = "classification requires ExecutionStore or a legacy portfolio source"
        raise ValueError(msg)

    reserved_entries = (
        reserved_entry_tickers(execution_store) if execution_store is not None else set()
    )
    for ticker in settings.universe_list:
        if not exchange.symbol_exists(ticker):
            logger.warning(
                "Skipping %s. Symbol not supported by Kraken.",
                ticker,
            )
            continue
        try:
            if execution_store is not None:
                is_owned = execution_store.get_position(ticker) is not None
            elif settings.trading_mode == "PAPER" and paper_portfolio:
                is_owned = paper_portfolio.has_open_position(ticker)
            elif cached_exchange is not None:
                is_owned = cached_exchange.has_open_position(ticker)
            else:
                is_owned = False

            if is_owned:
                owned_tickers.append(ticker)
            elif ticker not in reserved_entries:
                available_tickers.append(ticker)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error classifying %s: %s", ticker, exc)

    return owned_tickers, available_tickers


def reserved_entry_tickers(store: ExecutionStore) -> set[str]:
    """Return symbols whose active BUY mutations reserve a future position slot."""
    return {
        str(order["symbol"])
        for order in store.list_order_contexts(active_only=True)
        if order["side"] == Side.BUY.value
        and order["purpose"] in {OrderPurpose.ENTRY.value, OrderPurpose.DCA.value}
    }
