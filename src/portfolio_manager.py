"""Portfolio manager — ticker classification and portfolio-level context helpers."""

import logging

from cache import CachedExchange
from config import Settings
from exchange import KrakenExchange
from paper_portfolio import PaperPortfolio

logger = logging.getLogger("OraclePythia")


def compute_available_slots(n_available: int, n_owned: int, max_total: int) -> int:
    """Number of entry slots capital should be divided across.

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
    cached_exchange: CachedExchange,
    paper_portfolio: PaperPortfolio | None,
) -> tuple[list[str], list[str]]:
    """Split the universe into owned and available tickers.

    Returns (owned_tickers, available_tickers).
    """
    owned_tickers: list[str] = []
    available_tickers: list[str] = []

    for ticker in settings.universe_list:
        if not exchange.symbol_exists(ticker):
            logger.warning(
                "Skipping %s. Symbol not supported by Kraken.",
                ticker,
            )
            continue
        try:
            if settings.trading_mode == "PAPER" and paper_portfolio:
                is_owned = paper_portfolio.has_open_position(ticker)
            else:
                is_owned = cached_exchange.has_open_position(ticker)

            if is_owned:
                owned_tickers.append(ticker)
            else:
                available_tickers.append(ticker)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error classifying %s: %s", ticker, exc)

    return owned_tickers, available_tickers
