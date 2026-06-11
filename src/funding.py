"""Perpetual funding-rate fetcher — read-only, secondary exchange via CCXT.

Kraken spot does not expose perp funding; we use a separate CCXT client
(Binance by default) purely for read-only signal. No orders are placed on
the secondary exchange. Failures degrade to ``None`` so a funding outage
cannot block trading.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class FundingRateFetcher:
    """Pull the latest perpetual funding rate for a spot symbol."""

    def __init__(self, exchange_id: str = "binance") -> None:
        """Lazily construct the secondary ccxt client.

        Any initialization error is caught and the fetcher degrades to a
        no-op — callers get ``None`` back. This keeps trading decoupled
        from a data-source problem.
        """
        self._exchange_id = exchange_id
        self._client = None
        try:
            import ccxt

            cls = getattr(ccxt, exchange_id, None)
            if cls is None:
                logger.warning("FundingRateFetcher: unknown exchange %s", exchange_id)
                return
            self._client = cls({"enableRateLimit": True})
        except Exception:  # noqa: BLE001
            logger.warning(
                "FundingRateFetcher: failed to init %s client; disabled.",
                exchange_id,
                exc_info=True,
            )
            self._client = None

    @property
    def enabled(self) -> bool:
        """True when the secondary client was constructed successfully."""
        return self._client is not None

    def fetch(self, symbol: str) -> float | None:
        """Return the current funding rate for ``symbol`` or None on any failure.

        ``symbol`` is a spot pair (e.g. ``BTC/USDT``) — we map it to the
        perp contract name (``BTC/USDT:USDT``) which CCXT's unified API
        uses for linear perps on Binance/OKX/Bybit.
        """
        if self._client is None:
            return None
        perp_symbol = self._to_perp(symbol)
        try:
            data = self._client.fetch_funding_rate(perp_symbol)
        except Exception:  # noqa: BLE001
            logger.debug(
                "Funding rate fetch failed for %s (%s)",
                perp_symbol,
                self._exchange_id,
                exc_info=True,
            )
            return None
        rate = data.get("fundingRate") if isinstance(data, dict) else None
        if rate is None:
            return None
        try:
            return float(rate)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _to_perp(spot_symbol: str) -> str:
        """Map a spot pair to a linear perp contract (CCXT unified format).

        Funding-rate signal is quote-agnostic — the funding rate on BTC is
        the funding rate on BTC regardless of whether we hold USDC or USDT
        spot. Binance has much deeper USDT-perp coverage than USDC-perp
        (e.g., DOT/USDC:USDC does not exist, DOT/USDT:USDT does), so we
        always look up USDT-settled perps.
        """
        if ":" in spot_symbol:
            return spot_symbol
        base_quote = spot_symbol.split("/")
        if len(base_quote) != 2:  # noqa: PLR2004
            return spot_symbol
        base = base_quote[0]
        return f"{base}/USDT:USDT"
