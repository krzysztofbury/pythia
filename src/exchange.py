"""Kraken exchange integration with strict safety protocols."""

import logging
import time
from typing import Any

import ccxt
from pydantic import SecretStr
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# Retry decorator for transient errors (TLS cert bundle hiccups, connection resets)
_retry_transient = retry(
    retry=retry_if_exception_type((OSError, ConnectionError)),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    stop=stop_after_attempt(3),
    reraise=True,
    before_sleep=lambda rs: logger.warning(
        "Exchange: Transient error, retrying (attempt %d)...",
        rs.attempt_number,
    ),
)


class KrakenExchange:
    """Interface for Kraken exchange interactions."""

    def __init__(self, api_key: SecretStr, private_key: SecretStr) -> None:
        """Initialize the Kraken client."""
        self._client = ccxt.kraken(
            {
                "apiKey": api_key.get_secret_value(),
                "secret": private_key.get_secret_value(),
                "enableRateLimit": True,
            }
        )
        self._markets = None
        # Load markets to ensure precision and limits are available
        try:
            self._client.load_markets()
            self._markets = self._client.markets
        except Exception:
            logger.exception("Failed to load markets on initialization.")

    def symbol_exists(self, symbol: str) -> bool:
        """Check if the exchange supports this symbol."""
        if self._markets is None:
            self._markets = self._client.load_markets()
        return symbol in self._markets

    @_retry_transient
    def fetch_price(self, symbol: str) -> float:
        """Fetch the current last price for a symbol."""
        ticker = self._client.fetch_ticker(symbol)
        # Verify ticker structure (TigerStyle: Trust but verify)
        if "last" not in ticker or ticker["last"] is None:
            msg = f"Invalid ticker data for {symbol}"
            raise ValueError(msg)
        return float(ticker["last"])

    @_retry_transient
    def get_market_structure(self, symbol: str) -> dict[str, float]:
        """Fetch Bid, Ask, and calculate Spread.

        Returns:
            dict with 'bid', 'ask', 'spread_pct'

        """
        ticker = self._client.fetch_ticker(symbol)
        bid = float(ticker.get("bid", 0.0))
        ask = float(ticker.get("ask", 0.0))

        if bid <= 0 or ask <= 0:
            msg = f"Invalid market depth for {symbol}: Bid={bid}, Ask={ask}"
            raise ValueError(msg)

        spread_pct = (ask - bid) / bid
        return {"bid": bid, "ask": ask, "spread_pct": spread_pct}

    @_retry_transient
    def get_24h_stats(self, symbol: str) -> dict[str, float]:
        """Fetch 24-hour ticker statistics (Volume, VWAP/Average)."""
        ticker = self._client.fetch_ticker(symbol)
        # ccxt 'vwap' is volume weighted average price. 'average' is sometimes available.
        # If not, (high+low)/2 is a rough proxy, or use 'last'.
        # Kraken ticker has 'vwap'.
        vwap = float(ticker.get("vwap", 0.0))
        vol = float(ticker.get("baseVolume", 0.0))  # Volume in base currency

        # We also need 'average' or open price to compare.
        # Let's use vwap as the "24h average".

        return {"vwap": vwap, "volume": vol}

    @_retry_transient
    def get_open_orders(self, symbol: str) -> list[dict[str, Any]]:
        """Fetch all open orders for a symbol."""
        return self._client.fetch_open_orders(symbol)

    @_retry_transient
    def get_free_balance(self, currency: str) -> float:
        """Fetch free (available) balance for a specific currency."""
        balance = self._client.fetch_balance()
        return float(balance.get("free", {}).get(currency, 0.0))

    @_retry_transient
    def get_total_balance(self, currency: str) -> float:
        """Fetch total balance (free + locked in orders) for a currency."""
        balance = self._client.fetch_balance()
        return float(balance.get("total", {}).get(currency, 0.0))

    def has_open_position(self, symbol: str) -> bool:
        """Check if we hold a significant USD-value amount of the base asset.

        Uses total balance (not free) because stop-loss orders lock funds,
        making free balance 0 even when we own the asset.
        """
        base_currency = symbol.split("/")[0]
        balance = self.get_total_balance(base_currency)
        if balance <= 0:
            return False
        price = self.fetch_price(symbol)
        usd_value = balance * price
        return usd_value >= 1.0

    def place_maker_buy(self, symbol: str, price: float, amount: float) -> dict[str, Any]:
        """Place a LIMIT BUY order with POST-ONLY flag.

        Safety:
        - Enforces 'limit' type.
        - Enforces 'postOnly' to guarantee Maker fees.
        """
        # TigerStyle Safety: Assertions against Logic Errors
        if price <= 0:
            msg = "Price must be positive"
            raise ValueError(msg)
        if amount <= 0:
            msg = "Amount must be positive"
            raise ValueError(msg)

        # Ensure precision
        price_str = self._client.price_to_precision(symbol, price)
        amount_str = self._client.amount_to_precision(symbol, amount)

        # Explicitly define params for Maker-only
        params = {"postOnly": True}

        # Log intent (in a real app, use logger)
        logger.info("I: Placing MAKER BUY on %s: %s @ %s", symbol, amount_str, price_str)

        return self._client.create_order(
            symbol=symbol,
            type="limit",
            side="buy",
            amount=amount_str,
            price=price_str,
            params=params,
        )

    def cancel_all_orders(self, symbol: str) -> dict[str, Any]:
        """Cancel all open orders for a specific symbol."""
        logger.info("I: Cancelling all open orders for %s", symbol)
        # ccxt Kraken implementation for cancel_all_orders might need params
        return self._client.cancel_all_orders(symbol)

    def place_maker_sell(self, symbol: str, price: float, amount: float) -> dict[str, Any]:
        """Place a LIMIT SELL order with POST-ONLY flag.

        Safety: Enforces 'limit' and 'postOnly'.
        """
        if price <= 0:
            msg = "Price must be positive"
            raise ValueError(msg)
        if amount <= 0:
            msg = "Amount must be positive"
            raise ValueError(msg)

        # Ensure precision
        price_str = self._client.price_to_precision(symbol, price)
        amount_str = self._client.amount_to_precision(symbol, amount)

        params = {"postOnly": True}
        logger.info("I: Placing STRATEGIC MAKER SELL on %s: %s @ %s", symbol, amount_str, price_str)

        return self._client.create_order(
            symbol=symbol,
            type="limit",
            side="sell",
            amount=amount_str,
            price=price_str,
            params=params,
        )

    def place_protection_orders(
        self,
        symbol: str,
        amount: float,
        _entry_price: float,
        sl_price: float,
        tp_price: float,
    ) -> list[dict[str, Any]]:
        """Place Stop Loss and Take Profit orders.

        NOTE:Ideally these should be linked (OCO).
        Currently for SPOT, we can only place ONE order locking the funds.
        PRIORITY: STOP LOSS.

        Args:
            symbol: Ticker
            amount: Quantity
            entry_price: Entry price (for reference)
            sl_price: Stop Loss Price
            tp_price: Take Profit Price

        """
        orders = []

        # Precision handling
        sl_price_str = self._client.price_to_precision(symbol, sl_price)
        amount_str = self._client.amount_to_precision(symbol, amount)
        tp_price_str = self._client.price_to_precision(symbol, tp_price)

        # Stop Loss (Stop Market usually ensures exit, Stop Limit risks non-fill)
        # Using 'stop-loss' type (market sell when triggered)
        logger.info("I: Placing STOP LOSS on %s: %s @ %s", symbol, amount_str, sl_price_str)
        # Fix for Kraken: Trigger price must be in params['price'] for stop-loss
        sl_order = self._client.create_order(
            symbol=symbol,
            type="stop-loss",
            side="sell",
            amount=amount_str,
            price=None,  # Must be None for Kraken stop-loss via CCXT
            params={"price": sl_price_str},  # Trigger price here
        )
        orders.append(sl_order)

        # Take Profit - Skipped for now to avoid Insufficient Funds error on Spot
        logger.warning(
            "I: Skipping TAKE PROFIT %s on Exchange to avoid locking funds. Monitor via Bot.",
            tp_price_str,
        )

        return orders

    @_retry_transient
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 100,
    ) -> list[list]:
        """Fetch OHLCV candle data from the exchange.

        Args:
            symbol: Trading pair (e.g. 'BTC/USDT')
            timeframe: Candle interval ('1m','5m','15m','1h','4h','1d')
            limit: Number of candles to fetch

        Returns:
            List of [timestamp, open, high, low, close, volume] candles.

        """
        return self._client.fetch_ohlcv(symbol, timeframe, limit=limit)

    def fetch_multi_day_average(
        self,
        symbol: str,
        days: int = 7,
    ) -> dict[str, float]:
        """Compute a multi-day SMA and compare with 24h VWAP.

        Returns:
            dict with 'sma', 'vwap', 'price', 'sma_deviation_pct', 'vwap_deviation_pct'

        """
        timeframe = "1d"
        limit = days + 1  # extra candle in case current day is partial
        candles = self._client.fetch_ohlcv(symbol, timeframe, limit=limit)

        # Each candle: [timestamp, open, high, low, close, volume]
        closes = [c[4] for c in candles[-days:]]
        sma = sum(closes) / len(closes) if closes else 0.0

        stats = self.get_24h_stats(symbol)
        vwap = stats["vwap"]
        price = self.fetch_price(symbol)

        sma_dev = ((price - sma) / sma * 100) if sma else 0.0
        vwap_dev = ((price - vwap) / vwap * 100) if vwap else 0.0

        return {
            "sma": sma,
            "vwap": vwap,
            "price": price,
            "sma_deviation_pct": round(sma_dev, 2),
            "vwap_deviation_pct": round(vwap_dev, 2),
        }

    def cancel_stale_orders(
        self,
        symbol: str,
        max_age_seconds: int,
    ) -> list[dict[str, Any]]:
        """Cancel open orders older than max_age_seconds.

        Returns:
            List of cancelled order details (id, side, price, amount).

        """
        open_orders = self.get_open_orders(symbol)
        now = time.time() * 1000  # ccxt timestamps are in milliseconds
        cancelled: list[dict[str, Any]] = []

        for order in open_orders:
            order_ts = order.get("timestamp", 0) or 0
            age_ms = now - order_ts
            if age_ms > max_age_seconds * 1000:
                order_id = order["id"]
                logger.info(
                    "Stale order %s on %s (age: %.0fh). Cancelling.",
                    order_id,
                    symbol,
                    age_ms / 3_600_000,
                )
                self._client.cancel_order(order_id, symbol)
                cancelled.append(
                    {
                        "id": order_id,
                        "side": order.get("side", "unknown"),
                        "price": float(order.get("price", 0) or 0),
                        "amount": float(order.get("amount", 0) or 0),
                    }
                )

        return cancelled

    @_retry_transient
    def fetch_my_trades(
        self,
        symbol: str,
        since_ms: int | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch actual trade fills from Kraken for a symbol.

        Args:
            symbol: Trading pair (e.g. 'BTC/USDC')
            since_ms: Fetch fills after this timestamp (milliseconds).
                      None fetches recent fills.

        Returns:
            List of fill dicts with keys: id, timestamp, symbol, side,
            price, amount, cost, fee (dict with cost + currency).

        """
        return self._client.fetch_my_trades(symbol, since=since_ms, limit=200)
