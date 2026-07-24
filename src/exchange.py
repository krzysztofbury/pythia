"""Kraken exchange integration with strict safety protocols."""

import logging
import time
from decimal import Decimal, InvalidOperation
from typing import Any, TypeGuard

import ccxt  # type: ignore[import-untyped]
from pydantic import SecretStr
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

DEFAULT_REQUEST_TIMEOUT_MS = 30_000
KRAKEN_FILL_PAGE_SIZE = 50
MAX_FILL_PAGES = 20
MAX_PAGINATION_PAGES = 100

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


class FillPaginationLimitError(RuntimeError):
    """Raised instead of silently truncating account-wide fill history."""


class KrakenExchange:
    """Interface for Kraken exchange interactions."""

    fill_window_granularity_ms = 1_000

    def __init__(
        self,
        api_key: SecretStr,
        private_key: SecretStr,
        *,
        request_timeout_ms: int = DEFAULT_REQUEST_TIMEOUT_MS,
    ) -> None:
        """Initialize the Kraken client."""
        if isinstance(request_timeout_ms, bool) or request_timeout_ms <= 0:
            msg = "request_timeout_ms must be a positive integer"
            raise ValueError(msg)
        self._client = ccxt.kraken(
            {
                "apiKey": api_key.get_secret_value(),
                "secret": private_key.get_secret_value(),
                "enableRateLimit": True,
                "timeout": request_timeout_ms,
            },
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
        return self.fetch_open_orders(symbol)

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

    def normalize_limit_order(
        self,
        symbol: str,
        amount: float | Decimal | str,
        price: float | Decimal | str,
        *,
        stop_loss_price: float | Decimal | str | None = None,
    ) -> tuple[Decimal, Decimal, Decimal | None]:
        """Return exact venue values for durable persistence without mutation."""
        price_str, amount_str = self._prepare_limit_order(symbol, price, amount)
        stop_str = None
        if stop_loss_price is not None:
            stop_str = self._prepare_price(symbol, stop_loss_price, "stop_loss_price")
            self._validate_notional(symbol, amount_str, stop_str)
        normalized_stop = None if stop_str is None else Decimal(stop_str)
        return Decimal(amount_str), Decimal(price_str), normalized_stop

    def normalize_market_sell(
        self,
        symbol: str,
        amount: float | Decimal | str,
        *,
        reference_price: float | Decimal | str | None = None,
    ) -> tuple[Decimal, Decimal | None]:
        """Return exact venue market-sell values without mutation."""
        amount_str = self._prepare_amount(symbol, amount)
        if reference_price is None:
            return Decimal(amount_str), None
        price_str = self._prepare_price(symbol, reference_price, "reference_price")
        self._validate_notional(symbol, amount_str, price_str)
        return Decimal(amount_str), Decimal(price_str)

    def normalize_stop_loss(
        self,
        symbol: str,
        amount: float | Decimal | str,
        trigger_price: float | Decimal | str,
    ) -> tuple[Decimal, Decimal]:
        """Return exact venue stop values without mutation."""
        amount_str = self._prepare_amount(symbol, amount)
        trigger_str = self._prepare_price(symbol, trigger_price, "trigger_price")
        self._validate_notional(symbol, amount_str, trigger_str)
        return Decimal(amount_str), Decimal(trigger_str)

    def place_maker_buy(
        self,
        symbol: str,
        price: float | Decimal | str,
        amount: float | Decimal | str,
        *,
        client_order_id: str | None = None,
        stop_loss_price: float | Decimal | str | None = None,
    ) -> dict[str, Any]:
        """Place a LIMIT BUY order with POST-ONLY flag.

        Safety:
        - Enforces 'limit' type.
        - Enforces 'postOnly' to guarantee Maker fees.
        """
        amount_value, price_value, stop_value = self.normalize_limit_order(
            symbol,
            amount,
            price,
            stop_loss_price=stop_loss_price,
        )
        price_str = format(price_value, "f")
        amount_str = format(amount_value, "f")
        params = self._mutation_params(client_order_id)
        params["postOnly"] = True
        if stop_value is not None:
            # Kraken creates this conditional close only if AddOrder accepts the entry.
            params["close"] = {"ordertype": "stop-loss", "price": format(stop_value, "f")}

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

    def place_maker_sell(
        self,
        symbol: str,
        price: float | Decimal | str,
        amount: float | Decimal | str,
        *,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        """Place a LIMIT SELL order with POST-ONLY flag.

        Safety: Enforces 'limit' and 'postOnly'.
        """
        amount_value, price_value, _stop_value = self.normalize_limit_order(symbol, amount, price)
        price_str = format(price_value, "f")
        amount_str = format(amount_value, "f")
        params = self._mutation_params(client_order_id)
        params["postOnly"] = True
        logger.info("I: Placing STRATEGIC MAKER SELL on %s: %s @ %s", symbol, amount_str, price_str)

        return self._client.create_order(
            symbol=symbol,
            type="limit",
            side="sell",
            amount=amount_str,
            price=price_str,
            params=params,
        )

    def place_market_sell(
        self,
        symbol: str,
        amount: float | Decimal | str,
        *,
        client_order_id: str | None = None,
        reference_price: float | Decimal | str | None = None,
    ) -> dict[str, Any]:
        """Place one exact market sell for a controlled emergency exit."""
        amount_value, _reference_value = self.normalize_market_sell(
            symbol,
            amount,
            reference_price=reference_price,
        )
        amount_str = format(amount_value, "f")
        logger.warning("I: Placing EMERGENCY MARKET SELL on %s: %s", symbol, amount_str)
        return self._client.create_order(
            symbol=symbol,
            type="market",
            side="sell",
            amount=amount_str,
            price=None,
            params=self._mutation_params(client_order_id),
        )

    def place_stop_loss(
        self,
        symbol: str,
        amount: float | Decimal | str,
        trigger_price: float | Decimal | str,
        *,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        """Place one exact Kraken stop-loss market sell."""
        amount_value, trigger_value = self.normalize_stop_loss(symbol, amount, trigger_price)
        trigger_price_str = format(trigger_value, "f")
        amount_str = format(amount_value, "f")
        logger.info(
            "I: Placing STOP LOSS on %s: %s @ %s",
            symbol,
            amount_str,
            trigger_price_str,
        )
        return self._client.create_order(
            symbol=symbol,
            type="stop-loss",
            side="sell",
            amount=amount_str,
            price=None,
            params={
                **self._mutation_params(client_order_id),
                "price": trigger_price_str,
            },
        )

    def amend_stop_loss(
        self,
        order_id: str,
        symbol: str,
        amount: float | Decimal | str,
        trigger_price: float | Decimal | str,
    ) -> dict[str, Any]:
        """Atomically amend one exact Kraken stop and retain its order identity."""
        self._validate_exact_id("order_id", order_id)
        amount_value, trigger_value = self.normalize_stop_loss(symbol, amount, trigger_price)
        return self._client.edit_order(
            order_id,
            symbol,
            "stop-loss",
            "sell",
            format(amount_value, "f"),
            None,
            {"stopLossPrice": format(trigger_value, "f")},
        )

    def place_protection_orders(
        self,
        symbol: str,
        amount: float | Decimal | str,
        _entry_price: float | Decimal | str,
        sl_price: float | Decimal | str,
        tp_price: float | Decimal | str,
        *,
        client_order_id: str | None = None,
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
            client_order_id: Exact identity for the stop mutation when supplied.

        """
        orders = []

        sl_price_str = self._prepare_price(symbol, sl_price, "sl_price")
        amount_str = self._prepare_amount(symbol, amount)
        tp_price_str = self._prepare_price(symbol, tp_price, "tp_price")
        self._validate_notional(symbol, amount_str, sl_price_str)

        sl_order = self.place_stop_loss(
            symbol,
            amount_str,
            sl_price_str,
            client_order_id=client_order_id,
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
                    },
                )

        return cancelled

    def cancel_order(
        self,
        order_id: str,
        symbol: str | None = None,
        *,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        """Cancel one known order by an exact exchange or client identifier."""
        self._validate_exact_id("order_id", order_id)
        params = self._mutation_params(client_order_id)
        return self._client.cancel_order(order_id, symbol, params)

    @_retry_transient
    def fetch_order(
        self,
        order_id: str,
        symbol: str | None = None,
        *,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        """Fetch one order by exact exchange ID and verify its client ID if supplied."""
        self._validate_exact_id("order_id", order_id)
        order = self._client.fetch_order(order_id, symbol)
        self._verify_exact_order(order, order_id=order_id, client_order_id=client_order_id)
        return order

    @_retry_transient
    def fetch_open_orders(
        self,
        symbol: str | None = None,
        *,
        client_order_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch open orders, optionally restricted to one exact client ID."""
        params = self._client_id_params(client_order_id)
        orders = (
            self._client.fetch_open_orders(symbol, params=params)
            if params
            else self._client.fetch_open_orders(symbol)
        )
        return self._exact_client_orders(orders, client_order_id)

    @_retry_transient
    def fetch_closed_orders(
        self,
        symbol: str | None = None,
        *,
        client_order_id: str | None = None,
        since_ms: int | None = None,
        until_ms: int | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch closed orders using exact filters rather than inferred matching."""
        params = self._client_id_params(client_order_id)
        if until_ms is not None:
            self._validate_timestamp("until_ms", until_ms)
            params["until"] = until_ms
        if since_ms is not None:
            self._validate_timestamp("since_ms", since_ms)
        orders = self._client.fetch_closed_orders(
            symbol,
            since=since_ms,
            limit=limit,
            params=params,
        )
        return self._exact_client_orders(orders, client_order_id)

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

    @_retry_transient
    def fetch_account_fills(
        self,
        since_ms: int,
        end_ms: int,
        *,
        page_size: int = KRAKEN_FILL_PAGE_SIZE,
        max_pages: int = MAX_FILL_PAGES,
    ) -> list[dict[str, Any]]:
        """Fetch a bounded, fixed-window account fill snapshot using Kraken offsets."""
        self._validate_timestamp("since_ms", since_ms)
        self._validate_timestamp("end_ms", end_ms)
        if end_ms < since_ms:
            msg = "end_ms must be greater than or equal to since_ms"
            raise ValueError(msg)
        # Kraken's TradesHistory endpoint always pages in blocks of 50; CCXT's
        # limit only truncates the parsed response and cannot change that block size.
        if isinstance(page_size, bool) or page_size != KRAKEN_FILL_PAGE_SIZE:
            msg = f"page_size must be exactly {KRAKEN_FILL_PAGE_SIZE} for Kraken"
            raise ValueError(msg)
        if isinstance(max_pages, bool) or not 1 <= max_pages <= MAX_PAGINATION_PAGES:
            msg = f"max_pages must be between 1 and {MAX_PAGINATION_PAGES}"
            raise ValueError(msg)

        end_seconds = end_ms // 1000
        fills: list[dict[str, Any]] = []
        for page_index in range(max_pages):
            offset = page_index * page_size
            page = self._client.fetch_my_trades(
                None,
                since=since_ms,
                limit=page_size,
                params={"end": end_seconds, "ofs": offset},
            )
            for fill in page:
                self._validate_fill_identity(fill)
            fills.extend(page)
            if len(page) < page_size:
                return fills
        msg = f"fill history exceeded max_pages={max_pages} for fixed end_ms={end_ms}"
        raise FillPaginationLimitError(msg)

    def _prepare_limit_order(
        self,
        symbol: str,
        price: float | Decimal | str,
        amount: float | Decimal | str,
    ) -> tuple[str, str]:
        price_str = self._prepare_price(symbol, price, "price")
        amount_str = self._prepare_amount(symbol, amount)
        self._validate_notional(symbol, amount_str, price_str)
        return price_str, amount_str

    def _prepare_price(
        self,
        symbol: str,
        value: float | Decimal | str,
        name: str,
    ) -> str:
        decimal_value = self._positive_decimal(name, value)
        rounded = str(self._client.price_to_precision(symbol, format(decimal_value, "f")))
        rounded_value = self._positive_decimal(f"rounded {name}", rounded)
        self._validate_market_limit(symbol, "price", rounded_value)
        return format(rounded_value, "f")

    def _prepare_amount(
        self,
        symbol: str,
        value: float | Decimal | str,
    ) -> str:
        decimal_value = self._positive_decimal("amount", value)
        rounded = str(self._client.amount_to_precision(symbol, format(decimal_value, "f")))
        rounded_value = self._positive_decimal("rounded amount", rounded)
        self._validate_market_limit(symbol, "amount", rounded_value)
        return format(rounded_value, "f")

    def _validate_notional(self, symbol: str, amount: str, price: str) -> None:
        self._validate_market_limit(symbol, "cost", Decimal(amount) * Decimal(price))

    def _validate_market_limit(self, symbol: str, kind: str, value: Decimal) -> None:
        market = self._market(symbol)
        limits = market.get("limits") or {}
        bounds = limits.get(kind) or {}
        minimum = bounds.get("min")
        maximum = bounds.get("max")
        if minimum is not None and value < self._decimal_value(minimum, f"{kind} minimum"):
            msg = f"rounded {kind} {value} is below market minimum {minimum} for {symbol}"
            raise ValueError(msg)
        if maximum is not None and value > self._decimal_value(maximum, f"{kind} maximum"):
            msg = f"rounded {kind} {value} exceeds market maximum {maximum} for {symbol}"
            raise ValueError(msg)

    def _market(self, symbol: str) -> dict[str, Any]:
        if self._markets is None:
            self._markets = self._client.load_markets()
        market = self._markets.get(symbol)
        if market is None:
            msg = f"unknown market {symbol!r}"
            raise ValueError(msg)
        if not isinstance(market, dict):
            msg = f"market metadata for {symbol!r} must be a dictionary"
            raise TypeError(msg)
        return market

    @classmethod
    def _positive_decimal(
        cls,
        name: str,
        value: float | Decimal | str,
    ) -> Decimal:
        parsed = cls._decimal_value(value, name)
        if parsed <= 0:
            msg = f"{name} must be positive"
            raise ValueError(msg)
        return parsed

    @staticmethod
    def _decimal_value(value: object, name: str) -> Decimal:
        if isinstance(value, bool):
            msg = f"{name} must be numeric"
            raise TypeError(msg)
        try:
            parsed = value if isinstance(value, Decimal) else Decimal(str(value))
        except (InvalidOperation, ValueError) as exc:
            msg = f"{name} must be a finite decimal"
            raise ValueError(msg) from exc
        if not parsed.is_finite():
            msg = f"{name} must be a finite decimal"
            raise ValueError(msg)
        return parsed

    @classmethod
    def _mutation_params(cls, client_order_id: str | None) -> dict[str, Any]:
        return cls._client_id_params(client_order_id)

    @classmethod
    def _client_id_params(cls, client_order_id: str | None) -> dict[str, Any]:
        if client_order_id is None:
            return {}
        cls._validate_exact_id("client_order_id", client_order_id)
        return {"clientOrderId": client_order_id}

    @staticmethod
    def _validate_exact_id(name: str, value: object) -> TypeGuard[str]:
        if not isinstance(value, str):
            msg = f"{name} must be a string"
            raise TypeError(msg)
        if not value or value.strip() != value:
            msg = f"{name} must be non-empty and have no surrounding whitespace"
            raise ValueError(msg)
        return True

    @staticmethod
    def _validate_timestamp(name: str, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            msg = f"{name} must be an integer"
            raise TypeError(msg)
        if value < 0:
            msg = f"{name} must not be negative"
            raise ValueError(msg)

    @classmethod
    def _verify_exact_order(
        cls,
        order: dict[str, Any],
        *,
        order_id: str,
        client_order_id: str | None,
    ) -> None:
        if order.get("id") != order_id:
            msg = f"exchange returned order {order.get('id')!r}, expected {order_id!r}"
            raise ValueError(msg)
        if client_order_id is None:
            return
        cls._validate_exact_id("client_order_id", client_order_id)
        if cls._order_client_id(order) != client_order_id:
            msg = f"exchange order {order_id!r} has a different clientOrderId"
            raise ValueError(msg)

    @classmethod
    def _exact_client_orders(
        cls,
        orders: list[dict[str, Any]],
        client_order_id: str | None,
    ) -> list[dict[str, Any]]:
        if client_order_id is None:
            return orders
        cls._validate_exact_id("client_order_id", client_order_id)
        return [order for order in orders if cls._order_client_id(order) == client_order_id]

    @staticmethod
    def _order_client_id(order: dict[str, Any]) -> str | None:
        client_order_id = order.get("clientOrderId")
        if client_order_id is not None:
            return str(client_order_id)
        info = order.get("info")
        if isinstance(info, dict) and info.get("cl_ord_id") is not None:
            return str(info["cl_ord_id"])
        return None

    @classmethod
    def _validate_fill_identity(cls, fill: dict[str, Any]) -> None:
        fill_id = fill.get("id")
        order_id = fill.get("order")
        cls._validate_exact_id("fill id", fill_id)
        cls._validate_exact_id("fill order id", order_id)
