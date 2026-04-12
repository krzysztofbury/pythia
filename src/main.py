"""Main execution loop and Risk Guardian."""

import datetime
import json
import logging
import logging.handlers
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from analytics import PortfolioAnalyzer
from cache import CachedExchange
from config import Settings
from exchange import KrakenExchange
from forecast import PriceForecaster
from indicators import TechnicalAnalyzer
from journal import TradeJournal
from models import MarketData, TradeSignal, fmt_price
from notifier import NullNotifier, WebhookNotifier
from paper_portfolio import PaperPortfolio
from regime import RegimeClassifier
from sentiment import SentimentOracle
from stop_manager import StopManager
from strategy import Strategist
from volatility import VolatilityAnalyzer

logger = logging.getLogger("OraclePythia")

# Constants
MIN_CONFIDENCE = 0.10  # Was 0.6 (designed for Gemini LLM confidence scale, not composite)
MIN_ORDER_USD = 10.0
ROUND_UP_THRESHOLD = 8.3
REGULATORY_DANGER_PHRASES = [
    "REGULATORY BAN",
    "BANNED",
    "DELISTED",
    "CEASE AND DESIST",
    "TRADING HALTED",
    "SECURITIES VIOLATION",
]

# Crypto correlation groups — assets that move together
_CORRELATION_GROUPS: dict[str, str] = {
    "BTC": "btc_ecosystem",
    "ETH": "eth_ecosystem",
    "SOL": "eth_ecosystem",
    "LINK": "eth_ecosystem",
    "DOT": "eth_ecosystem",
    "XRP": "alt_major",
    "DOGE": "meme",
    "SHIB": "meme",
    "LTC": "btc_ecosystem",
}


def setup_logging(settings: Settings) -> None:
    """Configure console and time-based file logging with daily rotation."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    file_handler = logging.handlers.TimedRotatingFileHandler(
        "logs/bot.log",
        when="midnight",
        backupCount=settings.log_retention_days,
        encoding="utf-8",
        utc=True,
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # Structured JSON logger for signals (machine-parseable)
    # JSON signal logger removed — signals now stored in SQLite via journal.log_signal()


def _write_healthcheck(settings: Settings) -> None:
    """Write a healthcheck timestamp file for Docker HEALTHCHECK."""
    try:
        path = Path(settings.healthcheck_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
            encoding="utf-8",
        )
    except Exception:  # noqa: BLE001
        logger.debug("Failed to write healthcheck file.", exc_info=True)


class RiskGuardian:
    """The Gatekeeper. Enforces safety limits before any order goes out."""

    def __init__(
        self,
        settings: Settings,
        exchange: CachedExchange | KrakenExchange,
    ) -> None:
        """Initialize the Guardian."""
        self.settings = settings
        self.exchange = exchange

    def check_liquidity(self, symbol: str) -> bool:
        """Check if the market liquidity (spread) is healthy."""
        try:
            structure = self.exchange.get_market_structure(symbol)
            spread = structure["spread_pct"]
            limit = self.settings.max_spread_pct

            if spread > limit:
                logger.warning(
                    "Guardian: High Spread for %s. Current: %.2f%% > Limit: %.2f%%",
                    symbol,
                    spread * 100,
                    limit * 100,
                )
                return False

            logger.info(
                "Guardian: Spread OK for %s (%.3f%%)",
                symbol,
                spread * 100,
            )
            return True  # noqa: TRY300
        except Exception:
            logger.exception(
                "Guardian: Failed to check liquidity for %s",
                symbol,
            )
            return False

    def check_exposure(self, symbol: str) -> bool:
        """Check if we already have an active position or open orders."""
        try:
            if self.exchange.has_open_position(symbol):
                logger.info(
                    "Guardian: Exposure detected. We already own %s.",
                    symbol,
                )
                return True

            open_orders = self.exchange.get_open_orders(symbol)
            if open_orders:
                logger.info(
                    "Guardian: Active orders found for %s. Skipping.",
                    symbol,
                )
                return True

            return False  # noqa: TRY300
        except Exception:
            logger.exception(
                "Guardian: Failed to check exposure for %s",
                symbol,
            )
            return True

    def check_market_conditions(self, ticker: str) -> bool:
        """Check Liquidity and Volatility before engaging Strategy."""
        return self.check_liquidity(ticker)

    def check_sell(self, symbol: str) -> float:
        """Validate a SELL signal. Returns the balance to sell (or 0.0)."""
        base_currency = symbol.split("/")[0]
        balance = self.exchange.get_free_balance(base_currency)

        if balance < 0.0001:  # noqa: PLR2004
            logger.info(
                "Guardian: SELL ignored for %s. No significant balance.",
                symbol,
            )
            return 0.0

        logger.info(
            "Guardian: APPROVED SELL %s: total balance %s",
            symbol,
            balance,
        )
        return balance

    def check_buy(
        self,
        symbol: str,
        price: float,
        trade_signal: TradeSignal,
        available_slots: int,
    ) -> float:
        """Validate a BUY signal with dynamic position sizing."""
        assert price > 0, f"Price must be positive, got {price}"  # noqa: S101

        if trade_signal.action != "BUY":
            return 0.0

        if trade_signal.confidence < MIN_CONFIDENCE:
            logger.info(
                "Guardian: Rejecting %s BUY. Confidence %s < %s",
                symbol,
                trade_signal.confidence,
                MIN_CONFIDENCE,
            )
            return 0.0

        try:
            total_balance = self.exchange.get_free_balance(
                self.settings.quote_currency,
            )
            slots = max(available_slots, 1)
            calculated_size = total_balance / slots
            target_usd = min(
                calculated_size,
                self.settings.max_position_size_usd,
            )

            if ROUND_UP_THRESHOLD <= target_usd < MIN_ORDER_USD:
                logger.info(
                    "Guardian: Rounding up %s to $%s for Kraken.",
                    target_usd,
                    MIN_ORDER_USD,
                )
                target_usd = MIN_ORDER_USD

            logger.info(
                "Guardian: Allocation for %s: $%.2f (Balance: $%.2f / %d slots, Cap: $%s)",
                symbol,
                calculated_size,
                total_balance,
                slots,
                self.settings.max_position_size_usd,
            )
        except Exception:
            logger.exception("Guardian: Failed to calculate position size")
            return 0.0

        if total_balance < target_usd:
            logger.warning(
                "Guardian: Insufficient %s funds for %s. Have %s, need %s",
                self.settings.quote_currency,
                symbol,
                total_balance,
                target_usd,
            )
            return 0.0

        amount = target_usd / price

        if target_usd < MIN_ORDER_USD:
            logger.warning(
                "Guardian: Target value %s too low for Kraken (min ~$10).",
                target_usd,
            )
            return 0.0

        logger.info(
            "Guardian: APPROVED BUY %s: %.6f @ $%s ($%s)",
            symbol,
            amount,
            price,
            target_usd,
        )
        return amount


# ---------------------------------------------------------------------------
# Dynamic risk helper — computes SL/TP prices with ATR or vol multipliers
# ---------------------------------------------------------------------------


def _compute_protection_prices(
    base_price: float,
    settings: Settings,
    data: MarketData,
) -> tuple[float, float]:
    """Compute stop-loss and take-profit prices.

    Uses ATR-based stops when available (more adaptive to market conditions),
    falling back to fixed percentage.

    Returns (stop_loss_price, take_profit_price).
    """
    # Try ATR-based stops first
    if settings.use_atr_stops and data.indicators and data.indicators.atr:
        atr = data.indicators.atr
        stop_loss_price = base_price - (atr * settings.atr_sl_multiplier)
        take_profit_price = base_price + (atr * settings.atr_tp_multiplier)

        # Apply vol multipliers on top of ATR if dynamic risk is enabled
        if settings.dynamic_risk_enabled and data.volatility:
            stop_loss_price = base_price - (
                atr * settings.atr_sl_multiplier * data.volatility.suggested_sl_multiplier
            )
            take_profit_price = base_price + (
                atr * settings.atr_tp_multiplier * data.volatility.suggested_tp_multiplier
            )

        # Clamp: SL must be positive and at most 50% below entry
        stop_loss_price = max(stop_loss_price, base_price * 0.5)
        assert stop_loss_price > 0, f"SL must be positive, got {stop_loss_price}"  # noqa: S101
        assert take_profit_price > base_price  # noqa: S101

        return stop_loss_price, take_profit_price

    # Fallback: fixed percentage
    stop_loss_price = base_price * (1 - settings.stop_loss_pct)
    take_profit_price = base_price * (1 + settings.take_profit_pct)

    if settings.dynamic_risk_enabled and data.volatility:
        stop_loss_multiplier = data.volatility.suggested_sl_multiplier
        take_profit_multiplier = data.volatility.suggested_tp_multiplier
        stop_loss_price = base_price * (1 - settings.stop_loss_pct * stop_loss_multiplier)
        take_profit_price = base_price * (1 + settings.take_profit_pct * take_profit_multiplier)

    return stop_loss_price, take_profit_price


# ---------------------------------------------------------------------------
# Data collection (parallelized)
# ---------------------------------------------------------------------------


def _build_market_data(  # noqa: PLR0913
    ticker: str,
    settings: Settings,
    exchange: CachedExchange,
    oracle: SentimentOracle,
    journal: TradeJournal,
    fear_greed: str,
    polymarket_markets: list[dict],
    tech_analyzer: TechnicalAnalyzer | None = None,
    vol_analyzer: VolatilityAnalyzer | None = None,
    price_forecaster: PriceForecaster | None = None,
    has_position: bool = False,  # noqa: FBT001, FBT002
    buy_price: float | None = None,
) -> MarketData:
    """Collect all data for a ticker and return a MarketData object."""
    price, vwap, sma_7d, sma_deviation_pct, vwap_deviation_pct = _fetch_price_context(
        exchange,
        ticker,
    )

    indicators = _fetch_indicators(
        tech_analyzer,
        exchange,
        settings,
        ticker,
        price,
        vwap,
    )
    volatility = _fetch_volatility(
        vol_analyzer,
        exchange,
        settings,
        ticker,
    )
    forecast = _fetch_forecast(
        price_forecaster,
        exchange,
        settings,
        ticker,
    )

    # Parallelize Grok calls (sentiment + consensus) with ThreadPoolExecutor
    sentiment = ""
    consensus = ""
    with ThreadPoolExecutor(max_workers=2) as executor:
        sent_future = executor.submit(oracle.analyze_ticker_sentiment, ticker)
        cons_future = executor.submit(oracle.get_consensus_odds, ticker)
        try:
            sentiment = sent_future.result(timeout=30)
        except Exception:  # noqa: BLE001
            logger.warning("Sentiment fetch timed out for %s", ticker)
            sentiment = "NEUTRAL (Score: 5/10): Timeout."
        try:
            consensus = cons_future.result(timeout=30)
        except Exception:  # noqa: BLE001
            logger.warning("Consensus fetch timed out for %s", ticker)
            consensus = "Consensus unavailable."

    if settings.polymarket_enabled and polymarket_markets:
        polymarket_odds = oracle.get_polymarket_odds_from_cache(
            ticker,
            polymarket_markets,
        )
    else:
        polymarket_odds = "Polymarket: No data available."

    pnl_pct, tp_distance_pct = _compute_position_context(
        price,
        buy_price,
        settings.take_profit_pct,
        has_position,
    )

    # Compute position age
    position_age_hours = None
    if has_position:
        open_time = journal.get_position_open_time(ticker)
        if open_time:
            now = datetime.datetime.now(tz=datetime.timezone.utc)
            delta = now - open_time
            position_age_hours = delta.total_seconds() / 3600

    history = journal.get_recent_history(ticker=ticker)

    # Classify market regime
    regime = None
    if indicators:
        classifier = RegimeClassifier()
        regime = classifier.classify(
            adx=indicators.adx,
            ema_bullish_cross=indicators.ema_bullish_cross,
            roc=indicators.roc,
            bb_pct_b=indicators.bb_pct_b,
            obv_slope=indicators.obv_slope,
        )

    # Multi-timeframe context
    multi_tf = None
    if settings.mtf_enabled and tech_analyzer:
        trend_ctx, short_tf = _collect_multi_timeframe_data(
            ticker,
            tech_analyzer,
            exchange,
            settings,
        )
        if trend_ctx:
            multi_tf = trend_ctx
        if short_tf and trend_ctx:
            trend_ctx.short_tf_rsi = short_tf.get("rsi")
            trend_ctx.short_tf_bb_pct_b = short_tf.get("bb_pct_b")

    data = MarketData(
        ticker=ticker,
        current_price=price,
        vwap=vwap,
        sma_7d=sma_7d,
        sma_deviation_pct=sma_deviation_pct,
        vwap_deviation_pct=vwap_deviation_pct,
        indicators=indicators,
        volatility=volatility,
        forecast=forecast,
        sentiment=sentiment,
        consensus=consensus,
        fear_greed=fear_greed,
        polymarket_odds=polymarket_odds,
        has_position=has_position,
        entry_price=buy_price,
        pnl_pct=pnl_pct,
        tp_distance_pct=tp_distance_pct,
        position_age_hours=position_age_hours,
        history=history,
        regime=regime,
        multi_tf=multi_tf,
    )

    # Persist market snapshot for backtest replay
    indicators_dict = {}
    if data.indicators:
        indicators_dict = {
            "rsi": data.indicators.rsi,
            "macd_histogram": data.indicators.macd_histogram,
            "bb_pct_b": data.indicators.bb_pct_b,
            "adx": data.indicators.adx,
            "roc": data.indicators.roc,
            "atr": data.indicators.atr,
        }
    journal.save_market_snapshot(
        ticker=ticker,
        price=data.current_price,
        sentiment=data.sentiment[:500],
        consensus=data.consensus[:500],
        fear_greed=data.fear_greed,
        polymarket_odds=data.polymarket_odds[:500],
        indicators_json=json.dumps(indicators_dict),
        regime=data.regime.value if data.regime else "RANGING",
    )

    return data


def _fetch_price_context(
    exchange: CachedExchange,
    ticker: str,
) -> tuple[float, float | None, float | None, float | None, float | None]:
    """Fetch price, VWAP, SMA and deviation data.

    Returns (price, vwap, sma_7d, sma_deviation_pct, vwap_deviation_pct).
    """
    vwap = None
    sma_7d = None
    sma_deviation_pct = None
    vwap_deviation_pct = None

    try:
        avg_data = exchange.fetch_multi_day_average(ticker)
        price = avg_data["price"]
        vwap = avg_data["vwap"]
        sma_7d = avg_data["sma"]
        sma_deviation_pct = avg_data["sma_deviation_pct"]
        vwap_deviation_pct = avg_data["vwap_deviation_pct"]
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to fetch price analysis for %s: %s",
            ticker,
            exc,
        )
        price = exchange.fetch_price(ticker)

    return price, vwap, sma_7d, sma_deviation_pct, vwap_deviation_pct


def _fetch_indicators(  # noqa: PLR0913
    tech_analyzer: TechnicalAnalyzer | None,
    exchange: CachedExchange,
    settings: Settings,
    ticker: str,
    price: float,
    vwap: float | None,
) -> object | None:
    """Fetch technical indicators, or None on failure."""
    if not tech_analyzer:
        return None
    try:
        candles = exchange.fetch_ohlcv(
            ticker,
            settings.ta_candle_timeframe,
            settings.ta_candle_limit,
        )
        return tech_analyzer.compute(
            candles,
            current_price=price,
            vwap=vwap,
        )
    except Exception:  # noqa: BLE001
        logger.warning(
            "Technical analysis failed for %s.",
            ticker,
            exc_info=True,
        )
        return None


def _fetch_volatility(
    vol_analyzer: VolatilityAnalyzer | None,
    exchange: CachedExchange,
    settings: Settings,
    ticker: str,
) -> object | None:
    """Fetch GARCH volatility forecast, or None on failure."""
    if not vol_analyzer or not settings.garch_enabled:
        return None
    try:
        candles_4h = exchange.fetch_ohlcv(
            ticker,
            settings.garch_candle_timeframe,
            settings.garch_candle_limit,
        )
        return vol_analyzer.fit_garch(candles_4h)
    except Exception:  # noqa: BLE001
        logger.warning(
            "Volatility analysis failed for %s.",
            ticker,
            exc_info=True,
        )
        return None


def _fetch_forecast(
    price_forecaster: PriceForecaster | None,
    exchange: CachedExchange,
    settings: Settings,
    ticker: str,
) -> object | None:
    """Fetch ARIMA price forecast, or None on failure."""
    if not price_forecaster or not settings.arima_enabled:
        return None
    try:
        candles_1h = exchange.fetch_ohlcv(
            ticker,
            settings.ta_candle_timeframe,
            settings.ta_candle_limit,
        )
        return price_forecaster.forecast(candles_1h)
    except Exception:  # noqa: BLE001
        logger.warning(
            "Price forecast failed for %s.",
            ticker,
            exc_info=True,
        )
        return None


def _compute_position_context(
    price: float,
    buy_price: float | None,
    take_profit_pct: float,
    has_position: bool,  # noqa: FBT001
) -> tuple[float | None, float | None]:
    """Compute PnL and TP distance for an owned position.

    Returns (pnl_pct, tp_distance_pct).
    """
    if not has_position or not buy_price:
        return None, None

    assert buy_price > 0, f"Buy price must be positive, got {buy_price}"  # noqa: S101

    pnl_pct = ((price - buy_price) / buy_price) * 100
    take_profit_price = buy_price * (1 + take_profit_pct)
    tp_distance_pct = ((take_profit_price - price) / price) * 100
    return pnl_pct, tp_distance_pct


def _collect_multi_timeframe_data(
    ticker: str,
    analyzer: TechnicalAnalyzer,
    exchange: CachedExchange,
    settings: Settings,
) -> tuple:
    """Collect short and long timeframe data for multi-timeframe analysis.

    Returns (trend_context, short_tf_indicators).
    """
    if not settings.mtf_enabled:
        return None, None

    trend_context = None
    short_tf = None

    try:
        # 4h candles for trend context
        long_candles = exchange.fetch_ohlcv(
            ticker,
            settings.mtf_long_timeframe,
            100,
        )
        if long_candles:
            trend_context = analyzer.compute_trend_context(long_candles)
    except Exception:  # noqa: BLE001
        logger.debug("Failed to fetch %s candles for %s", settings.mtf_long_timeframe, ticker)

    try:
        # 15m candles for entry timing
        short_candles = exchange.fetch_ohlcv(
            ticker,
            settings.mtf_short_timeframe,
            50,
        )
        if short_candles:
            short_tf = analyzer.compute_short_timeframe(short_candles)
    except Exception:  # noqa: BLE001
        logger.debug("Failed to fetch %s candles for %s", settings.mtf_short_timeframe, ticker)

    return trend_context, short_tf


# ---------------------------------------------------------------------------
# Correlation check
# ---------------------------------------------------------------------------


def _check_correlation_limit(
    ticker: str,
    owned_tickers: list[str],
    settings: Settings,
) -> bool:
    """Check if adding this ticker would exceed the correlated positions limit.

    Returns True if the entry is allowed, False if it would exceed the limit.
    """
    base = ticker.split("/")[0]
    new_group = _CORRELATION_GROUPS.get(base, base)

    # Count how many owned positions are in the same correlation group
    same_group_count = 0
    for owned in owned_tickers:
        owned_base = owned.split("/")[0]
        owned_group = _CORRELATION_GROUPS.get(owned_base, owned_base)
        if owned_group == new_group:
            same_group_count += 1

    if same_group_count >= settings.max_correlated_positions:
        logger.info(
            "Guardian: Correlation limit reached for %s (group '%s', %d/%d positions). Skipping.",
            ticker,
            new_group,
            same_group_count,
            settings.max_correlated_positions,
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Sell execution
# ---------------------------------------------------------------------------


def _execute_sell(  # noqa: PLR0913
    ticker: str,
    price: float,
    buy_price: float | None,
    reason: str,
    settings: Settings,
    exchange: CachedExchange,
    journal: TradeJournal,
    notifier: WebhookNotifier | NullNotifier,
    paper_portfolio: PaperPortfolio | None = None,
) -> None:
    """Execute a sell order (shared by TP and AI-driven exits)."""
    assert price > 0, f"Sell price must be positive, got {price}"  # noqa: S101

    if settings.trading_mode == "PAPER" and paper_portfolio:
        amount_to_sell = paper_portfolio.get_free_balance(
            ticker.split("/")[0],
        )
    else:
        amount_to_sell = exchange.get_free_balance(ticker.split("/")[0])

    if amount_to_sell < 0.0001:  # noqa: PLR2004
        logger.info("Sell skipped for %s: no significant balance.", ticker)
        return

    profit = 0.0
    tax = 0.0
    if buy_price:
        estimated_sell_fee = price * amount_to_sell * 0.0025  # 0.25% maker fee
        profit, tax = journal.compute_pnl_with_fees(
            ticker,
            price,
            amount_to_sell,
            estimated_sell_fee,
        )
        if profit > 0:
            logger.info(
                "Tax: Realized profit of %.2f %s (fee-adjusted). Tax liability: %.2f",
                profit,
                settings.quote_currency,
                tax,
            )

    if settings.trading_mode == "LIVE":
        logger.info(">>> EXECUTING LIVE SELL for %s <<<", ticker)
        exchange.cancel_all_orders(ticker)
        exchange.place_maker_sell(ticker, price, amount_to_sell)
    else:
        logger.info(">>> PAPER: SELL %s @ $%s <<<", ticker, price)
        if paper_portfolio:
            paper_portfolio.record_sell(ticker, amount_to_sell)

    journal.log_trade(
        account="Kraken",
        instrument=ticker,
        side="SELL",
        price=price,
        quantity=amount_to_sell,
        value=price * amount_to_sell,
        currency=settings.quote_currency,
        reasoning=reason,
        profit=profit,
        tax=tax,
        fee=price * amount_to_sell * 0.0025,
    )
    notifier.notify_trade(ticker, "SELL", price, amount_to_sell, reason)


def _execute_partial_sell(  # noqa: PLR0913
    ticker: str,
    price: float,
    sell_pct: float,
    reasoning: str,
    settings: Settings,
    exchange: CachedExchange,
    journal: TradeJournal,
    notifier: WebhookNotifier | NullNotifier,
    paper_portfolio: PaperPortfolio | None,
) -> bool:
    """Execute a partial position sell. Returns True if the order succeeded."""
    base_currency = ticker.split("/")[0]

    if settings.trading_mode == "PAPER" and paper_portfolio:
        total_balance = paper_portfolio.get_free_balance(base_currency)
    else:
        total_balance = exchange.get_free_balance(base_currency)

    sell_amount = total_balance * sell_pct

    if sell_amount < 0.0001:  # noqa: PLR2004
        return False

    sell_price = price * (1 + settings.limit_order_discount_pct)  # Sell slightly above market

    if settings.trading_mode == "LIVE":
        logger.info(">>> EXECUTING LIVE PARTIAL SELL %s: %.1f%% <<<", ticker, sell_pct * 100)
        try:
            exchange.place_maker_sell(ticker, sell_price, sell_amount)
        except Exception:  # noqa: BLE001
            logger.warning("Partial sell failed for %s", ticker, exc_info=True)
            return False
    else:
        logger.info(
            ">>> PAPER PARTIAL SELL %s: %.1f%% @ $%s <<<",
            ticker,
            sell_pct * 100,
            sell_price,
        )
        if paper_portfolio:
            paper_portfolio.record_sell(ticker, sell_amount)

    journal.log_trade(
        account="Kraken",
        instrument=ticker,
        side="SELL",
        price=sell_price,
        quantity=sell_amount,
        value=sell_price * sell_amount,
        currency=settings.quote_currency,
        reasoning=reasoning,
    )
    notifier.notify_trade(ticker, "PARTIAL SELL", sell_price, sell_amount, reasoning)
    return True


# ---------------------------------------------------------------------------
# Exit evaluation (extracted helpers keep each function under 70 lines)
# ---------------------------------------------------------------------------


def _check_take_profit(  # noqa: PLR0913
    ticker: str,
    data: MarketData,
    settings: Settings,
    exchange: CachedExchange,
    journal: TradeJournal,
    notifier: WebhookNotifier | NullNotifier,
    paper_portfolio: PaperPortfolio | None,
) -> bool:
    """Check hard take-profit and execute sell if hit. Returns True if sold."""
    buy_price = data.entry_price
    if not buy_price:
        return False

    _stop_loss_price, take_profit_price = _compute_protection_prices(
        buy_price,
        settings,
        data,
    )

    if data.current_price >= take_profit_price:
        logger.info(
            "Take-profit hit for %s: price $%.8f >= TP $%.8f",
            ticker,
            data.current_price,
            take_profit_price,
        )
        _execute_sell(
            ticker,
            data.current_price,
            buy_price,
            f"Take-profit triggered: price ${data.current_price:.8f} "
            f">= TP ${take_profit_price:.8f}",
            settings,
            exchange,
            journal,
            notifier,
            paper_portfolio,
        )
        journal.delete_position_state(ticker)
        return True

    return False


def _ensure_stop_loss(
    ticker: str,
    data: MarketData,
    settings: Settings,
    exchange: CachedExchange,
) -> None:
    """Place a stop-loss order if none exists (LIVE mode only)."""
    buy_price = data.entry_price
    if not buy_price or settings.trading_mode != "LIVE":
        return

    open_orders = exchange.get_open_orders(ticker)
    has_stop_loss = any(o.get("type") in ("stop-loss", "stop-loss-limit") for o in open_orders)
    if has_stop_loss:
        return

    stop_loss_price, take_profit_price = _compute_protection_prices(
        buy_price,
        settings,
        data,
    )

    base_currency = ticker.split("/")[0]
    balance = exchange.get_free_balance(base_currency)
    if balance > 0.0001:  # noqa: PLR2004
        logger.info(
            "Placing deferred SL for %s: %.8f @ %.8f",
            ticker,
            balance,
            stop_loss_price,
        )
        try:
            exchange.place_protection_orders(
                ticker,
                balance,
                buy_price,
                stop_loss_price,
                take_profit_price,
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "Failed to place deferred SL for %s.",
                ticker,
            )


def _execute_dca(  # noqa: PLR0913
    ticker: str,
    data: MarketData,
    trade_signal: TradeSignal,
    settings: Settings,
    exchange: CachedExchange,
    guardian: RiskGuardian,
    journal: TradeJournal,
    notifier: WebhookNotifier | NullNotifier,
    available_slots: int,
    paper_portfolio: PaperPortfolio | None,
) -> None:
    """Execute a DCA buy if the strategy signals BUY on an owned position."""
    if trade_signal.action != "BUY" or trade_signal.confidence < MIN_CONFIDENCE:
        return

    active_buys = journal.get_trade_count(ticker, "BUY")
    max_allowed = 1 + settings.max_dca_per_position
    if active_buys >= max_allowed:
        logger.info(
            "DCA limit reached for %s: %d active buys (max %d).",
            ticker,
            active_buys,
            max_allowed,
        )
        return

    open_orders = exchange.get_open_orders(ticker)
    if open_orders:
        logger.info(
            "DCA: Skipping %s - %d open orders remain.",
            ticker,
            len(open_orders),
        )
        return

    price = data.current_price
    limit_price = price * (1 - settings.limit_order_discount_pct)
    amount = guardian.check_buy(
        ticker,
        limit_price,
        trade_signal,
        available_slots,
    )
    if amount <= 0:
        return

    if settings.trading_mode == "LIVE":
        logger.info(">>> EXECUTING LIVE DCA BUY for %s <<<", ticker)
        exchange.place_maker_buy(ticker, limit_price, amount)

        stop_loss_price, take_profit_price = _compute_protection_prices(
            limit_price,
            settings,
            data,
        )
        try:
            exchange.place_protection_orders(
                ticker,
                amount,
                limit_price,
                stop_loss_price,
                take_profit_price,
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "Protection orders deferred for %s DCA.",
                ticker,
            )
    else:
        logger.info(
            ">>> PAPER DCA: BUY %s @ $%s <<<",
            ticker,
            limit_price,
        )
        if paper_portfolio:
            paper_portfolio.record_buy(ticker, limit_price, amount)

    journal.log_trade(
        account="Kraken",
        instrument=ticker,
        side="BUY",
        price=limit_price,
        quantity=amount,
        value=limit_price * amount,
        currency=settings.quote_currency,
        reasoning=f"DCA: {trade_signal.reasoning}",
    )
    notifier.notify_trade(
        ticker,
        "DCA BUY",
        limit_price,
        amount,
        trade_signal.reasoning,
    )

    # Recalculate SL based on average cost basis after DCA.
    avg_cost = journal.get_average_cost_basis(ticker)
    if avg_cost and settings.trading_mode == "LIVE":
        new_stop_loss, _new_tp = _compute_protection_prices(
            avg_cost,
            settings,
            data,
        )
        logger.info(
            "DCA SL recalc for %s: avg cost $%.8f -> SL $%.8f",
            ticker,
            avg_cost,
            new_stop_loss,
        )

    # Update position tracking state for DCA
    from models import PositionState, StopMode

    dca_stop_loss_price, dca_take_profit_price = _compute_protection_prices(
        limit_price,
        settings,
        data,
    )

    regime_classifier = RegimeClassifier()
    regime = (
        regime_classifier.classify(
            adx=data.indicators.adx if data.indicators else None,
            ema_bullish_cross=data.indicators.ema_bullish_cross if data.indicators else None,
            roc=data.indicators.roc if data.indicators else None,
            bb_pct_b=data.indicators.bb_pct_b if data.indicators else None,
            obv_slope=data.indicators.obv_slope if data.indicators else None,
        )
        if data.indicators
        else None
    )

    stop_mode_str = (
        RegimeClassifier.get_stop_mode_for_regime(regime) if regime else StopMode.FIXED.value
    )
    stop_mode = StopMode(stop_mode_str)

    # Use average cost basis if available for position state
    entry_for_state = avg_cost if avg_cost else limit_price

    existing_state = journal.get_position_state(ticker)
    pos_state = PositionState(
        ticker=ticker,
        entry_price=entry_for_state,
        high_watermark=max(
            entry_for_state,
            existing_state.high_watermark if existing_state else 0.0,
        ),
        stop_mode=stop_mode,
        current_sl=dca_stop_loss_price,
        current_tp=dca_take_profit_price,
        partial_exits=existing_state.partial_exits if existing_state else 0,
        regime_at_entry=regime.value if regime else "RANGING",
    )
    journal.upsert_position_state(pos_state)


def _evaluate_exit(  # noqa: PLR0913
    ticker: str,
    data: MarketData,
    settings: Settings,
    exchange: CachedExchange,
    strategist: Strategist,
    guardian: RiskGuardian,
    journal: TradeJournal,
    notifier: WebhookNotifier | NullNotifier,
    available_slots: int,
    paper_portfolio: PaperPortfolio | None = None,
) -> None:
    """Evaluate whether to exit an owned position, or DCA into it."""
    logger.info("Exit evaluation: %s", ticker)

    # Stop manager: trailing SL, ROI exit, partial exits
    stop_manager = StopManager(settings)
    pos_state = journal.get_position_state(ticker)

    if pos_state and data.current_price > 0:
        # Update high watermark
        journal.update_high_watermark(ticker, data.current_price)
        pos_state.high_watermark = max(pos_state.high_watermark, data.current_price)

        # Get ATR for stop calculations
        atr = data.indicators.atr if data.indicators else None
        pnl_pct = data.pnl_pct or 0.0
        age_minutes = (data.position_age_hours or 0.0) * 60.0

        stop_action = stop_manager.adjust(
            pos_state,
            data.current_price,
            atr,
            pnl_pct,
            age_minutes,
        )

        if stop_action.action == "SELL":
            logger.info("StopManager SELL for %s: %s", ticker, stop_action.reason)
            _execute_sell(
                ticker,
                data.current_price,
                data.entry_price,
                f"StopManager: {stop_action.reason}",
                settings,
                exchange,
                journal,
                notifier,
                paper_portfolio,
            )
            journal.delete_position_state(ticker)
            return

        if stop_action.action == "PARTIAL_SELL":
            logger.info("StopManager PARTIAL SELL for %s: %s", ticker, stop_action.reason)
            success = _execute_partial_sell(
                ticker,
                data.current_price,
                stop_action.sell_pct,
                f"Partial: {stop_action.reason}",
                settings,
                exchange,
                journal,
                notifier,
                paper_portfolio,
            )
            if success:
                pos_state.partial_exits += 1
                journal.upsert_position_state(pos_state)

        # Update position state with new SL
        if stop_action.updated_sl != pos_state.current_sl:
            pos_state.current_sl = stop_action.updated_sl
            journal.upsert_position_state(pos_state)

    # Hard TP check (before strategy).
    if _check_take_profit(
        ticker,
        data,
        settings,
        exchange,
        journal,
        notifier,
        paper_portfolio,
    ):
        return

    # Ensure stop-loss exists (LIVE mode).
    _ensure_stop_loss(ticker, data, settings, exchange)

    # Strategy evaluation.
    trade_signal = strategist.generate_signal(data)
    logger.info(
        "Exit signal for %s: %s (Conf: %s) - %s",
        ticker,
        trade_signal.action,
        trade_signal.confidence,
        trade_signal.reasoning,
    )

    # Log structured signal to SQLite
    vol_regime = data.volatility.regime.value if data.volatility else "NORMAL"
    journal.log_signal(
        ticker,
        trade_signal.action,
        trade_signal.confidence,
        trade_signal.composite,
        vol_regime,
        trade_signal.reasoning,
    )

    # Strategy says SELL.
    if trade_signal.action == "SELL" and trade_signal.confidence >= MIN_CONFIDENCE:
        _execute_sell(
            ticker,
            data.current_price,
            data.entry_price,
            f"Strategic Exit: {trade_signal.reasoning}",
            settings,
            exchange,
            journal,
            notifier,
            paper_portfolio,
        )
        journal.delete_position_state(ticker)
        return

    # DCA path: strategy says BUY on owned position.
    _execute_dca(
        ticker,
        data,
        trade_signal,
        settings,
        exchange,
        guardian,
        journal,
        notifier,
        available_slots,
        paper_portfolio,
    )


# ---------------------------------------------------------------------------
# Entry evaluation
# ---------------------------------------------------------------------------


def _evaluate_entry(  # noqa: PLR0913
    ticker: str,
    data: MarketData,
    settings: Settings,
    exchange: CachedExchange,
    strategist: Strategist,
    guardian: RiskGuardian,
    journal: TradeJournal,
    notifier: WebhookNotifier | NullNotifier,
    available_slots: int,
    owned_tickers: list[str],
    paper_portfolio: PaperPortfolio | None = None,
) -> bool:
    """Evaluate whether to enter a new position.

    Returns True if a buy order was placed.
    """
    logger.info("Entry evaluation: %s", ticker)

    open_orders = exchange.get_open_orders(ticker)
    if open_orders:
        logger.info(
            "Entry: Skipping %s - %d open orders remain.",
            ticker,
            len(open_orders),
        )
        return False

    if not guardian.check_market_conditions(ticker):
        return False

    # Correlation limit check
    if not _check_correlation_limit(ticker, owned_tickers, settings):
        return False

    price = data.current_price

    # Circuit breaker: abort if regulatory danger phrases are detected.
    combined = (data.sentiment + " " + data.consensus).upper()
    if any(phrase in combined for phrase in REGULATORY_DANGER_PHRASES):
        logger.warning(
            "Entry: Regulatory Risk detected for %s. Abort.",
            ticker,
        )
        return False

    logger.info(
        "Data for %s: Price=%s, Sentiment=%s..., Consensus=%s...",
        ticker,
        fmt_price(price),
        data.sentiment[:30],
        data.consensus[:30],
    )
    logger.info("Polymarket for %s: %s", ticker, data.polymarket_odds)

    trade_signal = strategist.generate_signal(data)
    logger.info(
        "Signal: %s (Conf: %s) - %s",
        trade_signal.action,
        trade_signal.confidence,
        trade_signal.reasoning,
    )

    # Log structured signal to SQLite
    vol_regime = data.volatility.regime.value if data.volatility else "NORMAL"
    journal.log_signal(
        ticker,
        trade_signal.action,
        trade_signal.confidence,
        trade_signal.composite,
        vol_regime,
        trade_signal.reasoning,
    )

    if trade_signal.action != "BUY":
        return False

    limit_price = price * (1 - settings.limit_order_discount_pct)
    amount = guardian.check_buy(
        ticker,
        limit_price,
        trade_signal,
        available_slots,
    )
    if amount <= 0:
        return False

    stop_loss_price, take_profit_price = _compute_protection_prices(
        limit_price,
        settings,
        data,
    )

    if settings.trading_mode == "LIVE":
        logger.info(">>> EXECUTING LIVE SNIPER BUY for %s <<<", ticker)
        exchange.place_maker_buy(ticker, limit_price, amount)

        try:
            exchange.place_protection_orders(
                ticker,
                amount,
                limit_price,
                stop_loss_price,
                take_profit_price,
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "Protection orders deferred for %s — buy not yet filled. "
                "SL will be placed on next cycle when position is detected.",
                ticker,
            )
    else:
        logger.info(
            ">>> PAPER SNIPER: BUY %s @ $%s <<<",
            ticker,
            limit_price,
        )
        if paper_portfolio:
            paper_portfolio.record_buy(ticker, limit_price, amount)

    journal.log_trade(
        account="Kraken",
        instrument=ticker,
        side="BUY",
        price=limit_price,
        quantity=amount,
        value=limit_price * amount,
        currency=settings.quote_currency,
        reasoning=trade_signal.reasoning,
    )
    notifier.notify_trade(
        ticker,
        "BUY",
        limit_price,
        amount,
        trade_signal.reasoning,
    )

    # Create position tracking state
    from models import PositionState, StopMode

    regime_classifier = RegimeClassifier()
    regime = (
        regime_classifier.classify(
            adx=data.indicators.adx if data.indicators else None,
            ema_bullish_cross=data.indicators.ema_bullish_cross if data.indicators else None,
            roc=data.indicators.roc if data.indicators else None,
            bb_pct_b=data.indicators.bb_pct_b if data.indicators else None,
            obv_slope=data.indicators.obv_slope if data.indicators else None,
        )
        if data.indicators
        else None
    )

    stop_mode_str = (
        RegimeClassifier.get_stop_mode_for_regime(regime) if regime else StopMode.FIXED.value
    )
    stop_mode = StopMode(stop_mode_str)

    pos_state = PositionState(
        ticker=ticker,
        entry_price=limit_price,
        high_watermark=limit_price,
        stop_mode=stop_mode,
        current_sl=stop_loss_price,
        current_tp=take_profit_price,
        regime_at_entry=regime.value if regime else "RANGING",
    )
    journal.upsert_position_state(pos_state)

    return True


# ---------------------------------------------------------------------------
# Cycle helpers (extracted to keep run_cycle under 70 lines)
# ---------------------------------------------------------------------------


def _reconcile_stale_orders(
    settings: Settings,
    exchange: KrakenExchange,
    journal: TradeJournal,
) -> None:
    """Cancel stale orders and log cancellations to the journal."""
    for ticker in settings.universe_list:
        if not exchange.symbol_exists(ticker):
            continue
        try:
            cancelled = exchange.cancel_stale_orders(
                ticker,
                settings.stale_order_max_age_seconds,
            )
            if cancelled:
                logger.info(
                    "Reconciliation: Cancelled %d stale orders for %s",
                    len(cancelled),
                    ticker,
                )
                for order in cancelled:
                    if order["side"] == "buy":
                        journal.log_trade(
                            account="Kraken",
                            instrument=ticker,
                            side="CANCEL",
                            price=order["price"],
                            quantity=order["amount"],
                            value=order["price"] * order["amount"],
                            currency=settings.quote_currency,
                            reasoning="Stale limit order cancelled.",
                        )
                        # Clean up orphaned position state from unfilled buys
                        journal.delete_position_state(ticker)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Reconciliation: Error for %s: %s",
                ticker,
                exc,
            )


def _reconcile_fills(
    tickers: list[str],
    exchange: KrakenExchange,
    journal: TradeJournal,
    notifier: WebhookNotifier | NullNotifier,
    quote_currency: str,
    tax_rate: float,
) -> None:
    """Sync Kraken fills with our journal to catch SL/TP executions."""
    for ticker in tickers:
        try:
            since_ms = journal.get_last_sync_timestamp(ticker)
            if since_ms is not None:
                since_ms = max(since_ms - 60_000, 0)

            fills = exchange.fetch_my_trades(ticker, since_ms=since_ms)
            for fill in fills:
                fill_id = fill.get("id", "")
                if not fill_id or journal.is_fill_recorded(fill_id):
                    continue

                side = fill.get("side", "").upper()
                fill_price = float(fill.get("price", 0))
                fill_qty = float(fill.get("amount", 0))
                fill_cost = float(fill.get("cost", 0))
                fee_info = fill.get("fee", {})
                fill_fee = float(fee_info.get("cost", 0)) if fee_info else 0.0

                if side not in ("BUY", "SELL") or fill_price <= 0:
                    continue

                profit = 0.0
                tax = 0.0
                if side == "SELL":
                    profit, tax = journal.compute_pnl_with_fees(
                        ticker,
                        fill_price,
                        fill_qty,
                        fill_fee,
                    )

                reasoning = (
                    f"Exchange fill: {side} detected via reconciliation. "
                    f"Price={fill_price}, Fee={fill_fee:.6f} [{fill_id}]"
                )

                journal.log_trade(
                    account="Kraken",
                    instrument=ticker,
                    side=side,
                    price=fill_price,
                    quantity=fill_qty,
                    value=fill_cost,
                    currency=quote_currency,
                    reasoning=reasoning,
                    profit=profit,
                    tax=tax,
                    fee=fill_fee,
                    fill_source="exchange",
                )

                logger.info(
                    "Reconciliation: Recorded %s fill for %s: %.8f @ %s (fee: %.6f, P&L: %.4f) [%s]",
                    side,
                    ticker,
                    fill_qty,
                    fmt_price(fill_price),
                    fill_fee,
                    profit,
                    fill_id,
                )

                if side == "SELL":
                    journal.delete_position_state(ticker)
                    notifier.notify_trade(
                        ticker,
                        "SELL",
                        fill_price,
                        fill_qty,
                        f"Exchange SL/TP fill detected [{fill_id}]",
                    )

        except Exception as exc:  # noqa: BLE001
            logger.warning("Fill reconciliation error for %s: %s", ticker, exc)


def _classify_tickers(
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


def _check_daily_loss_limit(journal: TradeJournal, settings: Settings) -> bool:
    """Check if daily loss limit has been exceeded. Returns True if trading is OK."""
    daily_pnl = journal.get_daily_realized_pnl()
    if daily_pnl < -settings.daily_loss_limit_usd:
        logger.warning(
            "DAILY LOSS LIMIT HIT: $%.2f (limit: -$%.2f). Pausing new entries for today.",
            daily_pnl,
            settings.daily_loss_limit_usd,
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Parallel data collection
# ---------------------------------------------------------------------------


def _collect_all_market_data(  # noqa: PLR0913
    tickers: list[str],
    settings: Settings,
    cached_exchange: CachedExchange,
    oracle: SentimentOracle,
    journal: TradeJournal,
    fear_greed: str,
    polymarket_markets: list[dict],
    tech_analyzer: TechnicalAnalyzer | None,
    vol_analyzer: VolatilityAnalyzer | None,
    price_forecaster: PriceForecaster | None,
    has_position_map: dict[str, bool],
    buy_price_map: dict[str, float | None],
) -> dict[str, MarketData]:
    """Collect market data for all tickers in parallel.

    Returns a dict mapping ticker -> MarketData.
    """
    results: dict[str, MarketData] = {}

    with ThreadPoolExecutor(max_workers=min(len(tickers), 4)) as executor:
        future_to_ticker = {}
        for ticker in tickers:
            future = executor.submit(
                _build_market_data,
                ticker,
                settings,
                cached_exchange,
                oracle,
                journal,
                fear_greed,
                polymarket_markets,
                tech_analyzer,
                vol_analyzer,
                price_forecaster,
                has_position=has_position_map.get(ticker, False),
                buy_price=buy_price_map.get(ticker),
            )
            future_to_ticker[future] = ticker

        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                results[ticker] = future.result()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Data collection failed for %s: %s",
                    ticker,
                    exc,
                )

    return results


# ---------------------------------------------------------------------------
# Main cycle
# ---------------------------------------------------------------------------


def run_cycle(  # noqa: PLR0913
    settings: Settings,
    exchange: KrakenExchange,
    oracle: SentimentOracle,
    strategist: Strategist,
    guardian: RiskGuardian,
    journal: TradeJournal,
    notifier: WebhookNotifier | NullNotifier,
    tech_analyzer: TechnicalAnalyzer | None = None,
    vol_analyzer: VolatilityAnalyzer | None = None,
    price_forecaster: PriceForecaster | None = None,
    portfolio_analyzer: PortfolioAnalyzer | None = None,
    paper_portfolio: PaperPortfolio | None = None,
) -> None:
    """Execute one complete trading analysis cycle (2-pass system)."""
    cycle_start = time.monotonic()

    # Determine if this is a full analysis cycle or fast entry/stop cycle
    is_full_cycle = not hasattr(run_cycle, "_last_full") or (
        time.time() - run_cycle._last_full >= settings.full_cycle_seconds
    )
    if is_full_cycle:
        run_cycle._last_full = time.time()

    logger.info(
        "Starting %s Cycle. Mode: %s",
        "FULL" if is_full_cycle else "FAST",
        settings.trading_mode,
    )

    cached_exchange = CachedExchange(exchange)
    guardian.exchange = cached_exchange

    _reconcile_stale_orders(settings, exchange, journal)

    # Sync Kraken fills to catch SL/TP triggers we missed
    if settings.trading_mode == "LIVE":
        _reconcile_fills(
            tickers=settings.universe_list,
            exchange=exchange,
            journal=journal,
            notifier=notifier,
            quote_currency=settings.quote_currency,
            tax_rate=settings.tax_rate,
        )

    # Heavy operations only on full cycles; reuse cached values on fast cycles
    if is_full_cycle:
        fear_greed = oracle.get_fear_and_greed_index()
        run_cycle._cached_fear_greed = fear_greed
    else:
        fear_greed = getattr(run_cycle, "_cached_fear_greed", "")
    logger.info("Market Sentiment: Fear & Greed Index = %s", fear_greed)

    polymarket_markets: list[dict] = []
    if is_full_cycle and settings.polymarket_enabled:
        polymarket_markets = oracle.fetch_polymarket_markets()
        run_cycle._cached_polymarket = polymarket_markets
        logger.info(
            "Polymarket: fetched %d markets for cycle",
            len(polymarket_markets),
        )
    elif not is_full_cycle:
        polymarket_markets = getattr(run_cycle, "_cached_polymarket", [])

    owned_tickers, available_tickers = _classify_tickers(
        settings,
        exchange,
        cached_exchange,
        paper_portfolio,
    )
    logger.info(
        "Universe: %d owned %s, %d available %s",
        len(owned_tickers),
        owned_tickers,
        len(available_tickers),
        available_tickers,
    )

    available_slots = len(available_tickers) if available_tickers else 1

    # Daily loss limit check (only blocks new entries, not exits)
    daily_loss_ok = _check_daily_loss_limit(journal, settings)

    # Prepare position context maps
    has_position_map: dict[str, bool] = {}
    buy_price_map: dict[str, float | None] = {}
    for ticker in owned_tickers:
        has_position_map[ticker] = True
        buy_price_map[ticker] = journal.get_average_cost_basis(
            ticker,
        ) or journal.get_last_buy_price(ticker)
    for ticker in available_tickers:
        has_position_map[ticker] = False
        buy_price_map[ticker] = None

    # Parallel data collection for ALL tickers at once
    # On fast cycles, skip heavy GARCH/ARIMA; only run indicators + stop management
    cycle_vol_analyzer = vol_analyzer if is_full_cycle else None
    cycle_price_forecaster = price_forecaster if is_full_cycle else None

    all_tickers = owned_tickers + available_tickers
    all_data = _collect_all_market_data(
        all_tickers,
        settings,
        cached_exchange,
        oracle,
        journal,
        fear_greed,
        polymarket_markets,
        tech_analyzer,
        cycle_vol_analyzer,
        cycle_price_forecaster,
        has_position_map,
        buy_price_map,
    )

    # Pass 1: Exit evaluation + DCA on owned positions.
    for ticker in owned_tickers:
        data = all_data.get(ticker)
        if not data:
            continue
        try:
            _evaluate_exit(
                ticker,
                data,
                settings,
                cached_exchange,
                strategist,
                guardian,
                journal,
                notifier,
                available_slots,
                paper_portfolio,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Error in exit evaluation for %s: %s",
                ticker,
                exc,
            )
            notifier.notify_error(
                f"Exit evaluation failed for {ticker}: {exc}",
            )

    # Pass 2: Entry evaluation on available positions.
    if daily_loss_ok:
        for ticker in available_tickers:
            data = all_data.get(ticker)
            if not data:
                continue
            try:
                bought = _evaluate_entry(
                    ticker,
                    data,
                    settings,
                    cached_exchange,
                    strategist,
                    guardian,
                    journal,
                    notifier,
                    available_slots,
                    owned_tickers,
                    paper_portfolio,
                )
                if bought:
                    available_slots = max(available_slots - 1, 1)
                    owned_tickers.append(ticker)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Error in entry evaluation for %s: %s",
                    ticker,
                    exc,
                )
                notifier.notify_error(
                    f"Entry evaluation failed for {ticker}: {exc}",
                )
    else:
        logger.info("Skipping entries due to daily loss limit.")

    if portfolio_analyzer:
        portfolio_analyzer.log_summary()

    # Write healthcheck
    _write_healthcheck(settings)

    cycle_duration_seconds = time.monotonic() - cycle_start
    logger.info("Cycle Complete. Duration: %.1fs", cycle_duration_seconds)
    notifier.notify_cycle_summary(
        f"Cycle done in {cycle_duration_seconds:.1f}s",
    )


def main() -> None:
    """Entry point for the bot."""
    try:
        settings = Settings()
    except Exception:  # noqa: BLE001
        logging.basicConfig(level=logging.INFO)
        logger.critical("Failed to load initial settings. Exiting.")
        return

    setup_logging(settings)
    logger.info("--- ORACLE PYTHIA BOOTING VERSION 8.0 ---")

    shutdown_event = threading.Event()

    def signal_handler(_signum: int, _frame: object) -> None:
        logger.info("Shutdown signal received. Exiting...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("Oracle Pythia Bot Started.")

    try:
        exchange = KrakenExchange(
            settings.kraken_api_key,
            settings.kraken_private_key,
        )
        oracle = SentimentOracle(
            settings.xai_api_key,
            settings.xai_model,
            settings.cmc_api_key,
            warp_proxy=settings.warp_proxy,
        )
        strategist = Strategist()
        guardian = RiskGuardian(settings, exchange)
        journal = TradeJournal()

        notifier: WebhookNotifier | NullNotifier
        if settings.webhook_url:
            notifier = WebhookNotifier(
                settings.webhook_url,
                settings.error_webhook_url,
            )
        else:
            notifier = NullNotifier()

        paper_portfolio: PaperPortfolio | None = None
        if settings.trading_mode == "PAPER":
            paper_portfolio = PaperPortfolio(
                initial_balance=(settings.max_position_size_usd * len(settings.universe_list)),
                quote_currency=settings.quote_currency,
                db_path=str(journal.db_path),
            )

        tech_analyzer = TechnicalAnalyzer()
        vol_analyzer = VolatilityAnalyzer()
        price_forecaster = PriceForecaster(
            max_order=settings.arima_max_order,
        )
        portfolio_analyzer = PortfolioAnalyzer(journal=journal)
    except Exception:  # noqa: BLE001
        logger.critical("Component initialization failed. Exiting.")
        return

    while not shutdown_event.is_set():
        try:
            run_cycle(
                settings,
                exchange,
                oracle,
                strategist,
                guardian,
                journal,
                notifier,
                tech_analyzer,
                vol_analyzer,
                price_forecaster,
                portfolio_analyzer,
                paper_portfolio,
            )
        except Exception:
            logger.exception("Cycle failed with unhandled error.")
            notifier.notify_error("Cycle failed with unhandled error.")

        if not shutdown_event.is_set():
            logger.info(
                "Sleeping for %s seconds ...",
                settings.entry_cycle_seconds,
            )
            if shutdown_event.wait(timeout=settings.entry_cycle_seconds):
                break

    logger.info("Bot Stopped.")


if __name__ == "__main__":
    main()
