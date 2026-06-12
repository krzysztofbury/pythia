"""Market data collection pipeline — parallel fetch of price, indicators, sentiment, regime."""

import datetime
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from cache import CachedExchange
from config import Settings
from forecast import PriceForecaster
from funding import FundingRateFetcher
from indicators import TechnicalAnalyzer
from journal import TradeJournal
from models import MarketData
from regime import RegimeClassifier
from sentiment import SentimentOracle
from volatility import VolatilityAnalyzer

logger = logging.getLogger("OraclePythia")

_FUNDING_FETCHER: FundingRateFetcher | None = None


def _get_funding_fetcher(settings: Settings) -> FundingRateFetcher | None:
    """Lazily build (and cache) the FundingRateFetcher when enabled."""
    if not settings.funding_rates_enabled:
        return None
    global _FUNDING_FETCHER  # noqa: PLW0603
    if _FUNDING_FETCHER is None:
        _FUNDING_FETCHER = FundingRateFetcher(settings.funding_rates_exchange)
    return _FUNDING_FETCHER if _FUNDING_FETCHER.enabled else None


def build_market_data(  # noqa: PLR0913
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

    funding_rate: float | None = None
    funding_fetcher = _get_funding_fetcher(settings)
    if funding_fetcher is not None:
        funding_rate = funding_fetcher.fetch(ticker)
        if funding_rate is not None:
            logger.info("Funding %s: %+0.5f%%", ticker, funding_rate * 100)

    pnl_pct, tp_distance_pct = _compute_position_context(
        price,
        buy_price,
        settings.take_profit_pct,
        has_position,
    )

    position_age_hours = None
    if has_position:
        open_time = journal.get_position_open_time(ticker)
        if open_time:
            now = datetime.datetime.now(tz=datetime.timezone.utc)
            delta = now - open_time
            position_age_hours = delta.total_seconds() / 3600

    history = journal.get_recent_history(ticker=ticker)

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
        funding_rate=funding_rate,
    )

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
        funding_rate=data.funding_rate,
    )

    return data


def _fetch_price_context(
    exchange: CachedExchange,
    ticker: str,
) -> tuple[float, float | None, float | None, float | None, float | None]:
    """Fetch price, VWAP, SMA and deviation data."""
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
    """Compute PnL and TP distance for an owned position."""
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
    """Collect short and long timeframe data for multi-timeframe analysis."""
    if not settings.mtf_enabled:
        return None, None

    trend_context = None
    short_tf = None

    try:
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


def collect_all_market_data(  # noqa: PLR0913
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
    """Collect market data for all tickers in parallel."""
    results: dict[str, MarketData] = {}

    if not tickers:
        return results

    with ThreadPoolExecutor(max_workers=max(min(len(tickers), 4), 1)) as executor:
        future_to_ticker = {}
        for ticker in tickers:
            future = executor.submit(
                build_market_data,
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
