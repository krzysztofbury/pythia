"""Main execution loop — boot, logging, cycle orchestration."""

import datetime
import logging
import logging.handlers
import signal
import sys
import threading
import time
from pathlib import Path

from analytics import PortfolioAnalyzer
from cache import CachedExchange
from config import Settings
from data_pipeline import collect_all_market_data
from decision_engine import evaluate_entry, evaluate_exit
from exchange import KrakenExchange
from forecast import PriceForecaster
from indicators import TechnicalAnalyzer
from journal import TradeJournal
from llm_challenger import LLMChallenger
from notifier import NullNotifier, WebhookNotifier
from order_executor import reconcile_fills, reconcile_stale_orders
from paper_portfolio import PaperPortfolio
from portfolio_manager import classify_tickers, compute_available_slots
from risk_guardian import RiskGuardian, check_daily_loss_limit
from sentiment import SentimentOracle
from strategy import Strategist
from volatility import VolatilityAnalyzer

logger = logging.getLogger("OraclePythia")


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


_DROUGHT_ALERT_THROTTLE_SECONDS = 24 * 3600.0
_drought_alert_state: dict[str, float] = {}


def check_entry_drought(
    journal: TradeJournal,
    settings: Settings,
    notifier: WebhookNotifier | NullNotifier,
    *,
    _state: dict[str, float] | None = None,
) -> bool:
    """Alert when a LIVE bot has not opened a position for too long.

    The bot executed zero entries between 2026-04-23 and 2026-06-11 without
    a single notification — silence must be treated as a failure mode.
    Returns True when an alert was sent. Throttled to once per 24h.
    """
    state = _drought_alert_state if _state is None else _state
    if settings.trading_mode != "LIVE" or settings.no_entry_alert_hours <= 0:
        return False

    last_entry = journal.get_last_entry_timestamp()
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    if last_entry is not None:
        idle_hours = (now - last_entry).total_seconds() / 3600.0
        if idle_hours < settings.no_entry_alert_hours:
            return False
    # No entry ever recorded counts as a drought: a LIVE bot that has never
    # bought anything is exactly the silent state this alert exists for.

    last_alert = state.get("last_alert_monotonic")
    if last_alert is not None and (time.monotonic() - last_alert) < _DROUGHT_ALERT_THROTTLE_SECONDS:
        return False

    idle_desc = f"{(now - last_entry).days}d" if last_entry is not None else "ever"
    message = (
        f"Entry drought: no BUY executed for {idle_desc} "
        f"(threshold {settings.no_entry_alert_hours:.0f}h). "
        "Check sentiment provider credits, guardian rejections, and balance."
    )
    logger.warning(message)
    try:
        notifier.notify_error(message)
    except Exception:  # noqa: BLE001
        logger.debug("notify_error failed for entry-drought alert", exc_info=True)
    state["last_alert_monotonic"] = time.monotonic()
    return True


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
    challenger: LLMChallenger | None = None,
) -> None:
    """Execute one complete trading analysis cycle (2-pass system)."""
    cycle_start = time.monotonic()

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

    reconcile_stale_orders(settings, exchange, journal)

    if settings.trading_mode == "LIVE":
        reconcile_fills(
            tickers=settings.universe_list,
            exchange=exchange,
            journal=journal,
            notifier=notifier,
            quote_currency=settings.quote_currency,
            tax_rate=settings.tax_rate,
        )

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

    owned_tickers, available_tickers = classify_tickers(
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

    available_slots = compute_available_slots(
        n_available=len(available_tickers),
        n_owned=len(owned_tickers),
        max_total=settings.max_total_crypto_positions,
    )

    daily_loss_ok = check_daily_loss_limit(journal, settings)

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

    cycle_vol_analyzer = vol_analyzer if is_full_cycle else None
    cycle_price_forecaster = price_forecaster if is_full_cycle else None

    all_tickers = owned_tickers + available_tickers
    all_data = collect_all_market_data(
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

    for ticker in owned_tickers:
        data = all_data.get(ticker)
        if not data:
            continue
        try:
            evaluate_exit(
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
                challenger=challenger,
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

    if daily_loss_ok:
        entries_this_cycle = 0
        # Intra-cycle budget prevents a 2nd entry from re-using stale cached
        # balance — otherwise two tier-4 entries could each allocate 40% of
        # balance in one cycle, producing an 80% concentration spike.
        try:
            cycle_budget_usd = cached_exchange.get_free_balance(settings.quote_currency)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to read cycle starting balance; skipping budget cap.")
            cycle_budget_usd = float("inf")
        for ticker in available_tickers:
            if entries_this_cycle >= settings.max_entries_per_cycle:
                logger.info(
                    "Entry limit reached (%d/%d). Skipping remaining tickers.",
                    entries_this_cycle,
                    settings.max_entries_per_cycle,
                )
                break
            data = all_data.get(ticker)
            if not data:
                continue
            try:
                bought, spent_usd = evaluate_entry(
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
                    challenger=challenger,
                    budget_usd=cycle_budget_usd,
                )
                if bought:
                    entries_this_cycle += 1
                    available_slots = max(available_slots - 1, 1)
                    owned_tickers.append(ticker)
                    cycle_budget_usd = max(cycle_budget_usd - spent_usd, 0.0)
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

    check_entry_drought(journal, settings, notifier)

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
    logger.info("--- ORACLE PYTHIA BOOTING VERSION 9.7 ---")
    logger.info(
        "Effective config: mode=%s quote=%s universe=%s conviction_sizing=%s "
        "challenger=%s funding=%s sentiment_model=%s fallback=%s "
        "sentiment_cache=%ss require_bullish=%s",
        settings.trading_mode,
        settings.quote_currency,
        settings.universe_list,
        settings.conviction_sizing_enabled,
        settings.challenger_enabled,
        settings.funding_rates_enabled,
        settings.sentiment_model,
        settings.sentiment_fallback_model or "(none)",
        settings.sentiment_cache_seconds,
        settings.require_bullish_sentiment,
    )

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
        extra_api_keys = {
            provider: key.get_secret_value()
            for provider, key in (
                ("gemini", settings.gemini_api_key),
                ("anthropic", settings.anthropic_api_key),
                ("openai", settings.openai_api_key),
            )
            if key is not None
        }
        oracle = SentimentOracle(
            settings.xai_api_key,
            settings.sentiment_model,
            settings.cmc_api_key,
            warp_proxy=settings.warp_proxy,
            fallback_model=settings.sentiment_fallback_model,
            extra_api_keys=extra_api_keys,
            cache_ttl_seconds=settings.sentiment_cache_seconds,
        )
        strategist = Strategist(settings=settings)
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
        from llm_challenger import build_challenger_from_settings

        challenger = build_challenger_from_settings(settings)
        logger.info(
            "LLM Challenger: enabled=%s model=%s",
            challenger.enabled,
            settings.challenger_model if challenger.enabled else "(off)",
        )
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
                challenger=challenger,
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
