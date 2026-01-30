"""Main execution loop and Risk Guardian."""
import logging
import signal
import sys
import threading
import time

from config import Settings
from exchange import KrakenExchange
from sentiment import SentimentOracle
from strategy import Strategist, TradeSignal

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/bot.log"),
    ],
)
logger = logging.getLogger("OraclePythia")

# Constants
MIN_CONFIDENCE = 0.7
MIN_ORDER_USD = 10.0


class RiskGuardian:
    """The Gatekeeper. Enforces safety limits before any order goes out."""

    def __init__(self, settings: Settings, exchange: KrakenExchange) -> None:
        """Initialize the Guardian."""
        self.settings = settings
        self.exchange = exchange

    def check_liquidity(self, symbol: str) -> bool:
        """Check if the market liquidity (spread) is healthy.
        
        Returns:
            True if spread is acceptable (< max_spread_pct), False otherwise.

        """
        try:
            structure = self.exchange.get_market_structure(symbol)
            spread = structure["spread_pct"]
            limit = self.settings.max_spread_pct

            if spread > limit:
                logger.warning(
                    "Guardian: High Spread for %s. Current: %.2f%% > Limit: %.2f%%",
                    symbol, spread * 100, limit * 100,
                )
                return False

            logger.info("Guardian: Spread OK for %s (%.3f%%)", symbol, spread * 100)
            return True  # noqa: TRY300
        except Exception:
            logger.exception("Guardian: Failed to check liquidity for %s", symbol)
            return False

    def check_exposure(self, symbol: str) -> bool:
        """Check if we already have an active position or open orders for this symbol.

        Returns True if we are already exposed (should skip new trades).
        """
        try:
            # 1. Check Wallet Balance (Do we own it?)
            if self.exchange.has_open_position(symbol):
                logger.info("Guardian: Exposure detected. We already own %s.", symbol)
                return True

            # 2. Check Open Orders (Are we trying to buy/sell it?)
            open_orders = self.exchange.get_open_orders(symbol)
            if open_orders:
                logger.info("Guardian: Active orders found for %s. Skipping new entry.", symbol)
                return True

            return False  # noqa: TRY300
        except Exception:
            logger.exception("Guardian: Failed to check exposure for %s", symbol)
            # Fail safe: Assume exposure to prevent double buying in error state
            return True

    def check_market_conditions(self, ticker: str) -> bool:
        """Check Liquidity and Volatility before engaging Strategy."""
        # 1. Spread Check
        return self.check_liquidity(ticker)

    def check_buy(self, symbol: str, price: float, signal: TradeSignal) -> float:
        """Validate a BUY signal.

        Returns the safe amount to buy (or 0.0 if rejected).
        """
        if signal.action != "BUY":
            return 0.0

        # Rule 1: Confidence Threshold
        if signal.confidence < MIN_CONFIDENCE:
            logger.info(
                "Guardian: Rejecting %s BUY. Confidence %s < %s",
                symbol, signal.confidence, MIN_CONFIDENCE,
            )
            return 0.0

        # Rule 2: Max Position Size (Tuition Cap)
        target_usd = self.settings.max_position_size_usd

        # Rule 3: Balance Check
        balance = self.exchange.get_free_balance("USD")
        if balance < target_usd:
            logger.warning(
                "Guardian: Insufficient funds. Have $%s, need $%s",
                balance, target_usd,
            )
            return 0.0

        # Calculate amount
        amount = target_usd / price

        # Rule 4: Minimum Order Size
        if target_usd < MIN_ORDER_USD:
            logger.warning(
                "Guardian: Order value too low for Kraken (usually ~$10 min).",
            )
            return 0.0

        logger.info(
            "Guardian: APPROVED BUY %s: %.6f @ $%s ($%s)",
            symbol, amount, price, target_usd,
        )
        return amount


def run_cycle() -> None:  # noqa: C901, PLR0915
    """Execute one complete trading analysis cycle (Sniper Mode)."""
    try:
        settings = Settings()
    except Exception:
        logger.exception("Failed to load settings")
        return

    logger.info("Starting Sniper Cycle. Mode: %s", settings.trading_mode)

    try:
        exchange = KrakenExchange(settings.kraken_api_key, settings.kraken_private_key)
        oracle = SentimentOracle(settings.xai_api_key)
        strategist = Strategist(settings.google_api_key, settings.gemini_model)
        guardian = RiskGuardian(settings, exchange)
    except Exception:
        logger.exception("Component initialization failed")
        return

    for ticker in settings.universe_list:
        logger.info("Scanning %s...", ticker)

        try:
            # 0. Reconciliation / Exposure Check (Data Integrity)
            if guardian.check_exposure(ticker):
                continue

            # 1. Cold Facts Check (Mean Reversion & Volume Risk)
            # Fetch 24h stats first to avoid wasting API calls on bad candidates
            stats = exchange.get_24h_stats(ticker)
            price = exchange.fetch_price(ticker)

            vwap = stats["vwap"]
            # Requirement: Price < 24h Avg * 0.98 (2% dip)
            target_entry = vwap * 0.98

            if price > target_entry:
                logger.info(
                    "Sniper: Skipping %s. Price $%s > Target $%s (No dip)",
                    ticker, price, target_entry,
                )
                continue

            # 2. Liquidity Check
            if not guardian.check_market_conditions(ticker):
                continue

            # 3. AI Analysis
            sentiment = oracle.analyze_ticker_sentiment(ticker)
            # Circuit Breaker: Regulatory Ban check
            if "BAN" in sentiment.upper() or "REGULATORY" in sentiment.upper():
                logger.warning(
                    "Sniper: Regulatory Risk detected in sentiment for %s. Abort.",
                    ticker,
                )
                continue

            odds = oracle.get_prediction_market_odds(f"Will {ticker.split('/')[0]} go up?")

            logger.info(
                "Data: Price=$%s (Target <$%s), Sentiment=%s..., Odds=%s",
                price, target_entry, sentiment[:30], odds,
            )

            # 4. Strategy Decision
            signal = strategist.generate_signal(ticker, price, sentiment, odds)
            logger.info(
                "Signal: %s (Conf: %s) - %s",
                signal.action, signal.confidence, signal.reasoning,
            )

            # 5. Execution (Sniper Entry)
            if signal.action == "BUY":
                # Sniper: Place Limit Buy at 0.5% below current price
                limit_price = price * 0.995

                amount = guardian.check_buy(ticker, limit_price, signal)

                if amount > 0:
                    if settings.trading_mode == "LIVE":
                        logger.info(">>> EXECUTING LIVE SNIPER ORDER <<<")
                        # 1. Place Entry
                        exchange.place_maker_buy(ticker, limit_price, amount)

                        # 2. Place Protection (SL/TP)
                        sl_price = limit_price * (1 - settings.stop_loss_pct)
                        tp_price = limit_price * (1 + settings.take_profit_pct)

                        exchange.place_protection_orders(
                            ticker, amount, limit_price, sl_price, tp_price,
                        )
                        logger.info(
                            "Protection Orders Set: SL=$%s, TP=$%s",
                            sl_price, tp_price,
                        )
                    else:
                        logger.info(
                            ">>> PAPER SNIPER: Order Simulated @ $%s <<<",
                            limit_price,
                        )

            # Rate limit sleep
            time.sleep(2)

        except Exception:
            logger.exception("Error processing %s", ticker)
            continue

    logger.info("Cycle Complete.")


def main() -> None:
    """Entry point for the bot."""
    # Graceful Shutdown Handling
    shutdown_event = threading.Event()

    def signal_handler(_signum: int, _frame: object) -> None:
        logger.info("Shutdown signal received. Exiting...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("Oracle Pythia Bot Started.")

    while not shutdown_event.is_set():
        run_cycle()

        # Sleep for 1 hour, but wake up if shutdown signal is received
        if not shutdown_event.is_set():
            logger.info("Sleeping for 1 hour...")
            # wait returns True if flag is set (shutdown), False if timeout (loop continues)
            if shutdown_event.wait(timeout=3600):
                break

    logger.info("Bot Stopped.")


if __name__ == "__main__":
    main()
