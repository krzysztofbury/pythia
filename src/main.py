"""Main execution loop and Risk Guardian."""
import logging
import signal
import sys
import threading
import time

from config import Settings
from exchange import KrakenExchange
from journal import TradeJournal
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
ROUND_UP_THRESHOLD = 8.3


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

    def check_sell(self, symbol: str) -> float:
        """Validate a SELL signal.

        Returns the total balance to sell (or 0.0 if nothing held).
        """
        base_currency = symbol.split("/")[0]
        balance = self.exchange.get_free_balance(base_currency)

        # Threshold to ignore dust (Kraken has mins, usually > 0.0001)
        if balance < 0.0001:
            logger.info("Guardian: SELL ignored for %s. No significant balance found.", symbol)
            return 0.0

        logger.info("Guardian: APPROVED SELL %s: total balance %s", symbol, balance)
        return balance

    def check_buy(self, symbol: str, price: float, signal: TradeSignal) -> float:
        """Validate a BUY signal with dynamic position sizing."""
        if signal.action != "BUY":
            return 0.0

        # Rule 1: Confidence Threshold
        if signal.confidence < MIN_CONFIDENCE:
            logger.info(
                "Guardian: Rejecting %s BUY. Confidence %s < %s",
                symbol, signal.confidence, MIN_CONFIDENCE,
            )
            return 0.0

        # Rule 2: Dynamic Position Sizing (Balance / Universe Size)
        try:
            total_balance = self.exchange.get_free_balance(self.settings.quote_currency)
            universe_size = len(self.settings.universe_list)

            # Divide total starting balance by universe size
            calculated_size = total_balance / universe_size

            # Apply Safety Cap (Tuition Cap) from settings
            target_usd = min(calculated_size, self.settings.max_position_size_usd)

            # Rule 2.5: Kraken Minimum Fulfillment
            if ROUND_UP_THRESHOLD <= target_usd < MIN_ORDER_USD:
                logger.info(
                    "Guardian: Rounding up %s to $%s for Kraken.",
                    target_usd, MIN_ORDER_USD,
                )
                target_usd = MIN_ORDER_USD

            logger.info(
                "Guardian: Calculated allocation for %s: %s (Cap: %s)",
                symbol, calculated_size, self.settings.max_position_size_usd,
            )
        except Exception:
            logger.exception("Guardian: Failed to calculate position size")
            return 0.0

        # Rule 3: Minimum Balance Check
        if total_balance < target_usd:
            logger.warning(
                "Guardian: Insufficient %s funds for %s. Have %s, need %s",
                self.settings.quote_currency, symbol, total_balance, target_usd,
            )
            return 0.0

        # Calculate amount
        amount = target_usd / price

        # Rule 4: Minimum Order Size
        if target_usd < MIN_ORDER_USD:
            logger.warning(
                "Guardian: Target value %s too low for Kraken (min ~$10).",
                target_usd,
            )
            return 0.0

        logger.info(
            "Guardian: APPROVED BUY %s: %.6f @ $%s ($%s)",
            symbol, amount, price, target_usd,
        )
        return amount


def run_cycle(settings: Settings) -> None:  # noqa: C901, PLR0915, PLR0912
    """Execute one complete trading analysis cycle (Sniper Mode)."""
    logger.info("Starting Sniper Cycle. Mode: %s", settings.trading_mode)

    try:
        exchange = KrakenExchange(settings.kraken_api_key, settings.kraken_private_key)
        oracle = SentimentOracle(
            settings.xai_api_key,
            settings.xai_model,
            settings.cmc_api_key,
        )
        strategist = Strategist(settings.google_api_key, settings.gemini_model)
        guardian = RiskGuardian(settings, exchange)
        journal = TradeJournal()
    except Exception:
        logger.exception("Component initialization failed")
        return

    # Fetch recent history once per cycle
    history = journal.get_recent_history()

    # Fetch Broad Market Sentiment (Fear & Greed) once per cycle
    fear_greed = oracle.get_fear_and_greed_index()
    logger.info("Market Sentiment: Fear & Greed Index = %s", fear_greed)

    for ticker in settings.universe_list:
        if not exchange.symbol_exists(ticker):
            logger.warning("Sniper: Skipping %s. Symbol not supported by Kraken.", ticker)
            continue

        # Fetch recent history for this specific ticker
        history = journal.get_recent_history(ticker=ticker)

        logger.info("Scanning %s...", ticker)

        try:
            # 0. Reconciliation / Exposure Check (Data Integrity)
            if guardian.check_exposure(ticker):
                continue

            # 1. Cold Facts Check (Mean Reversion & Volume Risk)
            stats = exchange.get_24h_stats(ticker)
            price = exchange.fetch_price(ticker)
            vwap = stats["vwap"]
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
            consensus = oracle.get_consensus_odds(ticker)

            # Circuit Breaker: Regulatory Ban check
            if "BAN" in sentiment.upper() or "REGULATORY" in sentiment.upper():
                logger.warning("Sniper: Regulatory Risk detected for %s. Abort.", ticker)
                continue

            logger.info(
                "Data: Price=$%s (Target <$%s), Sentiment=%s..., Consensus=%s...",
                price, target_entry, sentiment[:30], consensus[:30],
            )

            # 4. Strategy Decision
            signal = strategist.generate_signal(
                ticker, price, sentiment, consensus, fear_greed, history,
            )
            logger.info(
                "Signal: %s (Conf: %s) - %s",
                signal.action, signal.confidence, signal.reasoning,
            )

            # 5. Execution
            if signal.action == "BUY":
                limit_price = price * 0.995
                amount = guardian.check_buy(ticker, limit_price, signal)
                if amount > 0:
                    journal.log_trade(
                        account="Kraken", instrument=ticker, side="BUY",
                        price=limit_price, quantity=amount, value=limit_price * amount,
                        currency=settings.quote_currency, reasoning=signal.reasoning,
                    )
                    if settings.trading_mode == "LIVE":
                        logger.info(">>> EXECUTING LIVE SNIPER BUY <<<")
                        exchange.place_maker_buy(ticker, limit_price, amount)
                        sl_price = limit_price * (1 - settings.stop_loss_pct)
                        tp_price = limit_price * (1 + settings.take_profit_pct)
                        exchange.place_protection_orders(
                            ticker, amount, limit_price, sl_price, tp_price,
                        )
                    else:
                        logger.info(">>> PAPER SNIPER: BUY Order Recorded @ $%s <<<", limit_price)

            elif signal.action == "SELL":
                amount_to_sell = guardian.check_sell(ticker)
                if amount_to_sell > 0:
                    # TAX Calculation: Fetch last buy price
                    buy_price = journal.get_last_buy_price(ticker)
                    profit = 0.0
                    tax = 0.0

                    if buy_price:
                        # Profit per unit * quantity
                        profit = (price - buy_price) * amount_to_sell
                        # Polish Capital Gains Tax (19%)
                        if profit > 0:
                            tax = profit * 0.19
                            logger.info(
                                "Tax: Realized profit of %s %s. Tax liability: %s",
                                profit, settings.quote_currency, tax,
                            )

                    journal.log_trade(
                        account="Kraken", instrument=ticker, side="SELL",
                        price=price, quantity=amount_to_sell, value=price * amount_to_sell,
                        currency=settings.quote_currency,
                        reasoning=f"Strategic Exit: {signal.reasoning}",
                        profit=profit, tax=tax,
                    )

                    if settings.trading_mode == "LIVE":
                        logger.info(">>> EXECUTING LIVE STRATEGIC SELL <<<")
                        # 1. MUST Cancel existing protection orders (SL/TP) first
                        exchange.cancel_all_orders(ticker)
                        # 2. Place Limit Sell at current price (Maker)
                        exchange.place_maker_sell(ticker, price, amount_to_sell)
                    else:
                        logger.info(">>> PAPER SNIPER: SELL Order Recorded @ $%s <<<", price)



            time.sleep(2)

        except Exception:
            logger.exception("Error processing %s", ticker)
            continue

    logger.info("Cycle Complete.")


def main() -> None:
    """Entry point for the bot."""
    logger.info("--- ORACLE PYTHIA BOOTING VERSION 3.0 ---")
    shutdown_event = threading.Event()

    def signal_handler(_signum: int, _frame: object) -> None:
        logger.info("Shutdown signal received. Exiting...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("Oracle Pythia Bot Started.")

    try:
        settings = Settings()
    except Exception:
        logger.critical("Failed to load initial settings. Exiting.")
        return

    while not shutdown_event.is_set():
        run_cycle(settings)
        if not shutdown_event.is_set():
            logger.info("Sleeping for %s seconds ...", settings.timeout_seconds)
            if shutdown_event.wait(timeout=settings.timeout_seconds):
                break

    logger.info("Bot Stopped.")


if __name__ == "__main__":
    main()
