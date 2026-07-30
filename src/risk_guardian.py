"""RiskGuardian: pre-trade safety gates and portfolio-level risk checks."""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cache import CachedExchange
    from config import Settings
    from exchange import KrakenExchange
    from execution_store import ExecutionStore
    from journal import TradeJournal
    from models import TradeSignal

logger = logging.getLogger("OraclePythia")

MIN_ORDER_USD = 10.0
# v9.6: lowered 8.3 → 5.0. With a ~$100 account, 7 slots and the 0.6x tier-1
# multiplier, every allocation landed at ~$6-8 — below the old threshold — so
# the bot silently rejected ALL entries for weeks. Allocations at or above
# half the exchange minimum now round up to MIN_ORDER_USD (when balance and
# budget cover it); anything smaller is genuine dust and is still rejected.
ROUND_UP_THRESHOLD = 5.0


def compute_conviction_multiplier(composite: float, settings: Settings) -> float:
    """Map a composite score to a position-size multiplier via the Settings tiers.

    Below ``sizing_tier_1_threshold`` the multiplier is 0 (strategy would have
    already rejected the BUY, but this keeps the function total). The tier
    thresholds must be ascending; we walk them in order and return the
    highest-tier multiplier whose threshold is met.
    """
    assert -1.0 <= composite <= 1.0, f"composite out of range: {composite}"  # noqa: S101
    t1 = settings.sizing_tier_1_threshold
    t2 = settings.sizing_tier_2_threshold
    t3 = settings.sizing_tier_3_threshold
    t4 = settings.sizing_tier_4_threshold
    assert t1 < t2 < t3 < t4, (  # noqa: S101
        f"sizing tier thresholds must be ascending: {(t1, t2, t3, t4)}"
    )
    if composite < t1:
        return 0.0
    if composite < t2:
        return settings.sizing_tier_1_multiplier
    if composite < t3:
        return settings.sizing_tier_2_multiplier
    if composite < t4:
        return settings.sizing_tier_3_multiplier
    return settings.sizing_tier_4_multiplier


REGULATORY_DANGER_PHRASES = [
    "REGULATORY BAN",
    "BANNED",
    "DELISTED",
    "CEASE AND DESIST",
    "TRADING HALTED",
    "SECURITIES VIOLATION",
]

# Crypto correlation groups — assets that move together.
# v9.5 tightening: ETH and L1 alts no longer share a group (their dispersion is
# high enough that treating them as one bucket over-constrains entries).
CORRELATION_GROUPS: dict[str, str] = {
    "BTC": "btc_core",
    "LTC": "btc_core",
    "ETH": "eth_core",
    "SOL": "l1_alt",
    "AVAX": "l1_alt",
    "DOT": "l1_alt",
    "LINK": "defi",
    "XRP": "payments",
    "DOGE": "meme",
    "SHIB": "meme",
}


class RiskGuardian:
    """The Gatekeeper. Enforces safety limits before any order goes out."""

    def __init__(
        self,
        settings: Settings,
        exchange: CachedExchange | KrakenExchange,
        *,
        market_exchange: CachedExchange | KrakenExchange | None = None,
        execution_store: ExecutionStore | None = None,
    ) -> None:
        """Initialize the Guardian."""
        self.settings = settings
        self.exchange = exchange
        self.market_exchange = market_exchange or exchange
        self.execution_store = execution_store

    def check_liquidity(self, symbol: str) -> bool:
        """Check if the market liquidity (spread) is healthy."""
        try:
            structure = self.market_exchange.get_market_structure(symbol)
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

    def check_buy(  # noqa: C901, D417, PLR0911, PLR0913
        self,
        symbol: str,
        price: float,
        trade_signal: TradeSignal,
        available_slots: int,
        *,
        budget_usd: float | None = None,
        is_dca: bool = False,
    ) -> float:
        """Validate a BUY signal with dynamic position sizing.

        Parameters
        ----------
        budget_usd:
            Optional intra-cycle budget ceiling. Prevents two tier-4 entries
            in the same cycle from combining to exceed the concentration cap
            (fixes a cache-staleness hazard since ``get_free_balance`` reads
            cached balance mid-cycle).
        is_dca:
            When True, conviction multiplier is clamped to 1.0 so DCAs don't
            stack tier amplification on top of an existing position.

        """
        assert price > 0, f"Price must be positive, got {price}"  # noqa: S101

        if trade_signal.action != "BUY":
            return 0.0

        if trade_signal.confidence < self.settings.min_buy_confidence:
            logger.info(
                "Guardian: Rejecting %s BUY. Confidence %s < %s",
                symbol,
                trade_signal.confidence,
                self.settings.min_buy_confidence,
            )
            return 0.0

        try:
            free_quote = float(self.exchange.get_free_balance(self.settings.quote_currency))
            total_quote = (
                float(self.exchange.get_total_balance(self.settings.quote_currency))
                if self.execution_store is not None
                else free_quote
            )
            total_equity, existing_exposure = self._marked_equity(symbol, price, total_quote)
            slots = max(available_slots, 1)
            base_size = free_quote / slots

            multiplier = 1.0
            if self.settings.conviction_sizing_enabled:
                multiplier = compute_conviction_multiplier(
                    trade_signal.composite,
                    self.settings,
                )
                if multiplier <= 0.0:
                    if is_dca:
                        logger.info(
                            "Guardian: Rejecting %s DCA. Composite %.3f below "
                            "tier-1 threshold %.3f.",
                            symbol,
                            trade_signal.composite,
                            self.settings.sizing_tier_1_threshold,
                        )
                        return 0.0
                    # v9.6: regime modifiers can lower the strategy's entry
                    # threshold below the tier-1 floor. The strategy is the
                    # entry authority — size the marginal signal at tier-1
                    # instead of silently killing it (447 BUYs died this way
                    # in 2026-04/05).
                    logger.info(
                        "Guardian: Composite %.3f below tier-1 %.3f for %s; "
                        "sizing at tier-1 multiplier %.2fx.",
                        trade_signal.composite,
                        self.settings.sizing_tier_1_threshold,
                        symbol,
                        self.settings.sizing_tier_1_multiplier,
                    )
                    multiplier = self.settings.sizing_tier_1_multiplier
                if is_dca and multiplier > 1.0:
                    logger.info(
                        "Guardian: DCA multiplier clamped %.2fx -> 1.0x for %s "
                        "(no tier amplification on existing position).",
                        multiplier,
                        symbol,
                    )
                    multiplier = 1.0

            tier_scaled = base_size * multiplier
            concentration_cap = max(
                total_equity * self.settings.max_position_pct_of_balance - existing_exposure,
                0.0,
            )
            candidates = [
                tier_scaled,
                concentration_cap,
                self.settings.max_position_size_usd,
            ]
            if budget_usd is not None:
                assert budget_usd >= 0, f"budget_usd must be non-negative, got {budget_usd}"  # noqa: S101
                candidates.append(budget_usd)
            target_usd = min(candidates)

            # Round up to the exchange minimum only when balance AND the
            # intra-cycle budget can actually cover the bumped order.
            spendable = free_quote if budget_usd is None else min(free_quote, budget_usd)
            # The bump must never breach the concentration cap — on a depleted
            # account (balance < MIN_ORDER/max_pct) rounding up would put >40%
            # of the book into one position.
            if (
                ROUND_UP_THRESHOLD <= target_usd < MIN_ORDER_USD <= spendable
                and concentration_cap >= MIN_ORDER_USD
            ):
                logger.info(
                    "Guardian: Rounding up %s to $%s for Kraken.",
                    target_usd,
                    MIN_ORDER_USD,
                )
                target_usd = MIN_ORDER_USD

            logger.info(
                "Guardian: Allocation for %s: $%.2f (Balance $%.2f / %d slots = "
                "base $%.2f x %.2fx tier → concentration cap $%.2f, hard cap $%s, "
                "budget $%s, dca=%s)",
                symbol,
                target_usd,
                free_quote,
                slots,
                base_size,
                multiplier,
                concentration_cap,
                self.settings.max_position_size_usd,
                budget_usd if budget_usd is not None else "-",
                is_dca,
            )
        except Exception:
            logger.exception("Guardian: Failed to calculate position size")
            return 0.0

        if free_quote < target_usd:
            logger.warning(
                "Guardian: Insufficient %s funds for %s. Have %s, need %s",
                self.settings.quote_currency,
                symbol,
                free_quote,
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

    def _marked_equity(
        self,
        candidate_symbol: str,
        candidate_price: float,
        total_quote: float,
    ) -> tuple[float, float]:
        """Return total marked equity and candidate post-trade exposure already reserved."""
        store = self.execution_store
        if store is None:
            return total_quote, 0.0

        marked_positions = 0.0
        candidate_exposure = 0.0
        for position in store.list_open_positions():
            mark = (
                candidate_price
                if position.symbol == candidate_symbol
                else float(self.market_exchange.fetch_price(position.symbol))
            )
            value = float(position.quantity) * mark
            marked_positions += value
            if position.symbol == candidate_symbol:
                candidate_exposure += value

        # Active BUY quantities are conservative marked reservations. Free quote
        # already enforces the cash reservation; this term enforces concentration.
        for order in store.list_orders(active_only=True):
            context = store.get_order_context(order.order_id)
            if context["side"] != "BUY" or context["symbol"] != candidate_symbol:
                continue
            remaining = max(order.requested_quantity - order.filled_quantity, Decimal(0))
            candidate_exposure += float(remaining) * candidate_price
        return total_quote + marked_positions, candidate_exposure


def check_correlation_limit(
    ticker: str,
    owned_tickers: list[str],
    settings: Settings,
) -> bool:
    """Gate a new entry against correlation + portfolio-wide exposure caps.

    Two caps apply in order:
    1. ``max_total_crypto_positions`` — portfolio-wide count across every
       correlation group. Backstop against the v9.5 regrouping accidentally
       letting the bot hold 7+ correlated crypto positions.
    2. ``max_correlated_positions`` — per-group cap (the historical gate).
    """
    # Portfolio-wide exposure backstop (v9.5).
    max_total = getattr(settings, "max_total_crypto_positions", None)
    if max_total is not None and len(owned_tickers) >= max_total:
        logger.info(
            "Guardian: Total position limit reached for %s (%d/%d positions). Skipping.",
            ticker,
            len(owned_tickers),
            max_total,
        )
        return False

    base = ticker.split("/")[0]
    new_group = CORRELATION_GROUPS.get(base, base)

    same_group_count = 0
    for owned in owned_tickers:
        owned_base = owned.split("/")[0]
        owned_group = CORRELATION_GROUPS.get(owned_base, owned_base)
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


def check_daily_loss_limit(journal: TradeJournal | ExecutionStore, settings: Settings) -> bool:
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
