"""Stop manager — trailing SL, breakeven lock, ROI decay, partial exits.

Manages per-position stop-loss and take-profit levels dynamically based on
market regime, price movement, and time-based profit targets.
"""

import logging
from dataclasses import dataclass
from typing import Literal

from models import PositionState, StopMode

logger = logging.getLogger(__name__)


@dataclass
class StopAction:
    """Result of stop manager evaluation."""

    action: Literal["HOLD", "SELL", "PARTIAL_SELL"]
    reason: str = ""
    updated_sl: float = 0.0
    updated_tp: float = 0.0
    sell_pct: float = 1.0  # 1.0 = full sell, 0.5 = half

    def __post_init__(self) -> None:
        """Validate sell_pct bounds."""
        assert 0.0 < self.sell_pct <= 1.0, (  # noqa: S101
            f"sell_pct must be in (0.0, 1.0], got {self.sell_pct}"
        )


def parse_roi_schedule(schedule_str: str) -> list[tuple[int, float]]:
    """Parse ROI schedule string into sorted list of (minutes, profit_pct) tuples.

    Format: "0:4.0,60:3.0,120:2.0,240:1.0"
    Returns: [(0, 4.0), (60, 3.0), (120, 2.0), (240, 1.0)]
    """
    # Strip surrounding quotes that .env files may include
    schedule_str = schedule_str.strip().strip('"').strip("'")
    pairs = []
    for item in schedule_str.split(","):
        item = item.strip().strip('"').strip("'")
        if ":" not in item:
            continue
        minutes_str, pct_str = item.split(":", 1)
        pairs.append((int(minutes_str.strip()), float(pct_str.strip())))
    return sorted(pairs, key=lambda x: x[0])


class StopManager:
    """Manages dynamic stop-loss, take-profit, ROI exits, and partial fills."""

    def __init__(self, settings) -> None:
        """Initialize with bot settings."""
        self.settings = settings
        self._roi_schedule = (
            parse_roi_schedule(settings.roi_schedule) if settings.roi_enabled else []
        )

    def compute_trailing_sl(
        self,
        state: PositionState,
        current_price: float,
        atr: float | None,
    ) -> float:
        """Compute trailing stop-loss level.

        SL = high_watermark - trailing_sl_atr_multiplier * ATR
        Never moves down from current SL.
        """
        if atr is None or atr <= 0:
            return state.current_sl

        new_watermark = max(state.high_watermark, current_price)
        trailing_sl = new_watermark - self.settings.trailing_sl_atr_multiplier * atr

        # Never lower the stop-loss
        return max(trailing_sl, state.current_sl)

    def check_breakeven_lock(
        self,
        state: PositionState,
        current_price: float,
        atr: float | None,
    ) -> float | None:
        """Check if position qualifies for breakeven lock.

        If price >= entry + breakeven_threshold * ATR, move SL to entry price.
        Returns new SL if breakeven triggered, None otherwise.
        """
        if atr is None or atr <= 0:
            return None

        breakeven_trigger = state.entry_price + self.settings.breakeven_atr_threshold * atr
        if current_price >= breakeven_trigger and state.current_sl < state.entry_price:
            return state.entry_price
        return None

    def check_roi_exit(
        self,
        pnl_pct: float,
        position_age_minutes: float,
    ) -> tuple[bool, str]:
        """Check if position should exit based on time-decaying ROI targets.

        Returns (should_exit, reason).
        """
        if not self._roi_schedule:
            return False, ""

        # Find the applicable ROI target (highest minutes that position has exceeded)
        target_pct = None
        for minutes, pct in reversed(self._roi_schedule):
            if position_age_minutes >= minutes:
                target_pct = pct
                break

        if target_pct is not None and pnl_pct >= target_pct:
            return True, (
                f"ROI exit: profit {pnl_pct:.1f}% >= target {target_pct:.1f}% "
                f"at {position_age_minutes:.0f}min"
            )
        return False, ""

    def check_partial_exit(
        self,
        state: PositionState,
        current_price: float,
        atr: float | None,
    ) -> tuple[bool, float, str]:
        """Check if position should do a partial exit.

        Returns (should_partial, sell_pct, reason).
        """
        if not self.settings.partial_exit_enabled:
            return False, 0.0, ""

        if state.partial_exits > 0:
            return False, 0.0, ""  # Already did a partial exit

        if atr is None or atr <= 0:
            return False, 0.0, ""

        profit_in_atr = (current_price - state.entry_price) / atr
        if profit_in_atr >= self.settings.partial_exit_atr_threshold:
            return (
                True,
                self.settings.partial_exit_pct,
                (
                    f"Partial exit: profit {profit_in_atr:.1f}x ATR >= "
                    f"{self.settings.partial_exit_atr_threshold}x threshold"
                ),
            )
        return False, 0.0, ""

    def adjust(
        self,
        state: PositionState,
        current_price: float,
        atr: float | None,
        pnl_pct: float,
        position_age_minutes: float,
    ) -> StopAction:
        """Main orchestrator: evaluate all stop conditions and return action.

        Evaluation order:
        1. ROI time-based exit
        2. Partial exit check
        3. Trailing SL adjustment (TRAILING or TRAILING_BREAKEVEN modes)
        4. Breakeven lock check (TRAILING_BREAKEVEN mode)
        5. Stop-loss trigger check
        """
        # 1. ROI exit check
        if self.settings.roi_enabled:
            should_roi, roi_reason = self.check_roi_exit(pnl_pct, position_age_minutes)
            if should_roi:
                return StopAction(action="SELL", reason=roi_reason)

        # 2. Partial exit check
        should_partial, sell_pct, partial_reason = self.check_partial_exit(
            state,
            current_price,
            atr,
        )
        if should_partial:
            return StopAction(
                action="PARTIAL_SELL",
                reason=partial_reason,
                sell_pct=sell_pct,
            )

        # 3 & 4. Trailing stop and breakeven adjustments
        updated_sl = state.current_sl
        if state.stop_mode in (StopMode.TRAILING, StopMode.TRAILING_BREAKEVEN):
            if self.settings.trailing_stop_enabled:
                updated_sl = self.compute_trailing_sl(state, current_price, atr)

        if state.stop_mode == StopMode.TRAILING_BREAKEVEN:
            breakeven_sl = self.check_breakeven_lock(state, current_price, atr)
            if breakeven_sl is not None:
                updated_sl = max(updated_sl, breakeven_sl)

        # Safety invariant: SL must never decrease
        assert updated_sl >= state.current_sl, (  # noqa: S101
            f"SL must never decrease: {updated_sl} < {state.current_sl}"
        )
        assert updated_sl >= 0, f"SL must be non-negative: {updated_sl}"  # noqa: S101

        # 5. Check if current price hit the stop-loss
        if updated_sl > 0 and current_price <= updated_sl:
            return StopAction(
                action="SELL",
                reason=f"Stop-loss triggered: price {current_price:.8f} <= SL {updated_sl:.8f}",
                updated_sl=updated_sl,
            )

        return StopAction(
            action="HOLD",
            updated_sl=updated_sl,
            updated_tp=state.current_tp,
        )
