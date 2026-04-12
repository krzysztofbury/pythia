"""Tests for stop manager — trailing SL, breakeven lock, ROI decay, partial exits."""

import pytest

from models import PositionState, StopMode
from stop_manager import StopManager, parse_roi_schedule


class FakeSettings:
    """Minimal settings stub for StopManager tests."""

    trailing_stop_enabled = True
    trailing_sl_atr_multiplier = 1.5
    breakeven_atr_threshold = 1.0
    roi_enabled = True
    roi_schedule = "0:4.0,60:3.0,120:2.0,240:1.0"
    partial_exit_enabled = True
    partial_exit_pct = 0.50
    partial_exit_atr_threshold = 2.0


# ---------------------------------------------------------------------------
# parse_roi_schedule
# ---------------------------------------------------------------------------


class TestParseRoiSchedule:
    def test_valid_schedule(self):
        result = parse_roi_schedule("0:4.0,60:3.0,120:2.0,240:1.0")
        assert result == [(0, 4.0), (60, 3.0), (120, 2.0), (240, 1.0)]

    def test_empty_string(self):
        assert parse_roi_schedule("") == []

    def test_single_entry(self):
        assert parse_roi_schedule("0:5.0") == [(0, 5.0)]

    def test_unsorted_input_is_sorted(self):
        result = parse_roi_schedule("120:2.0,0:4.0,60:3.0")
        assert result == [(0, 4.0), (60, 3.0), (120, 2.0)]

    def test_whitespace_tolerance(self):
        result = parse_roi_schedule(" 0 : 4.0 , 60 : 3.0 ")
        assert result == [(0, 4.0), (60, 3.0)]

    def test_items_without_colon_skipped(self):
        result = parse_roi_schedule("0:4.0,bad,60:3.0")
        assert result == [(0, 4.0), (60, 3.0)]

    def test_quoted_schedule_from_env(self):
        """Handles .env values wrapped in double quotes: ROI_SCHEDULE="0:4.0,60:3.0"."""
        result = parse_roi_schedule('"0:4.0,60:3.0,120:2.0,240:1.0"')
        assert result == [(0, 4.0), (60, 3.0), (120, 2.0), (240, 1.0)]

    def test_single_quoted_schedule_from_env(self):
        """Handles .env values wrapped in single quotes."""
        result = parse_roi_schedule("'0:4.0,60:3.0'")
        assert result == [(0, 4.0), (60, 3.0)]


# ---------------------------------------------------------------------------
# compute_trailing_sl
# ---------------------------------------------------------------------------


class TestComputeTrailingSl:
    def setup_method(self):
        self.mgr = StopManager(FakeSettings())

    def _state(self, **kwargs) -> PositionState:
        defaults = {
            "ticker": "BTC/USDT",
            "entry_price": 100.0,
            "high_watermark": 110.0,
            "stop_mode": StopMode.TRAILING,
            "current_sl": 95.0,
        }
        defaults.update(kwargs)
        return PositionState(**defaults)

    def test_sl_ratchets_up(self):
        state = self._state(high_watermark=110.0, current_sl=95.0)
        # new SL = max(110, 115) - 1.5*5 = 115 - 7.5 = 107.5
        new_sl = self.mgr.compute_trailing_sl(state, current_price=115.0, atr=5.0)
        assert new_sl == pytest.approx(107.5)

    def test_sl_never_moves_down(self):
        state = self._state(high_watermark=120.0, current_sl=112.0)
        # price dropped, new_watermark = max(120, 105) = 120
        # trailing_sl = 120 - 1.5*10 = 105 < current 112 => keep 112
        new_sl = self.mgr.compute_trailing_sl(state, current_price=105.0, atr=10.0)
        assert new_sl == pytest.approx(112.0)

    def test_none_atr_returns_current_sl(self):
        state = self._state(current_sl=95.0)
        assert self.mgr.compute_trailing_sl(state, current_price=120.0, atr=None) == 95.0

    def test_zero_atr_returns_current_sl(self):
        state = self._state(current_sl=95.0)
        assert self.mgr.compute_trailing_sl(state, current_price=120.0, atr=0.0) == 95.0

    def test_negative_atr_returns_current_sl(self):
        state = self._state(current_sl=95.0)
        assert self.mgr.compute_trailing_sl(state, current_price=120.0, atr=-1.0) == 95.0

    def test_watermark_updates_with_higher_price(self):
        state = self._state(high_watermark=100.0, current_sl=90.0)
        # new_watermark = max(100, 108) = 108; trailing_sl = 108 - 1.5*2 = 105
        new_sl = self.mgr.compute_trailing_sl(state, current_price=108.0, atr=2.0)
        assert new_sl == pytest.approx(105.0)


# ---------------------------------------------------------------------------
# check_breakeven_lock
# ---------------------------------------------------------------------------


class TestCheckBreakevenLock:
    def setup_method(self):
        self.mgr = StopManager(FakeSettings())

    def _state(self, **kwargs) -> PositionState:
        defaults = {
            "ticker": "BTC/USDT",
            "entry_price": 100.0,
            "high_watermark": 100.0,
            "stop_mode": StopMode.TRAILING_BREAKEVEN,
            "current_sl": 95.0,
        }
        defaults.update(kwargs)
        return PositionState(**defaults)

    def test_triggers_at_threshold(self):
        state = self._state(entry_price=100.0, current_sl=95.0)
        # breakeven_trigger = 100 + 1.0 * 5 = 105; price 106 >= 105 => entry
        result = self.mgr.check_breakeven_lock(state, current_price=106.0, atr=5.0)
        assert result == 100.0

    def test_does_not_trigger_below_threshold(self):
        state = self._state(entry_price=100.0, current_sl=95.0)
        # breakeven_trigger = 100 + 1.0 * 5 = 105; price 104 < 105
        result = self.mgr.check_breakeven_lock(state, current_price=104.0, atr=5.0)
        assert result is None

    def test_does_not_trigger_when_sl_already_above_entry(self):
        state = self._state(entry_price=100.0, current_sl=101.0)
        # SL already above entry => no action needed
        result = self.mgr.check_breakeven_lock(state, current_price=110.0, atr=5.0)
        assert result is None

    def test_none_atr(self):
        state = self._state()
        assert self.mgr.check_breakeven_lock(state, current_price=110.0, atr=None) is None

    def test_zero_atr(self):
        state = self._state()
        assert self.mgr.check_breakeven_lock(state, current_price=110.0, atr=0.0) is None

    def test_exact_threshold_triggers(self):
        state = self._state(entry_price=100.0, current_sl=95.0)
        # breakeven_trigger = 100 + 1.0 * 10 = 110; price == 110
        result = self.mgr.check_breakeven_lock(state, current_price=110.0, atr=10.0)
        assert result == 100.0


# ---------------------------------------------------------------------------
# check_roi_exit
# ---------------------------------------------------------------------------


class TestCheckRoiExit:
    def setup_method(self):
        self.mgr = StopManager(FakeSettings())

    def test_matches_correct_tier_early(self):
        # At 30 min, tier 0:4.0 applies. PnL 5% >= 4.0% => exit
        should_exit, reason = self.mgr.check_roi_exit(pnl_pct=5.0, position_age_minutes=30)
        assert should_exit is True
        assert "4.0%" in reason

    def test_matches_correct_tier_later(self):
        # At 130 min, tier 120:2.0 applies. PnL 2.5% >= 2.0% => exit
        should_exit, reason = self.mgr.check_roi_exit(pnl_pct=2.5, position_age_minutes=130)
        assert should_exit is True
        assert "2.0%" in reason

    def test_below_target_no_exit(self):
        # At 30 min, tier 0:4.0. PnL 3% < 4.0% => hold
        should_exit, _ = self.mgr.check_roi_exit(pnl_pct=3.0, position_age_minutes=30)
        assert should_exit is False

    def test_boundary_exact_minutes(self):
        # At exactly 60 min, tier 60:3.0 applies. PnL 3.0% >= 3.0% => exit
        should_exit, reason = self.mgr.check_roi_exit(pnl_pct=3.0, position_age_minutes=60)
        assert should_exit is True
        assert "3.0%" in reason

    def test_latest_tier_at_long_hold(self):
        # At 500 min, tier 240:1.0 applies. PnL 1.5% >= 1.0% => exit
        should_exit, reason = self.mgr.check_roi_exit(pnl_pct=1.5, position_age_minutes=500)
        assert should_exit is True
        assert "1.0%" in reason

    def test_roi_disabled(self):
        settings = FakeSettings()
        settings.roi_enabled = False
        mgr = StopManager(settings)
        should_exit, _ = mgr.check_roi_exit(pnl_pct=10.0, position_age_minutes=500)
        assert should_exit is False


# ---------------------------------------------------------------------------
# check_partial_exit
# ---------------------------------------------------------------------------


class TestCheckPartialExit:
    def setup_method(self):
        self.mgr = StopManager(FakeSettings())

    def _state(self, **kwargs) -> PositionState:
        defaults = {
            "ticker": "BTC/USDT",
            "entry_price": 100.0,
            "high_watermark": 100.0,
            "stop_mode": StopMode.TRAILING,
            "current_sl": 95.0,
            "partial_exits": 0,
        }
        defaults.update(kwargs)
        return PositionState(**defaults)

    def test_triggers_at_threshold(self):
        state = self._state(entry_price=100.0)
        # profit_in_atr = (110 - 100) / 5 = 2.0 >= 2.0 threshold
        should, pct, reason = self.mgr.check_partial_exit(state, current_price=110.0, atr=5.0)
        assert should is True
        assert pct == pytest.approx(0.50)
        assert "2.0x ATR" in reason

    def test_below_threshold_no_exit(self):
        state = self._state(entry_price=100.0)
        # profit_in_atr = (108 - 100) / 5 = 1.6 < 2.0
        should, pct, _ = self.mgr.check_partial_exit(state, current_price=108.0, atr=5.0)
        assert should is False
        assert pct == 0.0

    def test_blocked_after_first_partial(self):
        state = self._state(entry_price=100.0, partial_exits=1)
        should, _, _ = self.mgr.check_partial_exit(state, current_price=120.0, atr=5.0)
        assert should is False

    def test_disabled_in_settings(self):
        settings = FakeSettings()
        settings.partial_exit_enabled = False
        mgr = StopManager(settings)
        state = self._state()
        should, _, _ = mgr.check_partial_exit(state, current_price=120.0, atr=5.0)
        assert should is False

    def test_none_atr(self):
        state = self._state()
        should, _, _ = self.mgr.check_partial_exit(state, current_price=120.0, atr=None)
        assert should is False

    def test_zero_atr(self):
        state = self._state()
        should, _, _ = self.mgr.check_partial_exit(state, current_price=120.0, atr=0.0)
        assert should is False


# ---------------------------------------------------------------------------
# adjust (orchestrator)
# ---------------------------------------------------------------------------


class TestAdjust:
    def setup_method(self):
        self.mgr = StopManager(FakeSettings())

    def _state(self, **kwargs) -> PositionState:
        defaults = {
            "ticker": "BTC/USDT",
            "entry_price": 100.0,
            "high_watermark": 100.0,
            "stop_mode": StopMode.TRAILING_BREAKEVEN,
            "current_sl": 90.0,
            "current_tp": 120.0,
            "partial_exits": 0,
        }
        defaults.update(kwargs)
        return PositionState(**defaults)

    def test_roi_exit_takes_priority(self):
        """ROI exit fires before partial exit or trailing SL checks."""
        state = self._state(entry_price=100.0)
        # pnl 5% at 0 min => ROI tier 0:4.0 hit
        action = self.mgr.adjust(
            state,
            current_price=115.0,
            atr=5.0,
            pnl_pct=5.0,
            position_age_minutes=10,
        )
        assert action.action == "SELL"
        assert "ROI exit" in action.reason

    def test_partial_exit_after_roi_miss(self):
        """Partial exit fires when ROI doesn't trigger."""
        state = self._state(entry_price=100.0)
        # pnl 2% at 0 min < 4.0% ROI target, but profit_in_atr = 10/5 = 2.0 >= 2.0
        action = self.mgr.adjust(
            state,
            current_price=110.0,
            atr=5.0,
            pnl_pct=2.0,
            position_age_minutes=10,
        )
        assert action.action == "PARTIAL_SELL"
        assert action.sell_pct == pytest.approx(0.50)

    def test_trailing_sl_adjustment(self):
        """Trailing SL ratchets up on HOLD."""
        state = self._state(
            entry_price=100.0,
            high_watermark=108.0,
            current_sl=90.0,
            partial_exits=1,  # Block partial exit
        )
        # pnl 1% < 4% => no ROI. Partial blocked. price=108, atr=5
        # trailing_sl = max(108, 108) - 1.5*5 = 100.5 > 90 => updated
        action = self.mgr.adjust(
            state,
            current_price=108.0,
            atr=5.0,
            pnl_pct=1.0,
            position_age_minutes=10,
        )
        assert action.action == "HOLD"
        assert action.updated_sl == pytest.approx(100.5)

    def test_breakeven_lock_in_trailing_breakeven_mode(self):
        """Breakeven lock raises SL to entry price."""
        state = self._state(
            entry_price=100.0,
            high_watermark=100.0,
            current_sl=90.0,
            stop_mode=StopMode.TRAILING_BREAKEVEN,
            partial_exits=1,
        )
        # price=106, atr=5 => breakeven_trigger=100+1*5=105, 106>=105 => breakeven=100
        # trailing_sl = max(100, 106) - 1.5*5 = 106 - 7.5 = 98.5 > 90 => 98.5
        # breakeven = 100 => max(98.5, 100) = 100
        action = self.mgr.adjust(
            state,
            current_price=106.0,
            atr=5.0,
            pnl_pct=1.0,
            position_age_minutes=10,
        )
        assert action.action == "HOLD"
        assert action.updated_sl == pytest.approx(100.0)

    def test_stop_loss_triggered(self):
        """Price below SL triggers sell."""
        state = self._state(
            entry_price=100.0,
            high_watermark=100.0,
            current_sl=95.0,
            stop_mode=StopMode.FIXED,
            partial_exits=1,
        )
        # FIXED mode => no trailing adjustment. SL stays 95. price 94 <= 95 => SELL
        action = self.mgr.adjust(
            state,
            current_price=94.0,
            atr=5.0,
            pnl_pct=-6.0,
            position_age_minutes=10,
        )
        assert action.action == "SELL"
        assert "Stop-loss triggered" in action.reason

    def test_hold_when_nothing_triggers(self):
        """Normal hold with no triggers."""
        state = self._state(
            entry_price=100.0,
            high_watermark=100.0,
            current_sl=90.0,
            stop_mode=StopMode.FIXED,
            partial_exits=1,
        )
        # FIXED mode, price 102 > SL 90, pnl 2% < ROI 4%, partial blocked
        action = self.mgr.adjust(
            state,
            current_price=102.0,
            atr=5.0,
            pnl_pct=2.0,
            position_age_minutes=10,
        )
        assert action.action == "HOLD"
        assert action.updated_sl == pytest.approx(90.0)
        assert action.updated_tp == pytest.approx(120.0)

    def test_fixed_mode_does_not_trail(self):
        """FIXED stop mode should not adjust SL via trailing logic."""
        state = self._state(
            entry_price=100.0,
            high_watermark=120.0,
            current_sl=90.0,
            stop_mode=StopMode.FIXED,
            partial_exits=1,
        )
        action = self.mgr.adjust(
            state,
            current_price=118.0,
            atr=5.0,
            pnl_pct=1.0,
            position_age_minutes=10,
        )
        assert action.action == "HOLD"
        assert action.updated_sl == pytest.approx(90.0)  # Unchanged

    def test_trailing_mode_does_not_breakeven(self):
        """TRAILING mode (not TRAILING_BREAKEVEN) should not apply breakeven lock."""
        state = self._state(
            entry_price=100.0,
            high_watermark=100.0,
            current_sl=90.0,
            stop_mode=StopMode.TRAILING,
            partial_exits=1,
        )
        # price=106, atr=5 => trailing_sl = 106 - 7.5 = 98.5
        # breakeven would be 100 but TRAILING mode doesn't check it
        action = self.mgr.adjust(
            state,
            current_price=106.0,
            atr=5.0,
            pnl_pct=1.0,
            position_age_minutes=10,
        )
        assert action.action == "HOLD"
        assert action.updated_sl == pytest.approx(98.5)  # Not 100

    def test_roi_disabled_skips_roi(self):
        """When ROI is disabled, high PnL doesn't trigger ROI exit."""
        settings = FakeSettings()
        settings.roi_enabled = False
        mgr = StopManager(settings)
        state = self._state(
            entry_price=100.0,
            stop_mode=StopMode.FIXED,
            partial_exits=1,
        )
        action = mgr.adjust(
            state,
            current_price=110.0,
            atr=5.0,
            pnl_pct=10.0,
            position_age_minutes=500,
        )
        assert action.action == "HOLD"

    def test_none_atr_trailing_keeps_sl(self):
        """Trailing with None ATR should keep current SL and not crash."""
        state = self._state(
            entry_price=100.0,
            high_watermark=110.0,
            current_sl=95.0,
            stop_mode=StopMode.TRAILING,
            partial_exits=1,
        )
        action = self.mgr.adjust(
            state,
            current_price=108.0,
            atr=None,
            pnl_pct=1.0,
            position_age_minutes=10,
        )
        assert action.action == "HOLD"
        assert action.updated_sl == pytest.approx(95.0)
