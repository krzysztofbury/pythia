"""Tests for persistent LLM budget reservations and client integration."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from llm_budget import BudgetExceededError, LLMBudgetManager
from llm_client import LLMClient

if TYPE_CHECKING:
    from pathlib import Path


def _response(text: str = "ok", *, cost_usd: float = 0.003) -> dict[str, object]:
    return {
        "model": "provider/actual-model",
        "choices": [{"message": {"content": text}}],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 4,
            "total_tokens": 16,
            "prompt_tokens_details": {"cached_tokens": 3},
        },
        "_hidden_params": {"response_cost": cost_usd},
    }


def test_atomic_reservations_are_shared_across_managers(tmp_path: Path) -> None:
    """Do not let concurrent processes reserve beyond a shared cap."""
    path = tmp_path / "budget.db"
    managers = [LLMBudgetManager(path, daily_cap_usd=0.05) for _ in range(2)]

    def reserve(index: int) -> bool:
        try:
            managers[index % 2].reserve(0.01)
        except BudgetExceededError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=10) as pool:
        accepted = list(pool.map(reserve, range(10)))

    expected_accepted = 5
    assert sum(accepted) == expected_accepted
    assert managers[0].snapshot().daily_usd == pytest.approx(0.05)


def test_finalize_replaces_reservation_with_actual_cost(tmp_path: Path) -> None:
    """Release unused reservation value once actual usage is known."""
    manager = LLMBudgetManager(tmp_path / "budget.db", monthly_cap_usd=0.10)
    reservation = manager.reserve(0.08)
    manager.finalize(reservation, 0.02)

    manager.reserve(0.08)

    assert manager.snapshot().monthly_usd == pytest.approx(0.10)


def test_shadow_cap_is_monthly_and_additive(tmp_path: Path) -> None:
    """Apply the shadow cap in addition to normal daily and monthly caps."""
    now = datetime(2026, 7, 24, 12, tzinfo=UTC)
    manager = LLMBudgetManager(
        tmp_path / "budget.db",
        daily_cap_usd=0.10,
        monthly_cap_usd=0.20,
        shadow_cap_usd=0.03,
        clock=lambda: now,
    )
    manager.reserve(0.02, shadow=True)
    manager.reserve(0.03)

    with pytest.raises(BudgetExceededError, match="shadow"):
        manager.reserve(0.011, shadow=True)

    snapshot = manager.snapshot()
    assert snapshot.daily_usd == pytest.approx(0.05)
    assert snapshot.monthly_usd == pytest.approx(0.05)
    assert snapshot.shadow_monthly_usd == pytest.approx(0.02)


def test_zero_cap_blocks_provider_before_call(tmp_path: Path) -> None:
    """Interpret a zero cap as permission for no positive-cost calls."""
    manager = LLMBudgetManager(tmp_path / "budget.db", monthly_cap_usd=0)
    client = LLMClient(
        "provider/primary",
        budget_manager=manager,
        cost_estimator=lambda _model, _messages, _max_tokens: 0.01,
    )

    with patch("litellm.completion") as completion, pytest.raises(BudgetExceededError):
        client.chat("system", "user")

    completion.assert_not_called()


def test_reservation_covers_all_retry_and_fallback_attempts(tmp_path: Path) -> None:
    """Reserve two bounded attempts for both primary and fallback routes."""
    manager = LLMBudgetManager(tmp_path / "budget.db", monthly_cap_usd=0.079)
    client = LLMClient(
        "provider/primary",
        fallback_model="provider/fallback",
        budget_manager=manager,
        cost_estimator=lambda _model, _messages, _max_tokens: 0.02,
    )

    with patch("litellm.completion") as completion, pytest.raises(BudgetExceededError):
        client.chat("system", "user")

    completion.assert_not_called()
    assert manager.snapshot().monthly_usd == 0


def test_result_metadata_and_actual_cost_finalize(tmp_path: Path) -> None:
    """Expose provider metadata while charging provider-reported actual cost."""
    manager = LLMBudgetManager(tmp_path / "budget.db", monthly_cap_usd=0.08)
    client = LLMClient(
        "provider/primary",
        fallback_model="provider/fallback",
        budget_manager=manager,
        shadow=True,
        cost_estimator=lambda _model, _messages, _max_tokens: 0.02,
    )

    with patch("litellm.completion", return_value=_response("answer")):
        result = client.chat_result("system", "user")

    assert result.text == "answer"
    assert result.provider == "provider"
    assert result.model == "provider/actual-model"
    expected_prompt_tokens = 12
    assert result.usage.prompt_tokens == expected_prompt_tokens
    assert result.usage.cost_usd == pytest.approx(0.003)
    assert result.cache_hit is True
    assert result.used_fallback is False
    assert result.latency_ms >= 0
    assert client.last_result == result
    assert manager.snapshot().shadow_monthly_usd == pytest.approx(0.003)


def test_crash_orphaned_reservations_expire_on_startup(tmp_path: Path) -> None:
    """Release reservations a SIGKILL orphaned so caps cannot brick silently."""
    path = tmp_path / "budget.db"
    crash_time = datetime(2026, 7, 24, 10, 0, 0, tzinfo=UTC)
    crashed = LLMBudgetManager(path, daily_cap_usd=0.05, clock=lambda: crash_time)
    crashed.reserve(0.05)  # never finalized: process killed mid-call

    restart_time = datetime(2026, 7, 24, 12, 0, 0, tzinfo=UTC)
    restarted = LLMBudgetManager(path, daily_cap_usd=0.05, clock=lambda: restart_time)

    assert restarted.snapshot().daily_usd == pytest.approx(0.0)
    restarted.reserve(0.05)


def test_recent_active_reservations_survive_startup(tmp_path: Path) -> None:
    """Never expire a reservation an in-flight call may still finalize."""
    path = tmp_path / "budget.db"
    now = datetime(2026, 7, 24, 10, 0, 0, tzinfo=UTC)
    first = LLMBudgetManager(path, daily_cap_usd=0.05, clock=lambda: now)
    reservation = first.reserve(0.05)

    second = LLMBudgetManager(path, daily_cap_usd=0.05, clock=lambda: now)

    assert second.snapshot().daily_usd == pytest.approx(0.05)
    with pytest.raises(BudgetExceededError):
        second.reserve(0.01)
    first.finalize(reservation, 0.01)
