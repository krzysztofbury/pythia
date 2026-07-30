"""Concurrency and failure-classification tests for LLM circuit breakers."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

import llm_client
from llm_client import CircuitBreaker, CircuitState, LLMClient

if TYPE_CHECKING:
    from _pytest.logging import LogCaptureFixture


def _response(text: str) -> dict[str, object]:
    return {
        "choices": [{"message": {"content": text}}],
        "_hidden_params": {"response_cost": 0.001},
    }


def test_half_open_allows_exactly_one_concurrent_probe() -> None:
    """Reject every concurrent request except the first half-open probe."""
    now = [0.0]
    breaker = CircuitBreaker(
        "primary",
        failure_threshold=1,
        cooldown_seconds=10,
        clock=lambda: now[0],
    )
    breaker.record_failure()
    now[0] = 11.0
    barrier = threading.Barrier(12)

    def acquire_probe(_index: int) -> bool:
        barrier.wait()
        return breaker.try_acquire()

    with ThreadPoolExecutor(max_workers=12) as pool:
        acquired = list(pool.map(acquire_probe, range(12)))

    assert sum(acquired) == 1
    assert breaker.state is CircuitState.HALF_OPEN
    assert breaker.is_open() is True


def test_failed_probe_restarts_the_full_cooldown() -> None:
    """Start cooldown from the failed probe rather than the original outage."""
    now = [0.0]
    breaker = CircuitBreaker(
        "primary",
        failure_threshold=1,
        cooldown_seconds=10,
        clock=lambda: now[0],
    )
    breaker.record_failure()
    now[0] = 11.0
    assert breaker.try_acquire() is True

    breaker.record_failure()

    now[0] = 20.9
    assert breaker.try_acquire() is False
    now[0] = 21.0
    assert breaker.try_acquire() is True


def test_client_allows_only_one_primary_probe_under_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Route concurrent cooldown traffic to fallback while one primary probes."""
    client = LLMClient("provider/primary", fallback_model="provider/fallback")
    client._consecutive_failures = 3  # noqa: SLF001
    client._opened_at = 0.0  # noqa: SLF001
    monkeypatch.setattr(
        llm_client.time,
        "monotonic",
        lambda: llm_client._CIRCUIT_OPEN_COOLDOWN_SECONDS + 1,  # noqa: SLF001
    )
    worker_count = 6
    fallback_worker_count = worker_count - 1
    start = threading.Barrier(worker_count)
    release_probe = threading.Event()
    fallbacks_finished = threading.Event()
    calls: list[str] = []
    calls_lock = threading.Lock()

    def completion(**kwargs: object) -> dict[str, object]:
        model = str(kwargs["model"])
        with calls_lock:
            calls.append(model)
            fallback_count = calls.count("provider/fallback")
        if model == "provider/primary":
            assert release_probe.wait(timeout=5)
            return _response("primary")
        if fallback_count == fallback_worker_count:
            fallbacks_finished.set()
        return _response("fallback")

    def chat(_index: int) -> str:
        start.wait()
        return client.chat("system", "user")

    with (
        patch("litellm.completion", side_effect=completion),
        ThreadPoolExecutor(max_workers=worker_count) as pool,
    ):
        futures = [pool.submit(chat, index) for index in range(worker_count)]
        assert fallbacks_finished.wait(timeout=5)
        release_probe.set()
        results = [future.result(timeout=5) for future in futures]

    assert calls.count("provider/primary") == 1
    assert calls.count("provider/fallback") == fallback_worker_count
    assert sorted(results) == ["fallback"] * fallback_worker_count + ["primary"]


def test_primary_and_fallback_breakers_are_independent() -> None:
    """Keep a healthy primary usable when a permanent fallback failure opens."""
    client = LLMClient("provider/primary", fallback_model="provider/fallback")
    primary_error = RuntimeError("temporary primary failure")
    fallback_error = RuntimeError("Invalid API key: sk-fallback-SECRET123456")

    def fail_both(**kwargs: object) -> dict[str, object]:
        if kwargs["model"] == "provider/primary":
            raise primary_error
        raise fallback_error

    with patch("litellm.completion", side_effect=fail_both), pytest.raises(RuntimeError):
        client.chat("system", "user")

    assert client.primary_circuit_state is CircuitState.CLOSED
    assert client.fallback_circuit_state is CircuitState.OPEN

    with patch("litellm.completion", return_value=_response("recovered")) as completion:
        assert client.chat("system", "user") == "recovered"
    assert completion.call_args.kwargs["model"] == "provider/primary"


def test_auth_error_opens_immediately_for_long_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Avoid repeated paid attempts for permanent authentication failures."""
    now = [100.0]
    monkeypatch.setattr(llm_client.time, "monotonic", lambda: now[0])
    client = LLMClient("provider/primary")

    auth_error = RuntimeError("Invalid API key: sk-primary-SECRET123456")
    with (
        patch("litellm.completion", side_effect=auth_error),
        pytest.raises(RuntimeError),
    ):
        client.chat("system", "user")

    assert client.primary_circuit_state is CircuitState.OPEN
    now[0] += llm_client._CIRCUIT_OPEN_COOLDOWN_SECONDS + 1  # noqa: SLF001
    assert client.is_circuit_open() is True
    now[0] = 100.0 + llm_client._PERMANENT_ERROR_COOLDOWN_SECONDS  # noqa: SLF001
    assert client.is_circuit_open() is False


def test_transient_retries_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retry transient failures no more than the fixed provider-attempt bound."""
    client = LLMClient("provider/primary")
    monkeypatch.setattr(llm_client.time, "sleep", lambda _seconds: None)

    timeout_error = TimeoutError("provider timed out")
    with (
        patch("litellm.completion", side_effect=timeout_error) as call,
        pytest.raises(RuntimeError),
    ):
        client.chat("system", "user")

    assert call.call_count == llm_client._MAX_PROVIDER_ATTEMPTS  # noqa: SLF001


def test_provider_errors_are_redacted_in_all_logs(caplog: LogCaptureFixture) -> None:
    """Never expose primary or fallback credentials through log records."""
    client = LLMClient("provider/primary", fallback_model="provider/fallback")
    primary_error = RuntimeError("Bearer primarySECRET123456789")
    fallback_error = RuntimeError("API key: sk-fallback-SECRET123456")

    def fail(**kwargs: object) -> dict[str, object]:
        if kwargs["model"] == "provider/primary":
            raise primary_error
        raise fallback_error

    with (
        caplog.at_level(logging.WARNING),
        patch("litellm.completion", side_effect=fail),
        pytest.raises(RuntimeError),
    ):
        client.chat("system", "user")

    assert "primarySECRET" not in caplog.text
    assert "sk-fallback" not in caplog.text
    assert "[REDACTED]" in caplog.text
