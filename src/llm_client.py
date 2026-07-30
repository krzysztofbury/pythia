"""Provider-agnostic LLM chat client with resilient fallback and budgeting."""

from __future__ import annotations

import logging
import math
import re
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import litellm
import litellm.exceptions as litellm_exceptions

if TYPE_CHECKING:
    from collections.abc import Callable

    from llm_budget import BudgetReservation, LLMBudgetManager

logger = logging.getLogger(__name__)

_MAX_CONSECUTIVE_FAILURES = 3
_MAX_PROVIDER_ATTEMPTS = 2
_CIRCUIT_OPEN_COOLDOWN_SECONDS = 300.0
_PERMANENT_ERROR_COOLDOWN_SECONDS = 3_600.0
_RETRY_DELAYS_SECONDS = (2.0,)

_SECRET_PATTERNS = [
    re.compile(r"(sk-[A-Za-z0-9_\-]{10,})"),
    re.compile(r"(xai-[A-Za-z0-9_\-]{10,})"),
    re.compile(r"(AIza[A-Za-z0-9_\-]{10,})"),
    re.compile(r"(Bearer\s+[A-Za-z0-9_\-\.]{10,})", re.IGNORECASE),
    re.compile(
        r"((?:api[-_ ]?key|authorization|token|secret)\s*[:=]\s*)[^\s,;]+",
        re.IGNORECASE,
    ),
]
_PERMANENT_ERROR_MARKERS = (
    "invalid_api_key",
    "invalid api key",
    "incorrect api key",
    "authentication",
    "permission denied",
    "access denied",
    "insufficient_quota",
    "quota",
    "quota exceeded",
    "billing hard limit",
    "model_not_found",
    "model not found",
    "invalid model",
    "unsupported model",
    "unknown model",
    "does not exist",
)


def _sanitize(text: str) -> str:
    """Redact common credentials before text reaches logs or callers."""
    redacted = text
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def _exception_types(*names: str) -> tuple[type[Exception], ...]:
    classes: list[type[Exception]] = []
    for name in names:
        candidate = getattr(litellm_exceptions, name, None)
        if isinstance(candidate, type) and issubclass(candidate, Exception):
            classes.append(candidate)
    return tuple(classes)


_TRANSIENT_EXCEPTIONS: tuple[type[Exception], ...] = (
    TimeoutError,
    ConnectionError,
    *_exception_types(
        "Timeout",
        "APIConnectionError",
        "RateLimitError",
        "ServiceUnavailableError",
        "InternalServerError",
    ),
)
_PERMANENT_EXCEPTIONS: tuple[type[Exception], ...] = _exception_types(
    "AuthenticationError",
    "PermissionDeniedError",
    "NotFoundError",
)


class CircuitState(str, Enum):
    """Represent a provider route's circuit-breaker state."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Allow one probe after cooldown and reject concurrent probes."""

    def __init__(
        self,
        name: str,
        *,
        failure_threshold: int = _MAX_CONSECUTIVE_FAILURES,
        cooldown_seconds: float = _CIRCUIT_OPEN_COOLDOWN_SECONDS,
        permanent_cooldown_seconds: float = _PERMANENT_ERROR_COOLDOWN_SECONDS,
        clock: Callable[[], float] | None = None,
    ) -> None:
        """Initialize an independent breaker for one configured route."""
        if failure_threshold < 1:
            msg = "failure_threshold must be >= 1"
            raise ValueError(msg)
        if cooldown_seconds < 0 or permanent_cooldown_seconds < 0:
            msg = "circuit cooldowns must be >= 0"
            raise ValueError(msg)
        self.name = name
        self._failure_threshold = failure_threshold
        self._default_cooldown = cooldown_seconds
        self._permanent_cooldown = permanent_cooldown_seconds
        self._clock = clock
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._opened_at = 0.0
        self._active_cooldown = cooldown_seconds
        self._probe_in_flight = False
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        """Return the current state, transitioning to half-open after cooldown."""
        with self._lock:
            self._refresh_state()
            return self._state

    @property
    def failure_count(self) -> int:
        """Return the number of consecutive failed logical calls."""
        with self._lock:
            return self._failure_count

    @property
    def opened_at(self) -> float:
        """Return the monotonic timestamp at which the circuit opened."""
        with self._lock:
            return self._opened_at

    def is_open(self) -> bool:
        """Report whether a new request would currently be rejected."""
        with self._lock:
            self._refresh_state()
            if self._state is CircuitState.OPEN:
                return True
            return self._state is CircuitState.HALF_OPEN and self._probe_in_flight

    def try_acquire(self) -> bool:
        """Acquire permission for a call, claiming the sole half-open probe."""
        with self._lock:
            self._refresh_state()
            if self._state is CircuitState.OPEN:
                return False
            if self._state is CircuitState.HALF_OPEN:
                if self._probe_in_flight:
                    return False
                self._probe_in_flight = True
            return True

    def record_success(self) -> None:
        """Close the circuit after a successful request or probe."""
        with self._lock:
            was_degraded = self._state is not CircuitState.CLOSED or self._failure_count > 0
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._opened_at = 0.0
            self._active_cooldown = self._default_cooldown
            self._probe_in_flight = False
        if was_degraded:
            logger.info("LLM circuit breaker CLOSED for %s after success.", self.name)

    def record_failure(self, *, permanent: bool = False) -> None:
        """Record failure and open immediately for a permanent provider error."""
        with self._lock:
            was_probe = self._state is CircuitState.HALF_OPEN
            self._failure_count += 1
            should_open = permanent or was_probe or self._failure_count >= self._failure_threshold
            if not should_open:
                return
            self._state = CircuitState.OPEN
            self._opened_at = self._now()
            self._active_cooldown = (
                self._permanent_cooldown if permanent else self._default_cooldown
            )
            self._probe_in_flight = False
            failures = self._failure_count
            cooldown = self._active_cooldown
        logger.warning(
            "LLM circuit breaker OPEN for %s after %d failures; cooldown %.0fs.",
            self.name,
            failures,
            cooldown,
        )

    def set_failure_count(self, value: int) -> None:
        """Set compatibility state for legacy tests and diagnostics."""
        if value < 0:
            msg = "failure count must be >= 0"
            raise ValueError(msg)
        with self._lock:
            self._failure_count = value
            if value >= self._failure_threshold:
                self._state = CircuitState.OPEN
                self._opened_at = self._now()
                self._probe_in_flight = False
            elif self._state is not CircuitState.CLOSED:
                self._state = CircuitState.CLOSED
                self._opened_at = 0.0
                self._probe_in_flight = False

    def set_opened_at(self, value: float) -> None:
        """Set the compatibility open timestamp without changing the threshold."""
        with self._lock:
            self._opened_at = value
            if self._failure_count >= self._failure_threshold:
                self._state = CircuitState.OPEN
                self._probe_in_flight = False

    def _refresh_state(self) -> None:
        if self._state is not CircuitState.OPEN:
            return
        if self._now() - self._opened_at < self._active_cooldown:
            return
        self._state = CircuitState.HALF_OPEN
        self._probe_in_flight = False
        logger.info("LLM circuit breaker HALF_OPEN for %s; one probe allowed.", self.name)

    def _now(self) -> float:
        return self._clock() if self._clock is not None else time.monotonic()


@dataclass(frozen=True, slots=True)
class LLMUsage:
    """Describe token and provider-reported cost metadata."""

    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    cost_usd: float | None


@dataclass(frozen=True, slots=True)
class LLMResult:
    """Return assistant text together with operational metadata."""

    text: str
    provider: str | None
    model: str
    usage: LLMUsage
    latency_ms: float
    cache_hit: bool
    used_fallback: bool

    @property
    def latency_seconds(self) -> float:
        """Expose latency in seconds for metrics systems that use SI units."""
        return self.latency_ms / 1_000.0

    @property
    def fallback_used(self) -> bool:
        """Expose a natural-language alias for ``used_fallback``."""
        return self.used_fallback


@dataclass(frozen=True, slots=True)
class _ProviderOutcome:
    result: LLMResult | None
    error: Exception | None
    attempts: int


class _ChatError(RuntimeError):
    def __init__(self, message: str, estimated_cost_usd: float) -> None:
        """Carry conservative cost through a failed chat execution."""
        super().__init__(message)
        self.estimated_cost_usd = estimated_cost_usd


class LLMClient:
    """Wrap ``litellm.completion`` with retries, independent breakers, and fallback."""

    def __init__(  # noqa: PLR0913 - preserves the existing constructor and adds opt-in hooks.
        self,
        primary_model: str,
        fallback_model: str | None = None,
        api_keys: dict[str, str] | None = None,
        *,
        max_tokens: int = 512,
        timeout: float = 30.0,
        budget_manager: LLMBudgetManager | None = None,
        shadow: bool = False,
        cost_estimator: Callable[[str, list[dict[str, str]], int], float] | None = None,
    ) -> None:
        """Store configuration without making a provider request."""
        assert primary_model, "primary_model must not be empty"  # noqa: S101
        if max_tokens < 1:
            msg = "max_tokens must be >= 1"
            raise ValueError(msg)
        if timeout <= 0:
            msg = "timeout must be > 0"
            raise ValueError(msg)
        self._primary = primary_model
        self._fallback = fallback_model
        self._api_keys = api_keys or {}
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._budget = budget_manager
        self._shadow = shadow
        self._cost_estimator = cost_estimator
        self._primary_breaker = CircuitBreaker(primary_model)
        self._fallback_breaker = CircuitBreaker(fallback_model or "fallback")
        self._result_local = threading.local()

    @property
    def last_result(self) -> LLMResult | None:
        """Return metadata for this thread's most recent successful chat."""
        return getattr(self._result_local, "last_result", None)

    @property
    def primary_circuit_state(self) -> CircuitState:
        """Return the primary route's current breaker state."""
        return self._primary_breaker.state

    @property
    def fallback_circuit_state(self) -> CircuitState:
        """Return the fallback route's current breaker state."""
        return self._fallback_breaker.state

    @property
    def _consecutive_failures(self) -> int:
        """Preserve the old primary-breaker diagnostic attribute."""
        return self._primary_breaker.failure_count

    @_consecutive_failures.setter
    def _consecutive_failures(self, value: int) -> None:
        self._primary_breaker.set_failure_count(value)

    @property
    def _opened_at(self) -> float:
        """Preserve the old primary-breaker diagnostic attribute."""
        return self._primary_breaker.opened_at

    @_opened_at.setter
    def _opened_at(self, value: float) -> None:
        self._primary_breaker.set_opened_at(value)

    def is_circuit_open(self) -> bool:
        """Report whether the primary route currently rejects a new request."""
        return self._primary_breaker.is_open()

    _is_circuit_open = is_circuit_open

    def _record_success(self) -> None:
        """Preserve the legacy primary success hook."""
        self._primary_breaker.record_success()

    def _record_failure(self) -> None:
        """Preserve the legacy primary failure hook."""
        self._primary_breaker.record_failure()

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """Send a chat request and preserve the existing string return type."""
        return self.chat_result(system_prompt, user_prompt).text

    def chat_result(self, system_prompt: str, user_prompt: str) -> LLMResult:
        """Send a chat request and return text plus usage and routing metadata."""
        assert system_prompt, "system_prompt must not be empty"  # noqa: S101
        assert user_prompt, "user_prompt must not be empty"  # noqa: S101
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        reservation, route_costs = self._reserve_budget(messages)
        try:
            result, actual_cost = self._execute_chat(messages, route_costs)
        except _ChatError as exc:
            self._finalize_budget(reservation, exc.estimated_cost_usd)
            raise RuntimeError(str(exc)) from None
        self._finalize_budget(reservation, actual_cost)
        self._result_local.last_result = result
        return result

    def _execute_chat(
        self,
        messages: list[dict[str, str]],
        route_costs: dict[str, float],
    ) -> tuple[LLMResult, float]:
        actual_cost = 0.0
        primary_error: Exception | None = None
        if self._primary_breaker.try_acquire():
            outcome = self._call_provider(self._primary, messages, used_fallback=False)
            actual_cost += self._outcome_cost(outcome, route_costs.get("primary", 0.0))
            if outcome.result is not None:
                self._primary_breaker.record_success()
                return outcome.result, actual_cost
            primary_error = outcome.error
            self._primary_breaker.record_failure(permanent=_is_permanent_error(primary_error))
            suffix = " Trying fallback." if self._fallback else ""
            logger.warning(
                "LLM primary model %s failed: %s.%s",
                self._primary,
                _sanitize(str(primary_error)),
                suffix,
            )

        if not self._fallback:
            detail = f": {_sanitize(str(primary_error))}" if primary_error else ""
            message = (
                f"LLM primary model {self._primary} unavailable{detail}; no fallback configured."
            )
            raise _ChatError(message, actual_cost)
        if not self._fallback_breaker.try_acquire():
            message = "Primary and fallback LLM circuits are unavailable."
            raise _ChatError(message, actual_cost)

        outcome = self._call_provider(self._fallback, messages, used_fallback=True)
        actual_cost += self._outcome_cost(outcome, route_costs.get("fallback", 0.0))
        if outcome.result is not None:
            self._fallback_breaker.record_success()
            return outcome.result, actual_cost
        fallback_error = outcome.error
        self._fallback_breaker.record_failure(permanent=_is_permanent_error(fallback_error))
        logger.warning(
            "LLM fallback model %s failed: %s.",
            self._fallback,
            _sanitize(str(fallback_error)),
        )
        safe_primary = _sanitize(str(primary_error)) if primary_error else "primary unavailable"
        safe_fallback = _sanitize(str(fallback_error))
        message = f"Both primary and fallback LLM calls failed: {safe_primary}; {safe_fallback}"
        raise _ChatError(message, actual_cost)

    def _call_provider(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        used_fallback: bool,
    ) -> _ProviderOutcome:
        started_at = time.monotonic()
        for attempt in range(1, _MAX_PROVIDER_ATTEMPTS + 1):
            try:
                response = self._completion(model, messages)
                latency_ms = (time.monotonic() - started_at) * 1_000.0
                result = _decode_response(
                    response,
                    model,
                    latency_ms,
                    used_fallback=used_fallback,
                )
                return _ProviderOutcome(result=result, error=None, attempts=attempt)
            except Exception as exc:  # noqa: BLE001, PERF203 - provider SDK errors vary.
                should_retry = (
                    attempt < _MAX_PROVIDER_ATTEMPTS
                    and _is_retryable_error(exc)
                    and not _is_permanent_error(exc)
                )
                if not should_retry:
                    return _ProviderOutcome(result=None, error=exc, attempts=attempt)
                logger.warning(
                    "Transient LLM error from %s on attempt %d/%d: %s.",
                    model,
                    attempt,
                    _MAX_PROVIDER_ATTEMPTS,
                    _sanitize(str(exc)),
                )
                time.sleep(_RETRY_DELAYS_SECONDS[attempt - 1])
        msg = "bounded LLM retry loop exhausted unexpectedly"
        raise AssertionError(msg)

    def _completion(self, model: str, messages: list[dict[str, str]]) -> object:
        litellm.suppress_debug_info = True
        logging.getLogger("LiteLLM").setLevel(logging.WARNING)
        kwargs: dict[str, object] = {
            "model": model,
            "messages": messages,
            "max_tokens": self._max_tokens,
            "timeout": self._timeout,
        }
        provider = _provider_for_model(model)
        provider_key_map = {
            "xai": "xai_api_key",
            "anthropic": "anthropic_api_key",
            "gemini": "gemini_api_key",
            "openai": "openai_api_key",
        }
        if provider in provider_key_map and provider in self._api_keys:
            kwargs[provider_key_map[provider]] = self._api_keys[provider]
        return litellm.completion(**kwargs)

    def _call(self, model: str, messages: list[dict[str, str]]) -> str:
        """Preserve the old private helper while keeping retries bounded."""
        outcome = self._call_provider(model, messages, used_fallback=model == self._fallback)
        if outcome.result is not None:
            return outcome.result.text
        if outcome.error is None:
            msg = "LLM provider failed without an exception."
            raise RuntimeError(msg)
        raise outcome.error

    def _reserve_budget(
        self,
        messages: list[dict[str, str]],
    ) -> tuple[BudgetReservation | None, dict[str, float]]:
        if self._budget is None:
            return None, {}
        route_costs = {"primary": self._estimate_attempt_cost(self._primary, messages)}
        if self._fallback:
            route_costs["fallback"] = self._estimate_attempt_cost(self._fallback, messages)
        worst_case = sum(route_costs.values()) * _MAX_PROVIDER_ATTEMPTS
        reservation = self._budget.reserve(worst_case, shadow=self._shadow)
        return reservation, route_costs

    def _estimate_attempt_cost(self, model: str, messages: list[dict[str, str]]) -> float:
        try:
            if self._cost_estimator is not None:
                estimated = float(self._cost_estimator(model, messages, self._max_tokens))
            else:
                prompt_tokens = litellm.token_counter(model=model, messages=messages)
                input_cost, output_cost = litellm.cost_per_token(
                    model=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=self._max_tokens,
                )
                estimated = float(input_cost + output_cost)
        except Exception as exc:  # noqa: BLE001 - fail closed on unknown provider pricing.
            safe = _sanitize(str(exc))
            message = f"Cannot safely estimate LLM cost for {model}: {safe}"
            raise RuntimeError(message) from None
        if not math.isfinite(estimated) or estimated < 0:
            message = f"Invalid LLM cost estimate for {model}."
            raise RuntimeError(message)
        return estimated

    def _finalize_budget(
        self,
        reservation: BudgetReservation | None,
        actual_cost_usd: float,
    ) -> None:
        if self._budget is not None and reservation is not None:
            self._budget.finalize(reservation, actual_cost_usd)

    @staticmethod
    def _outcome_cost(outcome: _ProviderOutcome, estimated_attempt_cost: float) -> float:
        failed_attempts = outcome.attempts if outcome.result is None else outcome.attempts - 1
        cost = failed_attempts * estimated_attempt_cost
        if outcome.result is None:
            return cost
        reported_cost = outcome.result.usage.cost_usd
        return cost + (estimated_attempt_cost if reported_cost is None else reported_cost)


def _decode_response(
    response: object,
    requested_model: str,
    latency_ms: float,
    *,
    used_fallback: bool,
) -> LLMResult:
    choices = _field(response, "choices") or []
    if not isinstance(choices, Sequence) or isinstance(choices, str) or not choices:
        msg = "LLM response had no choices."
        raise RuntimeError(msg)
    message = _field(choices[0], "message") or {}
    content = _field(message, "content")
    if not content or not isinstance(content, str):
        msg = "LLM response content was missing or non-string."
        raise RuntimeError(msg)

    response_model = _field(response, "model") or requested_model
    hidden = _field(response, "_hidden_params") or {}
    usage_raw = _field(response, "usage") or {}
    usage = LLMUsage(
        prompt_tokens=_optional_int(_field(usage_raw, "prompt_tokens")),
        completion_tokens=_optional_int(_field(usage_raw, "completion_tokens")),
        total_tokens=_optional_int(_field(usage_raw, "total_tokens")),
        cost_usd=_response_cost(response, requested_model, usage_raw, hidden),
    )
    provider_value = _field(hidden, "custom_llm_provider")
    provider = str(provider_value) if provider_value else _provider_for_model(requested_model)
    return LLMResult(
        text=content.strip(),
        provider=provider,
        model=str(response_model),
        usage=usage,
        latency_ms=latency_ms,
        cache_hit=_cache_hit(usage_raw, hidden),
        used_fallback=used_fallback,
    )


def _response_cost(
    response: object,
    model: str,
    usage: object,
    hidden: object,
) -> float | None:
    for candidate in (
        _field(hidden, "response_cost"),
        _field(response, "response_cost"),
        _field(usage, "cost"),
    ):
        parsed = _optional_float(candidate)
        if parsed is not None:
            return parsed
    token_values = (
        _field(usage, "prompt_tokens"),
        _field(usage, "completion_tokens"),
        _field(usage, "total_tokens"),
    )
    if all(value is None for value in token_values):
        return None
    try:
        calculated = float(litellm.completion_cost(completion_response=response, model=model))
    except Exception:  # noqa: BLE001 - missing cost metadata is valid for custom responses.
        return None
    return calculated if math.isfinite(calculated) and calculated >= 0 else None


def _cache_hit(usage: object, hidden: object) -> bool:
    hidden_hit = _field(hidden, "cache_hit")
    if isinstance(hidden_hit, bool):
        return hidden_hit
    details = _field(usage, "prompt_tokens_details") or {}
    cached_tokens = _optional_int(_field(details, "cached_tokens")) or 0
    cache_read_tokens = _optional_int(_field(usage, "cache_read_input_tokens")) or 0
    return cached_tokens > 0 or cache_read_tokens > 0


def _field(value: object, name: str) -> object | None:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed >= 0 else None


def _provider_for_model(model: str) -> str | None:
    return model.split("/", 1)[0] if "/" in model else None


def _is_retryable_error(exc: Exception | None) -> bool:
    return isinstance(exc, _TRANSIENT_EXCEPTIONS)


def _is_permanent_error(exc: Exception | None) -> bool:
    if exc is None:
        return False
    if isinstance(exc, _PERMANENT_EXCEPTIONS):
        return True
    status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
    if str(status) in {"401", "403", "404"}:
        return True
    message = str(exc).lower()
    return any(marker in message for marker in _PERMANENT_ERROR_MARKERS)
