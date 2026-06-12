"""Provider-agnostic LLM chat client with circuit breaker and fallback model."""

import logging
import re
import threading
import time

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

_MAX_CONSECUTIVE_FAILURES = 3
_CIRCUIT_OPEN_COOLDOWN_SECONDS = 300.0  # 5 minutes — allow a probe after cooldown

# Providers sometimes echo the API key back in auth-error messages. Strip
# anything that looks like a key before logging the exception, so secrets
# never hit rotating log files or Discord webhooks.
_SECRET_PATTERNS = [
    re.compile(r"(sk-[A-Za-z0-9_\-]{10,})"),
    re.compile(r"(xai-[A-Za-z0-9_\-]{10,})"),
    re.compile(r"(AIza[A-Za-z0-9_\-]{10,})"),
    re.compile(r"(Bearer\s+[A-Za-z0-9_\-\.]{10,})", re.IGNORECASE),
]


def _sanitize(text: str) -> str:
    """Redact anything that looks like an API key from an error string."""
    redacted = text
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def _retryable_exceptions() -> tuple[type[Exception], ...]:
    """Transient-error exception classes that warrant a tenacity retry.

    Resolved lazily so litellm's exception classes are only imported when
    available; falls back to stdlib ``TimeoutError`` / ``ConnectionError``
    if litellm is not installed (e.g. during isolated unit tests).
    """
    excs: list[type[Exception]] = [TimeoutError, ConnectionError]
    try:
        import litellm.exceptions as le

        for name in ("Timeout", "APIConnectionError", "RateLimitError", "ServiceUnavailableError"):
            cls = getattr(le, name, None)
            if isinstance(cls, type) and issubclass(cls, BaseException):
                excs.append(cls)
    except ImportError:
        pass
    return tuple(excs)


class LLMClient:
    """Thin wrapper over ``litellm.completion`` with retries, circuit breaker, fallback.

    The public surface is a single ``chat(system, user)`` method that returns
    the assistant's text content. On primary-model failure the client falls
    through to ``fallback_model`` if one is configured. After
    ``_MAX_CONSECUTIVE_FAILURES`` primary failures the circuit breaker opens
    and primary is skipped; after ``_CIRCUIT_OPEN_COOLDOWN_SECONDS`` the
    next call probes the primary again, resetting on success.
    """

    def __init__(
        self,
        primary_model: str,
        fallback_model: str | None = None,
        api_keys: dict[str, str] | None = None,
        *,
        max_tokens: int = 512,
        timeout: float = 30.0,
    ) -> None:
        """Store config; defer litellm import so tests can monkeypatch."""
        assert primary_model, "primary_model must not be empty"  # noqa: S101
        self._primary = primary_model
        self._fallback = fallback_model
        self._api_keys = api_keys or {}
        self._max_tokens = max_tokens
        self._timeout = timeout

        self._consecutive_failures = 0
        self._opened_at: float = 0.0
        self._cb_lock = threading.Lock()

    def is_circuit_open(self) -> bool:
        """True when the primary model should be skipped this call.

        Auto-closes after ``_CIRCUIT_OPEN_COOLDOWN_SECONDS`` so a single
        provider outage cannot permanently degrade the bot to the fallback.
        """
        with self._cb_lock:
            if self._consecutive_failures < _MAX_CONSECUTIVE_FAILURES:
                return False
            if (time.monotonic() - self._opened_at) > _CIRCUIT_OPEN_COOLDOWN_SECONDS:
                logger.info(
                    "LLM circuit breaker cooldown elapsed for %s; probing primary.",
                    self._primary,
                )
                return False
            return True

    # Backward-compat private alias for callers that still use the old name.
    _is_circuit_open = is_circuit_open

    def _record_success(self) -> None:
        """Reset the breaker on a successful primary call."""
        with self._cb_lock:
            if self._consecutive_failures > 0:
                logger.info(
                    "LLM circuit breaker reset after success on %s.",
                    self._primary,
                )
            self._consecutive_failures = 0
            self._opened_at = 0.0

    def _record_failure(self) -> None:
        """Record a primary-model failure; may open the circuit breaker."""
        with self._cb_lock:
            self._consecutive_failures += 1
            if self._consecutive_failures == _MAX_CONSECUTIVE_FAILURES:
                self._opened_at = time.monotonic()
            if self._consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                logger.warning(
                    "LLM circuit breaker OPEN for %s: %d consecutive failures.",
                    self._primary,
                    self._consecutive_failures,
                )

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """Send a chat request. Returns the assistant's text.

        Routes through the primary model first (if circuit is closed) and
        falls back to ``fallback_model`` on any exception. Raises
        ``RuntimeError`` only if both paths fail.
        """
        assert system_prompt, "system_prompt must not be empty"  # noqa: S101
        assert user_prompt, "user_prompt must not be empty"  # noqa: S101

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        use_primary = not self.is_circuit_open()
        if use_primary:
            try:
                text = self._call(self._primary, messages)
                self._record_success()
                return text  # noqa: TRY300
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "LLM primary model %s failed: %s. Trying fallback.",
                    self._primary,
                    _sanitize(str(exc)),
                )
                self._record_failure()

        if self._fallback:
            try:
                return self._call(self._fallback, messages)
            except Exception as exc:
                logger.exception(
                    "LLM fallback model %s failed",
                    self._fallback,
                )
                safe = _sanitize(str(exc))
                msg = f"Both primary and fallback LLM calls failed: {safe}"
                raise RuntimeError(msg) from None

        msg = f"LLM primary model {self._primary} failed and no fallback configured."
        raise RuntimeError(msg)

    @retry(
        retry=retry_if_exception_type(_retryable_exceptions()),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(2),
        reraise=True,
    )
    def _call(self, model: str, messages: list[dict]) -> str:
        """Execute a single litellm completion; pull text out of the response."""
        import litellm

        # Quiet litellm's console chatter: per-call INFO lines and the
        # "Give Feedback / Get Help" banners drown the trading log.
        # Idempotent — cheap to set on every call.
        litellm.suppress_debug_info = True
        logging.getLogger("LiteLLM").setLevel(logging.WARNING)

        kwargs: dict = {
            "model": model,
            "messages": messages,
            "max_tokens": self._max_tokens,
            "timeout": self._timeout,
        }

        provider = model.split("/")[0] if "/" in model else None
        provider_key_map = {
            "xai": "xai_api_key",
            "anthropic": "anthropic_api_key",
            "gemini": "gemini_api_key",
            "openai": "openai_api_key",
        }
        if provider and provider in provider_key_map and provider in self._api_keys:
            kwargs[provider_key_map[provider]] = self._api_keys[provider]

        response = litellm.completion(**kwargs)

        choices = getattr(response, "choices", None) or response.get("choices", [])
        if not choices:
            msg = f"LLM response had no choices: {response!r}"
            raise RuntimeError(msg)
        first = choices[0]
        message = getattr(first, "message", None) or first.get("message", {})
        content = getattr(message, "content", None) or message.get("content", "")
        if not content or not isinstance(content, str):
            msg = f"LLM response content missing or non-string: {content!r}"
            raise RuntimeError(msg)
        return content.strip()
