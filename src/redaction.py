"""Central logging redaction for credentials and secret-bearing URLs."""

from __future__ import annotations

import logging
import re
import traceback
from collections.abc import Iterable, Mapping
from typing import Any, Literal

REDACTED = "[REDACTED]"

_AUTHORIZATION_PATTERN = re.compile(
    r"(\b(?:proxy[-_ ]?)?authorization\b[\"']?\s*[:=]\s*[\"']?)"
    r"(?:(?:basic|bearer|token)\s+)?[^\"'\s,;}]+",
    re.IGNORECASE,
)
_SECRET_ASSIGNMENT_PATTERN = re.compile(
    r"(\b(?:api[-_ ]?key|x-api-key|access[-_ ]?token|auth[-_ ]?token|"
    r"refresh[-_ ]?token|client[-_ ]?secret|password|passwd|secret|signature)"
    r"\b[\"']?\s*[:=]\s*[\"']?)[^\"'\s,;}]+",
    re.IGNORECASE,
)
_URL_USERINFO_PATTERN = re.compile(
    r"([a-z][a-z0-9+.-]*://)[^/@\s:]+(?::[^/@\s]*)?@",
    re.IGNORECASE,
)
_URL_QUERY_SECRET_PATTERN = re.compile(
    r"([?&](?:api[-_]?key|access[-_]?token|auth[-_]?token|refresh[-_]?token|token|"
    r"client[-_]?secret|secret|password|passwd|signature|sig|key)=)[^&#\s\"']+",
    re.IGNORECASE,
)
_WEBHOOK_PATTERNS = (
    re.compile(
        r"(https?://[^/\s?#]+/(?:api/)?(?:webhooks?|hooks?)/)[^?\s#]+",
        re.IGNORECASE,
    ),
    re.compile(r"(https?://hooks\.slack\.com/services/)[^?\s#]+", re.IGNORECASE),
)
_PROVIDER_KEY_PATTERNS = (
    re.compile(r"\bsk-(?:proj-|ant-|live-|test-)?[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\bxai-[A-Za-z0-9_-]{12,}\b", re.IGNORECASE),
    re.compile(r"\bAIza[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b", re.IGNORECASE),
    re.compile(r"\b(?:rk_live|whsec)_[A-Za-z0-9_-]{12,}\b", re.IGNORECASE),
    re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
        re.DOTALL,
    ),
)


class Redactor:
    """Redact configured values and common credential representations."""

    def __init__(self, secret_values: Iterable[object] = ()) -> None:
        """Compile an immutable, longest-first set of configured secrets."""
        secrets = {_secret_text(value) for value in secret_values}
        secrets.discard("")
        self._secret_values = tuple(sorted(secrets, key=len, reverse=True))

    def redact(self, text: str) -> str:
        """Return text with credentials replaced by a stable marker."""
        redacted = text
        for secret in self._secret_values:
            redacted = redacted.replace(secret, REDACTED)
        for pattern in _WEBHOOK_PATTERNS:
            redacted = pattern.sub(rf"\1{REDACTED}", redacted)
        redacted = _URL_USERINFO_PATTERN.sub(rf"\1{REDACTED}@", redacted)
        redacted = _URL_QUERY_SECRET_PATTERN.sub(rf"\1{REDACTED}", redacted)
        redacted = _AUTHORIZATION_PATTERN.sub(rf"\1{REDACTED}", redacted)
        redacted = _SECRET_ASSIGNMENT_PATTERN.sub(rf"\1{REDACTED}", redacted)
        for pattern in _PROVIDER_KEY_PATTERNS:
            redacted = pattern.sub(REDACTED, redacted)
        return redacted

    def redact_value(self, value: object) -> object:  # noqa: PLR0911
        """Recursively redact strings in structured logging values."""
        if isinstance(value, str):
            return self.redact(value)
        if isinstance(value, bytes):
            return self.redact(value.decode("utf-8", errors="replace"))
        if isinstance(value, Mapping):
            return {self.redact(str(key)): self.redact_value(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(self.redact_value(item) for item in value)
        if isinstance(value, list):
            return [self.redact_value(item) for item in value]
        if isinstance(value, set):
            return {self.redact_value(item) for item in value}
        return value


class RedactingFilter(logging.Filter):
    """Sanitize a log record before any formatter or downstream handler sees it."""

    def __init__(
        self,
        secret_values: Iterable[object] = (),
        *,
        redactor: Redactor | None = None,
        name: str = "",
    ) -> None:
        """Create a filter from configured secrets or a shared redactor."""
        super().__init__(name)
        self.redactor = redactor or Redactor(secret_values)

    def filter(self, record: logging.LogRecord) -> bool:
        """Redact message arguments, structured extras, stack, and traceback text."""
        record.msg = self.redactor.redact(record.getMessage())
        record.args = ()

        for key, value in tuple(record.__dict__.items()):
            if key not in {"exc_info", "exc_text", "msg", "args", "stack_info"}:
                record.__dict__[key] = self.redactor.redact_value(value)

        if record.exc_info is not None:
            rendered = "".join(traceback.format_exception(*record.exc_info))
            record.exc_text = self.redactor.redact(rendered).rstrip()
            record.exc_info = None
        elif record.exc_text:
            record.exc_text = self.redactor.redact(record.exc_text)
        if record.stack_info:
            record.stack_info = self.redactor.redact(record.stack_info)
        return True


class RedactingFormatter(logging.Formatter):
    """Apply final-output redaction, including formatter-generated traceback text."""

    def __init__(  # noqa: PLR0913
        self,
        fmt: str | None = None,
        datefmt: str | None = None,
        style: Literal["%", "{", "$"] = "%",
        validate: bool = True,  # noqa: FBT001, FBT002
        *,
        defaults: Mapping[str, Any] | None = None,
        secret_values: Iterable[object] = (),
        redactor: Redactor | None = None,
    ) -> None:
        """Create a normal logging formatter with a final redaction pass."""
        super().__init__(fmt, datefmt, style, validate, defaults=defaults)
        self.redactor = redactor or Redactor(secret_values)

    def format(self, record: logging.LogRecord) -> str:
        """Format the record and redact every rendered field."""
        return self.redactor.redact(super().format(record))


def _secret_text(value: object) -> str:
    reveal = getattr(value, "get_secret_value", None)
    if callable(reveal):
        revealed = reveal()
        if isinstance(revealed, bytes):
            return revealed.decode("utf-8", errors="replace")
        return str(revealed)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)
