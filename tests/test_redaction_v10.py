"""Tests for centralized v10 logging redaction."""

from __future__ import annotations

import io
import logging

import pytest

from redaction import REDACTED, RedactingFilter, RedactingFormatter, Redactor


@pytest.mark.parametrize(
    "sensitive",
    [
        "Authorization: Bearer header-secret-value",
        "https://alice:password@example.com/private",
        "https://example.com/path?access_token=query-secret&safe=yes",
        "https://discord.com/api/webhooks/123456/webhook-secret-value",
        "sk-proj-abcdefghijklmnop123456",
        "xai-abcdefghijklmnop123456",
        "AIzaabcdefghijklmnopqrstuvwx",
    ],
)
def test_common_secret_shapes_are_redacted(sensitive: str) -> None:
    """Known credential representations never survive redaction."""
    output = Redactor().redact(f"request failed: {sensitive}")

    assert REDACTED in output
    assert "header-secret-value" not in output
    assert "alice:password" not in output
    assert "query-secret" not in output
    assert "webhook-secret-value" not in output
    assert "abcdefghijklmnop123456" not in output


def test_filter_redacts_configured_values_and_structured_arguments() -> None:
    """A handler filter sanitizes lazy and nested structured values."""
    configured_value = "configured-private-value"
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    redactor = Redactor([configured_value])
    handler.addFilter(RedactingFilter(redactor=redactor))
    handler.setFormatter(RedactingFormatter("%(message)s", redactor=redactor))
    logger = logging.getLogger("redaction-test")
    logger.handlers.clear()
    logger.propagate = False
    logger.addHandler(handler)

    logger.error("payload=%s", {"outer": {"token": configured_value}})

    rendered = stream.getvalue()
    assert configured_value not in rendered
    assert REDACTED in rendered


def test_nested_traceback_text_is_redacted() -> None:
    """Exception chains are sanitized even when secrets occur below the top error."""
    configured_value = "traceback-private-value"
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(RedactingFilter([configured_value]))
    logger = logging.getLogger("traceback-redaction-test")
    logger.handlers.clear()
    logger.propagate = False
    logger.addHandler(handler)

    try:
        _raise_nested_provider_error(configured_value)
    except RuntimeError:
        logger.exception("request failed")

    rendered = stream.getvalue()
    assert configured_value not in rendered
    assert "ValueError" in rendered
    assert "RuntimeError" in rendered
    assert REDACTED in rendered


def _raise_nested_provider_error(configured_value: str) -> None:
    try:
        _raise_provider_error(configured_value)
    except ValueError as error:
        message = "outer provider failure"
        raise RuntimeError(message) from error


def _raise_provider_error(configured_value: str) -> None:
    message = f"provider returned api_key={configured_value}"
    raise ValueError(message)
