"""Structured, redacted webhook behavior."""

from unittest.mock import MagicMock, patch

import requests

from notifier import WebhookNotifier
from redaction import Redactor


def test_notifier_sends_redacted_structured_event_and_checks_http_status() -> None:
    """Send a structured payload only after recursively redacting it."""
    response = MagicMock()
    notifier = WebhookNotifier(
        "https://hooks.example.test/services/private/path",
        redactor=Redactor(("private-token-value",)),
    )

    with patch("notifier.requests.post", return_value=response) as post:
        notifier.notify_trade(
            "BTC/USDC",
            "BUY",
            100.0,
            0.5,
            "provider token=private-token-value",
        )

    response.raise_for_status.assert_called_once_with()
    payload = post.call_args.kwargs["json"]
    assert payload["event"]["type"] == "trade_fill"
    assert payload["event"]["reason"] == "provider token=[REDACTED]"
    assert "private-token-value" not in payload["content"]


def test_notifier_handles_non_success_http_status() -> None:
    """Consume HTTP failures without leaking or breaking the trading loop."""
    response = MagicMock()
    response.raise_for_status.side_effect = requests.HTTPError("503")
    notifier = WebhookNotifier("https://hooks.example.test/services/private/path")

    with patch("notifier.requests.post", return_value=response):
        notifier.notify_error("upstream unavailable")

    response.raise_for_status.assert_called_once_with()
