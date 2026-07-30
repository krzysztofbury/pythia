"""Webhook notifications (Discord / Slack compatible)."""

from __future__ import annotations

import logging
from typing import Any

import requests

from redaction import Redactor

logger = logging.getLogger(__name__)


class WebhookNotifier:
    """Send notifications via Discord/Slack-compatible webhooks."""

    def __init__(
        self,
        webhook_url: str,
        error_webhook_url: str | None = None,
        *,
        redactor: Redactor | None = None,
    ) -> None:
        """Initialize with webhook URLs."""
        assert webhook_url, "Webhook URL must not be empty"  # noqa: S101

        self._url = webhook_url
        self._error_url = error_webhook_url or webhook_url
        self._redactor = redactor or Redactor((webhook_url, error_webhook_url or ""))

    def _post(self, url: str, event: dict[str, Any]) -> None:
        payload = self._redactor.redact_value(event)
        assert isinstance(payload, dict)  # noqa: S101
        text = self._render(payload)
        try:
            response = requests.post(
                url,
                json={"content": text, "text": text, "event": payload},
                timeout=10,
            )
            response.raise_for_status()
        except requests.RequestException as error:
            logger.warning(
                "Webhook notification failed: %s",
                self._redactor.redact(str(error)),
            )

    @staticmethod
    def _render(event: dict[str, Any]) -> str:
        event_type = str(event.get("type", "event"))
        if event_type == "trade_fill":
            return (
                f"**{event['side']} FILL** {event['ticker']}: "
                f"{event['amount']:.8f} @ ${event['price']:.8f}\n"
                f"> {str(event.get('reason', ''))[:200]}"
            )
        if event_type == "error":
            return f"**ERROR**: {str(event.get('message', ''))[:500]}"
        return f"**Cycle Summary**\n{str(event.get('summary', ''))[:1000]}"

    def notify_event(self, event: dict[str, Any], *, error: bool = False) -> None:
        """Send one redacted structured event."""
        self._post(self._error_url if error else self._url, event)

    def notify_trade(
        self,
        ticker: str,
        side: str,
        price: float,
        amount: float,
        reasoning: str,
    ) -> None:
        """Notify about a trade execution."""
        self.notify_event(
            {
                "type": "trade_fill",
                "ticker": ticker,
                "side": side,
                "price": price,
                "amount": amount,
                "reason": reasoning,
            },
        )

    def notify_error(self, message: str) -> None:
        """Notify about an error."""
        self.notify_event({"type": "error", "message": message}, error=True)

    def notify_cycle_summary(self, summary: str) -> None:
        """Notify cycle summary."""
        self.notify_event({"type": "cycle_summary", "summary": summary})


class NullNotifier:
    """No-op notifier when webhook is not configured."""

    def notify_trade(self, *_args: object, **_kwargs: object) -> None:
        """No-op."""

    def notify_error(self, *_args: object, **_kwargs: object) -> None:
        """No-op."""

    def notify_cycle_summary(self, *_args: object, **_kwargs: object) -> None:
        """No-op."""

    def notify_event(self, *_args: object, **_kwargs: object) -> None:
        """No-op."""
