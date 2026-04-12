"""Webhook notifications (Discord / Slack compatible)."""
from __future__ import annotations

import logging

import requests

logger = logging.getLogger(__name__)


class WebhookNotifier:
    """Send notifications via Discord/Slack-compatible webhooks."""

    def __init__(self, webhook_url: str, error_webhook_url: str | None = None) -> None:
        """Initialize with webhook URLs."""
        assert webhook_url, "Webhook URL must not be empty"  # noqa: S101

        self._url = webhook_url
        self._error_url = error_webhook_url or webhook_url

    def _post(self, url: str, text: str) -> None:
        try:
            # Discord uses "content", Slack uses "text" — send both
            requests.post(url, json={"content": text, "text": text}, timeout=10)
        except Exception:  # noqa: BLE001
            logger.warning("Webhook notification failed.", exc_info=True)

    def notify_trade(
        self,
        ticker: str,
        side: str,
        price: float,
        amount: float,
        reasoning: str,
    ) -> None:
        """Notify about a trade execution."""
        self._post(
            self._url,
            f"**{side}** {ticker}: {amount:.8f} @ ${price:.8f}\n> {reasoning[:200]}",
        )

    def notify_error(self, message: str) -> None:
        """Notify about an error."""
        self._post(self._error_url, f"**ERROR**: {message[:500]}")

    def notify_cycle_summary(self, summary: str) -> None:
        """Notify cycle summary."""
        self._post(self._url, f"**Cycle Summary**\n{summary[:1000]}")


class NullNotifier:
    """No-op notifier when webhook is not configured."""

    def notify_trade(self, *_args: object, **_kwargs: object) -> None:
        """No-op."""

    def notify_error(self, *_args: object, **_kwargs: object) -> None:
        """No-op."""

    def notify_cycle_summary(self, *_args: object, **_kwargs: object) -> None:
        """No-op."""
