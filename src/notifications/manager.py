"""Notification manager powered by Apprise.

Reads one or more Apprise URLs from the ``NOTIFY_URLS`` environment variable
(one URL per line) and sends notifications through all configured services.

Supported services (via Apprise): Telegram, Discord, Slack, Feishu/Lark,
DingTalk, Email, WeChat Work, and 80+ more. See https://github.com/caronc/apprise
for the full list and URL formats.

Environment variable:
  NOTIFY_URLS — newline-separated Apprise URLs, e.g.:
    tgram://123456:ABC-DEF/,-12345
    discord://webhook_id/webhook_token
    slack://tokenA/tokenB/tokenC
    lark://app_id:app_secret/@group
    dingtalk://token/secret
    mailto://user:pass@smtp.example.com
"""

from __future__ import annotations

import logging
import os

from src.notifications.base import NotificationMessage

logger = logging.getLogger(__name__)


def _build_apprise():
    """Create an Apprise instance from NOTIFY_URLS env var."""
    try:
        import apprise
    except ImportError:
        logger.warning("apprise library not installed; notifications disabled. Run: pip install apprise")
        return None

    raw = os.getenv("NOTIFY_URLS", "").strip()
    if not raw:
        return None

    ap = apprise.Apprise()
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ap.add(line):
            logger.info("Notification channel added: %s", _mask_url(line))
        else:
            logger.warning("Invalid Apprise URL skipped: %s", _mask_url(line))
    return ap


def _mask_url(url: str) -> str:
    """Mask sensitive parts of an Apprise URL for logging."""
    # Show scheme and first ~10 chars, mask the rest
    if "://" in url:
        scheme, rest = url.split("://", 1)
        visible = min(10, len(rest))
        return f"{scheme}://{rest[:visible]}***"
    return url[:15] + "***"


class NotificationManager:
    """Sends notifications via Apprise."""

    def __init__(self):
        self._apprise = _build_apprise()

    @property
    def configured_channels(self) -> list[str]:
        """Return a list of configured plugin names (e.g. ['telegram', 'discord'])."""
        if not self._apprise:
            return []
        return [p.service_name for p in self._apprise]

    @property
    def url_count(self) -> int:
        """Number of configured Apprise URLs."""
        return len(self._apprise) if self._apprise else 0

    def send_all(self, message: NotificationMessage) -> dict[str, bool]:
        """Send to all configured channels. Returns {service_name: success}."""
        if not self._apprise or self.url_count == 0:
            logger.debug("No notification channels configured")
            return {}

        body = message.body
        if message.title:
            body = f"**{message.title}**\n\n{body}"

        results: dict[str, bool] = {}
        for plugin in self._apprise:
            name = plugin.service_name
            try:
                ok = plugin.notify(body=body, title=message.title)
                results[name] = bool(ok)
                if ok:
                    logger.info("Notification sent: %s", name)
                else:
                    logger.warning("Notification returned False: %s", name)
            except Exception as exc:
                logger.warning("Notification failed for %s: %s", name, exc)
                results[name] = False

        logger.info("Notification results: %s", results)
        return results

    def send(self, message: NotificationMessage) -> bool:
        """Send to all channels. Returns True if at least one succeeded."""
        results = self.send_all(message)
        if not results:
            return False
        return any(results.values())


_manager: NotificationManager | None = None


def get_notification_manager() -> NotificationManager:
    global _manager
    if _manager is None:
        _manager = NotificationManager()
    return _manager
