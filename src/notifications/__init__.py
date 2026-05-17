"""Multi-channel notification system powered by Apprise."""

from src.notifications.manager import NotificationManager, get_notification_manager
from src.notifications.base import NotificationMessage

__all__ = ["NotificationManager", "NotificationMessage", "get_notification_manager"]
