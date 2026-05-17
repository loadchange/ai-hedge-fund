"""Notification message model."""

from __future__ import annotations

from pydantic import BaseModel


class NotificationMessage(BaseModel):
    """A notification to send."""

    title: str
    body: str
    format: str = "markdown"
    metadata: dict = {}
