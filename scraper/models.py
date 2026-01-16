"""Data models for iOS Digger."""
from dataclasses import dataclass
from datetime import datetime


@dataclass
class App:
    """Represents an App Store application."""
    id: str
    name: str
    developer: str
    category: str
    icon_url: str = ""

    def __str__(self) -> str:
        return f"{self.name} ({self.id})"


@dataclass
class Review:
    """Represents an App Store review."""
    app_id: str
    region: str
    rating: int  # 1-5
    title: str
    content: str
    author: str
    version: str
    date: datetime

    @property
    def is_positive(self) -> bool:
        """Rating >= 4 is considered positive."""
        return self.rating >= 4

    @property
    def is_negative(self) -> bool:
        """Rating <= 2 is considered negative."""
        return self.rating <= 2

    def __str__(self) -> str:
        return f"[{self.rating}★] {self.title}"
