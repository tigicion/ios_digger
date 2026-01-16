"""Tests for data models."""
from datetime import datetime
from scraper.models import App, Review


def test_app_creation():
    app = App(
        id="123456789",
        name="Test App",
        developer="Test Developer",
        category="效率",
        icon_url="https://example.com/icon.png",
    )
    assert app.id == "123456789"
    assert app.name == "Test App"
    assert app.developer == "Test Developer"
    assert app.category == "效率"


def test_review_creation():
    review = Review(
        app_id="123456789",
        region="cn",
        rating=5,
        title="很好用",
        content="这个应用非常好用，推荐！",
        author="用户A",
        version="1.0.0",
        date=datetime(2026, 1, 15),
    )
    assert review.app_id == "123456789"
    assert review.region == "cn"
    assert review.rating == 5
    assert review.is_positive is True


def test_review_is_negative():
    review = Review(
        app_id="123456789",
        region="cn",
        rating=2,
        title="不好",
        content="经常闪退",
        author="用户B",
        version="1.0.0",
        date=datetime(2026, 1, 15),
    )
    assert review.is_positive is False
    assert review.is_negative is True
