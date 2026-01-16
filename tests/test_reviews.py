"""Tests for review scraping functionality."""
import pytest
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch
from scraper.reviews import ReviewScraper
from scraper.models import Review


@pytest.fixture
def scraper():
    return ReviewScraper()


def test_build_reviews_url(scraper):
    """Test building reviews RSS URL."""
    url = scraper._build_url("123456789", "cn")
    assert "itunes.apple.com/cn/rss/customerreviews" in url
    assert "id=123456789" in url


def test_parse_review(scraper):
    """Test parsing a single review from RSS feed."""
    raw_entry = {
        "author": {"name": {"label": "用户A"}},
        "im:rating": {"label": "5"},
        "title": {"label": "很好用"},
        "content": {"label": "这个应用非常好用！"},
        "im:version": {"label": "1.0.0"},
        "updated": {"label": "2026-01-15T10:00:00-07:00"},
    }
    review = scraper._parse_review(raw_entry, "123456789", "cn")
    assert review.app_id == "123456789"
    assert review.region == "cn"
    assert review.rating == 5
    assert review.title == "很好用"
    assert review.author == "用户A"


@pytest.mark.asyncio
async def test_get_reviews(scraper):
    """Test getting reviews with mocked response."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "feed": {
            "entry": [
                {
                    "author": {"name": {"label": "User1"}},
                    "im:rating": {"label": "4"},
                    "title": {"label": "Good"},
                    "content": {"label": "Works well"},
                    "im:version": {"label": "2.0"},
                    "updated": {"label": "2026-01-10T12:00:00Z"},
                }
            ]
        }
    }
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
        reviews = await scraper.get_reviews("123", "us")

    assert len(reviews) == 1
    assert reviews[0].rating == 4
    assert reviews[0].title == "Good"
