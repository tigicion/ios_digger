"""Tests for App search functionality."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from scraper.search import AppSearcher
from scraper.models import App


@pytest.fixture
def searcher():
    return AppSearcher()


def test_parse_search_result(searcher):
    """Test parsing iTunes search API response."""
    raw_result = {
        "trackId": 123456789,
        "trackName": "Test App",
        "artistName": "Test Developer",
        "primaryGenreName": "Productivity",
        "artworkUrl100": "https://example.com/icon.png",
    }
    app = searcher._parse_app(raw_result)
    assert app.id == "123456789"
    assert app.name == "Test App"
    assert app.developer == "Test Developer"


def test_build_search_url(searcher):
    """Test building search URL."""
    url = searcher._build_search_url("备忘录", "cn", 10)
    assert "itunes.apple.com/search" in url
    assert "term=" in url
    assert "country=cn" in url
    assert "limit=10" in url


@pytest.mark.asyncio
async def test_search_by_keyword(searcher):
    """Test searching apps by keyword with mocked response."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "results": [
            {
                "trackId": 123,
                "trackName": "Test App",
                "artistName": "Developer",
                "primaryGenreName": "Utilities",
                "artworkUrl100": "https://example.com/icon.png",
            }
        ]
    }
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
        apps = await searcher.search_by_keyword("test", "us", 10)

    assert len(apps) == 1
    assert apps[0].name == "Test App"
    assert apps[0].id == "123"


@pytest.mark.asyncio
async def test_get_top_apps_by_category_invalid(searcher):
    """Test that invalid category raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        await searcher.get_top_apps_by_category("InvalidCategory", "cn", 10)
    assert "Unknown category" in str(exc_info.value)
