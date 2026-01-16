"""Tests for App search functionality."""
import pytest
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
