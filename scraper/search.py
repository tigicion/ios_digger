"""App Store search functionality."""
import asyncio

import httpx
from urllib.parse import urlencode

from config import ITUNES_SEARCH_URL, ITUNES_CATEGORY_RSS_URL, CATEGORIES
from scraper.models import App


class AppSearcher:
    """Search for apps on App Store."""

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout

    def _build_search_url(self, term: str, region: str, limit: int) -> str:
        """Build iTunes search API URL."""
        params = {
            "term": term,
            "country": region,
            "entity": "software",
            "limit": limit,
        }
        return f"{ITUNES_SEARCH_URL}?{urlencode(params)}"

    def _parse_app(self, raw: dict) -> App:
        """Parse raw API response into App object."""
        return App(
            id=str(raw.get("trackId", "")),
            name=raw.get("trackName", ""),
            developer=raw.get("artistName", ""),
            category=raw.get("primaryGenreName", ""),
            icon_url=raw.get("artworkUrl100", ""),
        )

    async def search_by_keyword(self, keyword: str, region: str, limit: int = 10) -> list[App]:
        """Search apps by keyword in a specific region."""
        url = self._build_search_url(keyword, region, limit)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
            return [self._parse_app(item) for item in data.get("results", [])]

    async def search_by_keyword_multi_region(
        self, keyword: str, regions: list[str], limit: int = 10
    ) -> dict[str, list[App]]:
        """Search apps by keyword across multiple regions."""
        async def search_region(region: str) -> tuple[str, list[App]]:
            apps = await self.search_by_keyword(keyword, region, limit)
            return region, apps

        tasks = [search_region(r) for r in regions]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        region_apps = {}
        for result in results:
            if isinstance(result, tuple):
                region, apps = result
                region_apps[region] = apps
        return region_apps

    async def get_top_apps_by_category(
        self, category: str, region: str, limit: int = 10
    ) -> list[App]:
        """Get top free apps in a category."""
        genre_id = CATEGORIES.get(category)
        if not genre_id:
            raise ValueError(f"Unknown category: {category}. Available: {list(CATEGORIES.keys())}")

        url = ITUNES_CATEGORY_RSS_URL.format(region=region, genre_id=genre_id, limit=limit)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()

            apps = []
            entries = data.get("feed", {}).get("entry", [])
            for entry in entries:
                app = App(
                    id=entry.get("id", {}).get("attributes", {}).get("im:id", ""),
                    name=entry.get("im:name", {}).get("label", ""),
                    developer=entry.get("im:artist", {}).get("label", ""),
                    category=category,
                    icon_url=entry.get("im:image", [{}])[-1].get("label", "") if entry.get("im:image") else "",
                )
                apps.append(app)
            return apps
