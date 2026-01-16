"""App Store review scraping functionality."""
import asyncio
from datetime import datetime
from typing import Optional

import httpx

from config import ITUNES_RSS_URL
from scraper.models import Review


class ReviewScraper:
    """Scrape reviews from App Store RSS feed."""

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout

    def _build_url(self, app_id: str, region: str) -> str:
        """Build RSS feed URL for app reviews."""
        return ITUNES_RSS_URL.format(region=region, app_id=app_id)

    def _parse_review(self, entry: dict, app_id: str, region: str) -> Review:
        """Parse a single review entry from RSS feed."""
        # Parse date
        date_str = entry.get("updated", {}).get("label", "")
        try:
            # Handle ISO format with timezone
            date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            date = datetime.now()

        return Review(
            app_id=app_id,
            region=region,
            rating=int(entry.get("im:rating", {}).get("label", "3")),
            title=entry.get("title", {}).get("label", ""),
            content=entry.get("content", {}).get("label", ""),
            author=entry.get("author", {}).get("name", {}).get("label", ""),
            version=entry.get("im:version", {}).get("label", ""),
            date=date,
        )

    async def get_reviews(self, app_id: str, region: str) -> list[Review]:
        """Get all reviews for an app in a specific region (up to 500)."""
        url = self._build_url(app_id, region)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()

                entries = data.get("feed", {}).get("entry", [])
                # First entry is often app info, skip it if no rating
                reviews = []
                for entry in entries:
                    if "im:rating" in entry:
                        reviews.append(self._parse_review(entry, app_id, region))
                return reviews
            except httpx.HTTPStatusError:
                # No reviews or app not found
                return []

    async def get_reviews_multi_region(
        self, app_id: str, regions: list[str]
    ) -> dict[str, list[Review]]:
        """Get reviews for an app across multiple regions."""
        async def fetch_region(region: str) -> tuple[str, list[Review]]:
            reviews = await self.get_reviews(app_id, region)
            return region, reviews

        tasks = [fetch_region(r) for r in regions]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        region_reviews = {}
        for result in results:
            if isinstance(result, tuple):
                region, reviews = result
                region_reviews[region] = reviews
        return region_reviews

    async def get_reviews_for_apps(
        self, app_ids: list[str], regions: list[str], on_progress: Optional[callable] = None
    ) -> dict[str, dict[str, list[Review]]]:
        """Get reviews for multiple apps across multiple regions."""
        all_reviews = {}  # app_id -> region -> reviews

        total = len(app_ids) * len(regions)
        completed = 0

        for app_id in app_ids:
            all_reviews[app_id] = {}
            region_reviews = await self.get_reviews_multi_region(app_id, regions)
            all_reviews[app_id] = region_reviews
            completed += len(regions)
            if on_progress:
                on_progress(completed, total)

        return all_reviews
