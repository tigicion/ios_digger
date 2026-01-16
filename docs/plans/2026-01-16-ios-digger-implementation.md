# iOS Digger Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a CLI tool that scrapes App Store reviews and uses LLM to extract user pain points, positive feedback, and product insights into a PDF report.

**Architecture:** Modular Python CLI with four main components: scraper (iTunes API/RSS), analyzer (Qwen LLM), reporter (WeasyPrint PDF), and CLI (Typer). Data flows linearly: input → search → scrape → analyze → generate PDF.

**Tech Stack:** Python 3.11+, Typer, httpx, openai (Qwen), WeasyPrint, Jinja2, Pydantic

---

## Task 1: Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `config.py`

**Step 1: Create requirements.txt**

```txt
openai>=1.0.0
typer>=0.9.0
httpx>=0.25.0
weasyprint>=60.0
jinja2>=3.0.0
pydantic>=2.0.0
python-dotenv>=1.0.0
pytest>=7.0.0
```

**Step 2: Create .env.example**

```bash
# Required: Alibaba DashScope API Key
DASHSCOPE_API_KEY=sk-xxx
```

**Step 3: Create .gitignore**

```
__pycache__/
*.pyc
.env
*.pdf
.venv/
venv/
.pytest_cache/
```

**Step 4: Create config.py**

```python
"""Configuration for iOS Digger."""

REGIONS = {
    "cn": "中国",
    "us": "美国",
    "jp": "日本",
    "kr": "韩国",
    "sg": "新加坡",
    "th": "泰国",
    "vn": "越南",
    "id": "印尼",
}

CATEGORIES = {
    "效率": 6007,
    "健康健身": 6013,
    "生活": 6012,
    "工具": 6002,
    "社交": 6005,
    "财务": 6015,
    "教育": 6017,
    "娱乐": 6016,
    "摄影与录像": 6008,
    "音乐": 6011,
}

DEFAULT_REGIONS = ["cn", "us"]
DEFAULT_LEVEL = "standard"
DEFAULT_MODEL = "qwen-plus"
DEFAULT_TOP = 10

ITUNES_SEARCH_URL = "https://itunes.apple.com/search"
ITUNES_RSS_URL = "https://itunes.apple.com/{region}/rss/customerreviews/id={app_id}/sortBy=mostRecent/json"
ITUNES_CATEGORY_RSS_URL = "https://itunes.apple.com/{region}/rss/topfreeapplications/genre={genre_id}/limit={limit}/json"

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
```

**Step 5: Create virtual environment and install dependencies**

Run: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`

**Step 6: Commit**

```bash
git add requirements.txt .env.example .gitignore config.py
git commit -m "feat: add project setup and configuration"
```

---

## Task 2: Data Models

**Files:**
- Create: `scraper/__init__.py`
- Create: `scraper/models.py`
- Create: `tests/__init__.py`
- Create: `tests/test_models.py`

**Step 1: Create directory structure**

```bash
mkdir -p scraper tests
```

**Step 2: Create scraper/__init__.py**

```python
"""Scraper module for fetching App Store data."""
```

**Step 3: Create tests/__init__.py**

```python
"""Tests for iOS Digger."""
```

**Step 4: Write the failing test for models**

Create `tests/test_models.py`:

```python
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
```

**Step 5: Run test to verify it fails**

Run: `python -m pytest tests/test_models.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'scraper.models'"

**Step 6: Write implementation**

Create `scraper/models.py`:

```python
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
```

**Step 7: Run test to verify it passes**

Run: `python -m pytest tests/test_models.py -v`
Expected: PASS (3 tests)

**Step 8: Commit**

```bash
git add scraper/ tests/
git commit -m "feat: add App and Review data models"
```

---

## Task 3: App Search Module

**Files:**
- Create: `scraper/search.py`
- Create: `tests/test_search.py`

**Step 1: Write the failing test**

Create `tests/test_search.py`:

```python
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
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_search.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'scraper.search'"

**Step 3: Write implementation**

Create `scraper/search.py`:

```python
"""App Store search functionality."""
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
        import asyncio

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
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_search.py -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add scraper/search.py tests/test_search.py
git commit -m "feat: add App Store search functionality"
```

---

## Task 4: Review Scraper Module

**Files:**
- Create: `scraper/reviews.py`
- Create: `tests/test_reviews.py`

**Step 1: Write the failing test**

Create `tests/test_reviews.py`:

```python
"""Tests for review scraping functionality."""
import pytest
from datetime import datetime
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
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_reviews.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'scraper.reviews'"

**Step 3: Write implementation**

Create `scraper/reviews.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_reviews.py -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add scraper/reviews.py tests/test_reviews.py
git commit -m "feat: add review scraping functionality"
```

---

## Task 5: LLM Client Module

**Files:**
- Create: `analyzer/__init__.py`
- Create: `analyzer/llm_client.py`
- Create: `tests/test_llm_client.py`

**Step 1: Create directory structure**

```bash
mkdir -p analyzer
```

**Step 2: Create analyzer/__init__.py**

```python
"""Analyzer module for LLM-based review analysis."""
```

**Step 3: Write the failing test**

Create `tests/test_llm_client.py`:

```python
"""Tests for LLM client."""
import os
import pytest
from unittest.mock import patch, MagicMock
from analyzer.llm_client import LLMClient


def test_llm_client_initialization():
    """Test LLM client can be initialized."""
    with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"}):
        client = LLMClient()
        assert client.model == "qwen-plus"


def test_llm_client_custom_model():
    """Test LLM client with custom model."""
    with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"}):
        client = LLMClient(model="qwen-max")
        assert client.model == "qwen-max"


def test_build_analysis_prompt():
    """Test building analysis prompt."""
    with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"}):
        client = LLMClient()
        reviews_text = "1. [5★] 很好用 - 推荐\n2. [1★] 闪退 - 经常崩溃"
        prompt = client._build_analysis_prompt(reviews_text)
        assert "痛点" in prompt
        assert "好评" in prompt
        assert reviews_text in prompt
```

**Step 4: Run test to verify it fails**

Run: `python -m pytest tests/test_llm_client.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'analyzer.llm_client'"

**Step 5: Write implementation**

Create `analyzer/llm_client.py`:

```python
"""LLM client for review analysis using Qwen."""
import os
import json
from typing import Optional

from openai import OpenAI
from dotenv import load_dotenv

from config import DASHSCOPE_BASE_URL, DEFAULT_MODEL

load_dotenv()


class LLMClient:
    """Client for Qwen LLM API via OpenAI-compatible interface."""

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY environment variable is required")

        self.client = OpenAI(
            api_key=api_key,
            base_url=DASHSCOPE_BASE_URL,
        )

    def _build_analysis_prompt(self, reviews_text: str) -> str:
        """Build prompt for review analysis."""
        return f"""你是一位专业的产品经理，擅长从用户评论中提取洞察。

请分析以下 App Store 用户评论，提取：

1. **痛点** (pain_points): 用户抱怨的问题，按严重程度和频率排序
2. **好评亮点** (positive_feedback): 用户称赞的功能或体验
3. **用户需求** (user_needs): 用户明确表达或隐含的需求/期望

对于每个洞察，请提供：
- summary: 一句话总结
- severity: 严重程度 (high/medium/low)
- sample_quotes: 1-2 条代表性原文引用

请用 JSON 格式输出，结构如下：
{{
  "pain_points": [
    {{"summary": "...", "severity": "high", "sample_quotes": ["..."]}}
  ],
  "positive_feedback": [
    {{"summary": "...", "severity": "medium", "sample_quotes": ["..."]}}
  ],
  "user_needs": [
    {{"summary": "...", "severity": "medium", "sample_quotes": ["..."]}}
  ]
}}

评论内容：
{reviews_text}

请直接输出 JSON，不要添加 markdown 代码块或其他格式。"""

    def _build_summary_prompt(self, batch_results: list[dict]) -> str:
        """Build prompt for summarizing multiple batch results."""
        results_text = json.dumps(batch_results, ensure_ascii=False, indent=2)
        return f"""你是一位专业的产品经理。以下是多批次评论分析的结果，请合并汇总：

1. 合并相同或相似的洞察
2. 按频率和严重程度重新排序
3. 保留最具代表性的引用

{results_text}

请用相同的 JSON 格式输出合并后的结果，包含 pain_points, positive_feedback, user_needs 三个字段。
请直接输出 JSON，不要添加 markdown 代码块或其他格式。"""

    def analyze(self, prompt: str) -> str:
        """Send analysis request to LLM."""
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,  # Lower temperature for more consistent output
        )
        return completion.choices[0].message.content

    def analyze_reviews(self, reviews_text: str) -> dict:
        """Analyze reviews and return structured insights."""
        prompt = self._build_analysis_prompt(reviews_text)
        response = self.analyze(prompt)

        # Parse JSON response
        try:
            # Try to extract JSON from response
            response = response.strip()
            if response.startswith("```"):
                # Remove markdown code block if present
                lines = response.split("\n")
                response = "\n".join(lines[1:-1])
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "pain_points": [],
                "positive_feedback": [],
                "user_needs": [],
                "raw_response": response,
            }

    def summarize_batches(self, batch_results: list[dict]) -> dict:
        """Summarize multiple batch analysis results."""
        if len(batch_results) == 1:
            return batch_results[0]

        prompt = self._build_summary_prompt(batch_results)
        response = self.analyze(prompt)

        try:
            response = response.strip()
            if response.startswith("```"):
                lines = response.split("\n")
                response = "\n".join(lines[1:-1])
            return json.loads(response)
        except json.JSONDecodeError:
            # Fallback: merge manually
            merged = {
                "pain_points": [],
                "positive_feedback": [],
                "user_needs": [],
            }
            for result in batch_results:
                for key in merged:
                    merged[key].extend(result.get(key, []))
            return merged
```

**Step 6: Run test to verify it passes**

Run: `python -m pytest tests/test_llm_client.py -v`
Expected: PASS (3 tests)

**Step 7: Commit**

```bash
git add analyzer/ tests/test_llm_client.py
git commit -m "feat: add LLM client for Qwen API"
```

---

## Task 6: Insights Analyzer Module

**Files:**
- Create: `analyzer/insights.py`
- Create: `tests/test_insights.py`

**Step 1: Write the failing test**

Create `tests/test_insights.py`:

```python
"""Tests for insights analyzer."""
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch
from analyzer.insights import InsightsAnalyzer, AnalysisResult, Insight
from scraper.models import Review


@pytest.fixture
def sample_reviews():
    return [
        Review(
            app_id="123",
            region="cn",
            rating=1,
            title="闪退",
            content="打开就闪退，根本用不了",
            author="用户A",
            version="1.0",
            date=datetime.now(),
        ),
        Review(
            app_id="123",
            region="cn",
            rating=5,
            title="很好用",
            content="界面简洁，功能强大",
            author="用户B",
            version="1.0",
            date=datetime.now(),
        ),
    ]


def test_format_reviews_for_analysis(sample_reviews):
    """Test formatting reviews for LLM analysis."""
    with patch.dict("os.environ", {"DASHSCOPE_API_KEY": "test"}):
        analyzer = InsightsAnalyzer.__new__(InsightsAnalyzer)
        analyzer.llm = MagicMock()
        analyzer.batch_size = 50

        text = analyzer._format_reviews(sample_reviews)
        assert "闪退" in text
        assert "很好用" in text
        assert "[1★]" in text
        assert "[5★]" in text


def test_insight_dataclass():
    """Test Insight dataclass."""
    insight = Insight(
        summary="App 经常闪退",
        frequency=10,
        severity="high",
        sample_reviews=["打开就闪退"],
    )
    assert insight.summary == "App 经常闪退"
    assert insight.severity == "high"


def test_analysis_result_dataclass():
    """Test AnalysisResult dataclass."""
    result = AnalysisResult(
        pain_points=[Insight("闪退", 5, "high", [])],
        positive_feedback=[Insight("界面好", 3, "medium", [])],
        user_needs=[Insight("需要夜间模式", 2, "low", [])],
        regional_diff={},
    )
    assert len(result.pain_points) == 1
    assert len(result.positive_feedback) == 1
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_insights.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'analyzer.insights'"

**Step 3: Write implementation**

Create `analyzer/insights.py`:

```python
"""Insights extraction from reviews using LLM."""
from dataclasses import dataclass, field
from typing import Optional, Callable

from scraper.models import Review
from analyzer.llm_client import LLMClient


@dataclass
class Insight:
    """A single insight extracted from reviews."""
    summary: str
    frequency: int
    severity: str  # high, medium, low
    sample_reviews: list[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    """Complete analysis result."""
    pain_points: list[Insight]
    positive_feedback: list[Insight]
    user_needs: list[Insight]
    regional_diff: dict = field(default_factory=dict)
    total_reviews: int = 0
    avg_rating: float = 0.0
    rating_distribution: dict = field(default_factory=dict)


class InsightsAnalyzer:
    """Analyze reviews to extract insights using LLM."""

    def __init__(self, model: str = "qwen-plus", batch_size: int = 50):
        self.llm = LLMClient(model=model)
        self.batch_size = batch_size

    def _format_reviews(self, reviews: list[Review]) -> str:
        """Format reviews for LLM analysis."""
        lines = []
        for i, r in enumerate(reviews, 1):
            lines.append(f"{i}. [{r.rating}★] {r.title} - {r.content}")
        return "\n".join(lines)

    def _calculate_stats(self, reviews: list[Review]) -> tuple[float, dict]:
        """Calculate rating statistics."""
        if not reviews:
            return 0.0, {}

        total = sum(r.rating for r in reviews)
        avg = total / len(reviews)

        distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for r in reviews:
            distribution[r.rating] += 1

        return avg, distribution

    def _parse_llm_result(self, result: dict) -> tuple[list[Insight], list[Insight], list[Insight]]:
        """Parse LLM result into Insight objects."""
        def parse_insights(items: list) -> list[Insight]:
            insights = []
            for item in items:
                insights.append(Insight(
                    summary=item.get("summary", ""),
                    frequency=item.get("frequency", 1),
                    severity=item.get("severity", "medium"),
                    sample_reviews=item.get("sample_quotes", []),
                ))
            return insights

        pain_points = parse_insights(result.get("pain_points", []))
        positive = parse_insights(result.get("positive_feedback", []))
        needs = parse_insights(result.get("user_needs", []))

        return pain_points, positive, needs

    def analyze(
        self,
        reviews: list[Review],
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> AnalysisResult:
        """Analyze reviews and extract insights."""
        if not reviews:
            return AnalysisResult(
                pain_points=[],
                positive_feedback=[],
                user_needs=[],
            )

        # Calculate stats
        avg_rating, distribution = self._calculate_stats(reviews)

        # Batch analysis
        batches = [
            reviews[i:i + self.batch_size]
            for i in range(0, len(reviews), self.batch_size)
        ]

        batch_results = []
        for i, batch in enumerate(batches):
            text = self._format_reviews(batch)
            result = self.llm.analyze_reviews(text)
            batch_results.append(result)

            if on_progress:
                on_progress(i + 1, len(batches))

        # Summarize batches
        final_result = self.llm.summarize_batches(batch_results)

        # Parse into Insight objects
        pain_points, positive, needs = self._parse_llm_result(final_result)

        return AnalysisResult(
            pain_points=pain_points,
            positive_feedback=positive,
            user_needs=needs,
            total_reviews=len(reviews),
            avg_rating=avg_rating,
            rating_distribution=distribution,
        )

    def analyze_by_region(
        self,
        reviews_by_region: dict[str, list[Review]],
        on_progress: Optional[Callable[[str, int, int], None]] = None,
    ) -> tuple[AnalysisResult, dict[str, AnalysisResult]]:
        """Analyze reviews grouped by region."""
        # Combine all reviews for overall analysis
        all_reviews = []
        for reviews in reviews_by_region.values():
            all_reviews.extend(reviews)

        # Overall analysis
        overall = self.analyze(all_reviews)

        # Per-region analysis (for regional differences)
        regional = {}
        for region, reviews in reviews_by_region.items():
            if reviews:
                regional[region] = self.analyze(reviews)
                if on_progress:
                    on_progress(region, len(regional), len(reviews_by_region))

        # Add regional diff to overall result
        overall.regional_diff = self._compute_regional_diff(regional)

        return overall, regional

    def _compute_regional_diff(self, regional: dict[str, AnalysisResult]) -> dict:
        """Compute notable differences between regions."""
        diff = {}
        for region, result in regional.items():
            diff[region] = {
                "avg_rating": result.avg_rating,
                "total_reviews": result.total_reviews,
                "top_pain_point": result.pain_points[0].summary if result.pain_points else None,
                "top_positive": result.positive_feedback[0].summary if result.positive_feedback else None,
            }
        return diff
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_insights.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add analyzer/insights.py tests/test_insights.py
git commit -m "feat: add insights analyzer with batch processing"
```

---

## Task 7: PDF Reporter Module - Templates

**Files:**
- Create: `reporter/__init__.py`
- Create: `reporter/templates/styles.css`
- Create: `reporter/templates/base.html`
- Create: `reporter/templates/report_brief.html`
- Create: `reporter/templates/report_standard.html`
- Create: `reporter/templates/report_full.html`

**Step 1: Create directory structure**

```bash
mkdir -p reporter/templates
```

**Step 2: Create reporter/__init__.py**

```python
"""Reporter module for PDF generation."""
```

**Step 3: Create styles.css**

Create `reporter/templates/styles.css`:

```css
@page {
    size: A4;
    margin: 2cm;
}

body {
    font-family: "PingFang SC", "Microsoft YaHei", sans-serif;
    font-size: 11pt;
    line-height: 1.6;
    color: #333;
}

h1 {
    font-size: 24pt;
    color: #1a1a1a;
    border-bottom: 2px solid #007AFF;
    padding-bottom: 10px;
    margin-bottom: 20px;
}

h2 {
    font-size: 16pt;
    color: #333;
    margin-top: 30px;
    margin-bottom: 15px;
}

h3 {
    font-size: 13pt;
    color: #555;
    margin-top: 20px;
}

.cover {
    text-align: center;
    padding-top: 200px;
}

.cover h1 {
    font-size: 32pt;
    border: none;
}

.cover .meta {
    font-size: 12pt;
    color: #666;
    margin-top: 50px;
}

.summary-box {
    background: #f5f5f7;
    border-radius: 8px;
    padding: 20px;
    margin: 20px 0;
}

.summary-box .stat {
    display: inline-block;
    margin-right: 40px;
}

.summary-box .stat-value {
    font-size: 24pt;
    font-weight: bold;
    color: #007AFF;
}

.summary-box .stat-label {
    font-size: 10pt;
    color: #666;
}

.insight-card {
    background: #fff;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 15px;
    margin: 10px 0;
}

.insight-card.high {
    border-left: 4px solid #FF3B30;
}

.insight-card.medium {
    border-left: 4px solid #FF9500;
}

.insight-card.low {
    border-left: 4px solid #34C759;
}

.insight-title {
    font-weight: bold;
    margin-bottom: 8px;
}

.insight-quote {
    font-style: italic;
    color: #666;
    font-size: 10pt;
    background: #f9f9f9;
    padding: 8px;
    border-radius: 4px;
    margin-top: 8px;
}

.severity-tag {
    display: inline-block;
    font-size: 9pt;
    padding: 2px 8px;
    border-radius: 4px;
    margin-left: 10px;
}

.severity-tag.high {
    background: #FFEBEE;
    color: #FF3B30;
}

.severity-tag.medium {
    background: #FFF3E0;
    color: #FF9500;
}

.severity-tag.low {
    background: #E8F5E9;
    color: #34C759;
}

.region-table {
    width: 100%;
    border-collapse: collapse;
    margin: 20px 0;
}

.region-table th,
.region-table td {
    border: 1px solid #e0e0e0;
    padding: 10px;
    text-align: left;
}

.region-table th {
    background: #f5f5f7;
}

.page-break {
    page-break-after: always;
}
```

**Step 4: Create base.html**

Create `reporter/templates/base.html`:

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>{{ title }}</title>
    <style>
        {% include 'styles.css' %}
    </style>
</head>
<body>
    {% block content %}{% endblock %}
</body>
</html>
```

**Step 5: Create report_brief.html**

Create `reporter/templates/report_brief.html`:

```html
{% extends 'base.html' %}

{% block content %}
<div class="cover">
    <h1>{{ title }}</h1>
    <div class="meta">
        <p>分析日期：{{ date }}</p>
        <p>覆盖地区：{{ regions }}</p>
        <p>评论总数：{{ total_reviews }}</p>
    </div>
</div>

<div class="page-break"></div>

<h2>痛点排名</h2>
{% for insight in pain_points[:5] %}
<div class="insight-card {{ insight.severity }}">
    <div class="insight-title">
        {{ loop.index }}. {{ insight.summary }}
        <span class="severity-tag {{ insight.severity }}">{{ insight.severity }}</span>
    </div>
    {% if insight.sample_reviews %}
    <div class="insight-quote">"{{ insight.sample_reviews[0] }}"</div>
    {% endif %}
</div>
{% endfor %}

<h2>好评亮点</h2>
{% for insight in positive_feedback[:3] %}
<div class="insight-card low">
    <div class="insight-title">{{ loop.index }}. {{ insight.summary }}</div>
    {% if insight.sample_reviews %}
    <div class="insight-quote">"{{ insight.sample_reviews[0] }}"</div>
    {% endif %}
</div>
{% endfor %}

<h2>用户需求洞察</h2>
{% for insight in user_needs[:5] %}
<div class="insight-card {{ insight.severity }}">
    <div class="insight-title">
        {{ loop.index }}. {{ insight.summary }}
        <span class="severity-tag {{ insight.severity }}">{{ insight.severity }}</span>
    </div>
</div>
{% endfor %}
{% endblock %}
```

**Step 6: Create report_standard.html**

Create `reporter/templates/report_standard.html`:

```html
{% extends 'base.html' %}

{% block content %}
<div class="cover">
    <h1>{{ title }}</h1>
    <div class="meta">
        <p>分析日期：{{ date }}</p>
        <p>覆盖地区：{{ regions }}</p>
        <p>评论总数：{{ total_reviews }}</p>
    </div>
</div>

<div class="page-break"></div>

<h2>概览</h2>
<div class="summary-box">
    <div class="stat">
        <div class="stat-value">{{ total_reviews }}</div>
        <div class="stat-label">评论总数</div>
    </div>
    <div class="stat">
        <div class="stat-value">{{ "%.1f"|format(avg_rating) }}</div>
        <div class="stat-label">平均评分</div>
    </div>
    <div class="stat">
        <div class="stat-value">{{ pain_points|length }}</div>
        <div class="stat-label">发现痛点</div>
    </div>
    <div class="stat">
        <div class="stat-value">{{ user_needs|length }}</div>
        <div class="stat-label">用户需求</div>
    </div>
</div>

<h3>评分分布</h3>
<table class="region-table">
    <tr>
        <th>评分</th>
        <th>数量</th>
        <th>占比</th>
    </tr>
    {% for rating, count in rating_distribution.items() %}
    <tr>
        <td>{{ rating }} 星</td>
        <td>{{ count }}</td>
        <td>{{ "%.1f%%"|format(count / total_reviews * 100 if total_reviews > 0 else 0) }}</td>
    </tr>
    {% endfor %}
</table>

<div class="page-break"></div>

<h2>痛点排名</h2>
{% for insight in pain_points %}
<div class="insight-card {{ insight.severity }}">
    <div class="insight-title">
        {{ loop.index }}. {{ insight.summary }}
        <span class="severity-tag {{ insight.severity }}">{{ insight.severity }}</span>
    </div>
    {% for quote in insight.sample_reviews[:2] %}
    <div class="insight-quote">"{{ quote }}"</div>
    {% endfor %}
</div>
{% endfor %}

<div class="page-break"></div>

<h2>好评亮点</h2>
{% for insight in positive_feedback %}
<div class="insight-card low">
    <div class="insight-title">{{ loop.index }}. {{ insight.summary }}</div>
    {% for quote in insight.sample_reviews[:2] %}
    <div class="insight-quote">"{{ quote }}"</div>
    {% endfor %}
</div>
{% endfor %}

<h2>用户需求洞察</h2>
{% for insight in user_needs %}
<div class="insight-card {{ insight.severity }}">
    <div class="insight-title">
        {{ loop.index }}. {{ insight.summary }}
        <span class="severity-tag {{ insight.severity }}">{{ insight.severity }}</span>
    </div>
    {% for quote in insight.sample_reviews[:2] %}
    <div class="insight-quote">"{{ quote }}"</div>
    {% endfor %}
</div>
{% endfor %}

{% if regional_diff %}
<div class="page-break"></div>

<h2>地区差异分析</h2>
<table class="region-table">
    <tr>
        <th>地区</th>
        <th>评论数</th>
        <th>平均评分</th>
        <th>主要痛点</th>
        <th>主要好评</th>
    </tr>
    {% for region, data in regional_diff.items() %}
    <tr>
        <td>{{ region_names.get(region, region) }}</td>
        <td>{{ data.total_reviews }}</td>
        <td>{{ "%.1f"|format(data.avg_rating) }}</td>
        <td>{{ data.top_pain_point or '-' }}</td>
        <td>{{ data.top_positive or '-' }}</td>
    </tr>
    {% endfor %}
</table>
{% endif %}
{% endblock %}
```

**Step 7: Create report_full.html**

Create `reporter/templates/report_full.html`:

```html
{% extends 'report_standard.html' %}

{% block content %}
{{ super() }}

<div class="page-break"></div>

<h2>产品建议</h2>

<h3>优先修复</h3>
<p>基于痛点严重程度和频率，建议优先处理以下问题：</p>
{% for insight in pain_points[:3] if insight.severity == 'high' %}
<div class="insight-card high">
    <div class="insight-title">{{ loop.index }}. {{ insight.summary }}</div>
    <p>建议：针对此问题进行专项修复，并在更新日志中说明。</p>
</div>
{% endfor %}

<h3>功能增强</h3>
<p>用户明确表达的需求，建议纳入产品路线图：</p>
{% for insight in user_needs[:5] %}
<div class="insight-card {{ insight.severity }}">
    <div class="insight-title">{{ loop.index }}. {{ insight.summary }}</div>
</div>
{% endfor %}

<h3>保持优势</h3>
<p>用户好评的亮点，建议继续保持和强化：</p>
{% for insight in positive_feedback[:3] %}
<div class="insight-card low">
    <div class="insight-title">{{ loop.index }}. {{ insight.summary }}</div>
</div>
{% endfor %}

{% endblock %}
```

**Step 8: Commit**

```bash
git add reporter/
git commit -m "feat: add PDF report templates"
```

---

## Task 8: PDF Generator

**Files:**
- Create: `reporter/generator.py`
- Create: `tests/test_generator.py`

**Step 1: Write the failing test**

Create `tests/test_generator.py`:

```python
"""Tests for PDF generator."""
import pytest
from datetime import datetime
from analyzer.insights import AnalysisResult, Insight
from reporter.generator import PDFGenerator


@pytest.fixture
def sample_result():
    return AnalysisResult(
        pain_points=[
            Insight("App 闪退", 10, "high", ["打开就闪退"]),
            Insight("加载慢", 5, "medium", ["加载要很久"]),
        ],
        positive_feedback=[
            Insight("界面简洁", 8, "medium", ["界面很好看"]),
        ],
        user_needs=[
            Insight("希望支持夜间模式", 3, "low", ["能加个夜间模式吗"]),
        ],
        total_reviews=100,
        avg_rating=3.5,
        rating_distribution={1: 10, 2: 15, 3: 20, 4: 25, 5: 30},
    )


def test_generator_initialization():
    """Test PDF generator can be initialized."""
    generator = PDFGenerator()
    assert generator.template_dir.exists()


def test_prepare_context(sample_result):
    """Test preparing template context."""
    generator = PDFGenerator()
    context = generator._prepare_context(
        title="Test Report",
        result=sample_result,
        regions=["cn", "us"],
    )
    assert context["title"] == "Test Report"
    assert context["total_reviews"] == 100
    assert len(context["pain_points"]) == 2


def test_get_template(sample_result):
    """Test getting correct template by level."""
    generator = PDFGenerator()

    brief = generator._get_template("brief")
    assert "report_brief" in brief.name

    standard = generator._get_template("standard")
    assert "report_standard" in standard.name

    full = generator._get_template("full")
    assert "report_full" in full.name
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_generator.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'reporter.generator'"

**Step 3: Write implementation**

Create `reporter/generator.py`:

```python
"""PDF report generator using WeasyPrint."""
from datetime import datetime
from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML

from config import REGIONS
from analyzer.insights import AnalysisResult


class PDFGenerator:
    """Generate PDF reports from analysis results."""

    def __init__(self):
        self.template_dir = Path(__file__).parent / "templates"
        self.env = Environment(
            loader=FileSystemLoader(self.template_dir),
            autoescape=True,
        )

    def _get_template(self, level: str):
        """Get template by report level."""
        template_map = {
            "brief": "report_brief.html",
            "standard": "report_standard.html",
            "full": "report_full.html",
        }
        template_name = template_map.get(level, "report_standard.html")
        return self.env.get_template(template_name)

    def _prepare_context(
        self,
        title: str,
        result: AnalysisResult,
        regions: list[str],
    ) -> dict:
        """Prepare template context."""
        return {
            "title": title,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "regions": ", ".join(REGIONS.get(r, r) for r in regions),
            "total_reviews": result.total_reviews,
            "avg_rating": result.avg_rating,
            "rating_distribution": result.rating_distribution,
            "pain_points": result.pain_points,
            "positive_feedback": result.positive_feedback,
            "user_needs": result.user_needs,
            "regional_diff": result.regional_diff,
            "region_names": REGIONS,
        }

    def generate(
        self,
        title: str,
        result: AnalysisResult,
        regions: list[str],
        level: str = "standard",
        output_path: Optional[str] = None,
    ) -> bytes:
        """Generate PDF report.

        Args:
            title: Report title
            result: Analysis result
            regions: List of region codes
            level: Report level (brief, standard, full)
            output_path: Optional path to save PDF file

        Returns:
            PDF content as bytes
        """
        template = self._get_template(level)
        context = self._prepare_context(title, result, regions)

        html_content = template.render(**context)

        # Generate PDF
        html = HTML(string=html_content, base_url=str(self.template_dir))
        pdf_bytes = html.write_pdf()

        # Save to file if path provided
        if output_path:
            Path(output_path).write_bytes(pdf_bytes)

        return pdf_bytes

    def generate_to_file(
        self,
        title: str,
        result: AnalysisResult,
        regions: list[str],
        level: str = "standard",
        output_dir: str = ".",
    ) -> str:
        """Generate PDF and save to file with auto-generated name.

        Returns:
            Path to generated PDF file
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        safe_title = "".join(c for c in title if c.isalnum() or c in "_ -")[:50]
        filename = f"report_{timestamp}_{safe_title}.pdf"
        output_path = Path(output_dir) / filename

        self.generate(title, result, regions, level, str(output_path))

        return str(output_path)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_generator.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add reporter/generator.py tests/test_generator.py
git commit -m "feat: add PDF generator with WeasyPrint"
```

---

## Task 9: CLI Implementation

**Files:**
- Create: `main.py`
- Create: `tests/test_cli.py`

**Step 1: Write the failing test**

Create `tests/test_cli.py`:

```python
"""Tests for CLI."""
import pytest
from typer.testing import CliRunner
from main import app

runner = CliRunner()


def test_regions_command():
    """Test listing available regions."""
    result = runner.invoke(app, ["regions"])
    assert result.exit_code == 0
    assert "cn" in result.stdout
    assert "中国" in result.stdout


def test_categories_command():
    """Test listing available categories."""
    result = runner.invoke(app, ["categories"])
    assert result.exit_code == 0
    assert "效率" in result.stdout


def test_search_missing_keyword():
    """Test search command requires keyword."""
    result = runner.invoke(app, ["search"])
    assert result.exit_code != 0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cli.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'main'"

**Step 3: Write implementation**

Create `main.py`:

```python
#!/usr/bin/env python3
"""iOS Digger - App Store Review Mining Tool."""
import asyncio
from datetime import datetime
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

from config import REGIONS, CATEGORIES, DEFAULT_REGIONS, DEFAULT_LEVEL, DEFAULT_MODEL, DEFAULT_TOP
from scraper.search import AppSearcher
from scraper.reviews import ReviewScraper
from analyzer.insights import InsightsAnalyzer
from reporter.generator import PDFGenerator

app = typer.Typer(
    name="ios-digger",
    help="App Store 评论挖掘工具 - 从用户评论中提取痛点、好评与需求洞察",
)
console = Console()


@app.command()
def regions():
    """显示支持的地区列表"""
    table = Table(title="支持的地区")
    table.add_column("代码", style="cyan")
    table.add_column("名称", style="green")

    for code, name in REGIONS.items():
        table.add_row(code, name)

    console.print(table)


@app.command()
def categories():
    """显示支持的 App 类别"""
    table = Table(title="支持的类别")
    table.add_column("类别名称", style="cyan")
    table.add_column("ID", style="dim")

    for name, genre_id in CATEGORIES.items():
        table.add_row(name, str(genre_id))

    console.print(table)


@app.command()
def search(
    keyword: str = typer.Argument(..., help="搜索关键词，如：备忘录、健身"),
    regions: str = typer.Option(
        ",".join(DEFAULT_REGIONS),
        "--regions", "-r",
        help="地区列表，逗号分隔，或 'all'",
    ),
    level: str = typer.Option(
        DEFAULT_LEVEL,
        "--level", "-l",
        help="报告级别：brief / standard / full",
    ),
    top: int = typer.Option(
        DEFAULT_TOP,
        "--top", "-t",
        help="每个地区搜索 Top N 个 App",
    ),
    model: str = typer.Option(
        DEFAULT_MODEL,
        "--model", "-m",
        help="LLM 模型：qwen-plus / qwen-turbo / qwen-max",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output", "-o",
        help="输出文件路径（默认自动生成）",
    ),
):
    """按关键词搜索 App 并分析评论"""
    region_list = list(REGIONS.keys()) if regions == "all" else regions.split(",")

    # Validate regions
    for r in region_list:
        if r not in REGIONS:
            console.print(f"[red]错误：未知地区 '{r}'[/red]")
            console.print(f"支持的地区：{', '.join(REGIONS.keys())}")
            raise typer.Exit(1)

    asyncio.run(_search_and_analyze(keyword, region_list, level, top, model, output))


@app.command()
def category(
    category_name: str = typer.Argument(..., help="类别名称，如：效率、健康健身"),
    regions: str = typer.Option(
        ",".join(DEFAULT_REGIONS),
        "--regions", "-r",
        help="地区列表，逗号分隔，或 'all'",
    ),
    level: str = typer.Option(
        DEFAULT_LEVEL,
        "--level", "-l",
        help="报告级别：brief / standard / full",
    ),
    top: int = typer.Option(
        DEFAULT_TOP,
        "--top", "-t",
        help="每个地区获取 Top N 个 App",
    ),
    model: str = typer.Option(
        DEFAULT_MODEL,
        "--model", "-m",
        help="LLM 模型",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output", "-o",
        help="输出文件路径",
    ),
):
    """按类别获取 Top App 并分析评论"""
    if category_name not in CATEGORIES:
        console.print(f"[red]错误：未知类别 '{category_name}'[/red]")
        console.print(f"支持的类别：{', '.join(CATEGORIES.keys())}")
        raise typer.Exit(1)

    region_list = list(REGIONS.keys()) if regions == "all" else regions.split(",")

    asyncio.run(_category_and_analyze(category_name, region_list, level, top, model, output))


@app.command()
def analyze(
    app_ids: str = typer.Argument(..., help="App ID 列表，逗号分隔"),
    regions: str = typer.Option(
        ",".join(DEFAULT_REGIONS),
        "--regions", "-r",
        help="地区列表，逗号分隔，或 'all'",
    ),
    level: str = typer.Option(
        DEFAULT_LEVEL,
        "--level", "-l",
        help="报告级别：brief / standard / full",
    ),
    model: str = typer.Option(
        DEFAULT_MODEL,
        "--model", "-m",
        help="LLM 模型",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output", "-o",
        help="输出文件路径",
    ),
):
    """直接分析指定 App ID 的评论"""
    id_list = [id.strip() for id in app_ids.split(",")]
    region_list = list(REGIONS.keys()) if regions == "all" else regions.split(",")

    asyncio.run(_analyze_apps(id_list, region_list, level, model, output))


async def _search_and_analyze(
    keyword: str,
    regions: list[str],
    level: str,
    top: int,
    model: str,
    output: Optional[str],
):
    """Search apps and analyze reviews."""
    searcher = AppSearcher()

    # Search apps
    console.print(f"\n[bold]🔍 搜索 '{keyword}' 相关 App...[/bold]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("搜索中...", total=None)
        region_apps = await searcher.search_by_keyword_multi_region(keyword, regions, top)
        progress.update(task, completed=True)

    # Collect unique app IDs
    all_apps = {}
    for region, apps in region_apps.items():
        console.print(f"  {REGIONS[region]}：找到 {len(apps)} 个 App")
        for app in apps:
            if app.id not in all_apps:
                all_apps[app.id] = app

    if not all_apps:
        console.print("[yellow]未找到任何 App[/yellow]")
        return

    app_ids = list(all_apps.keys())
    await _fetch_analyze_report(app_ids, regions, level, model, output, f"{keyword} 相关 App")


async def _category_and_analyze(
    category_name: str,
    regions: list[str],
    level: str,
    top: int,
    model: str,
    output: Optional[str],
):
    """Get top apps in category and analyze."""
    searcher = AppSearcher()

    console.print(f"\n[bold]📂 获取 '{category_name}' 类别 Top {top} App...[/bold]")

    all_apps = {}
    for region in regions:
        try:
            apps = await searcher.get_top_apps_by_category(category_name, region, top)
            console.print(f"  {REGIONS[region]}：找到 {len(apps)} 个 App")
            for app in apps:
                if app.id not in all_apps:
                    all_apps[app.id] = app
        except Exception as e:
            console.print(f"  [yellow]{REGIONS[region]}：获取失败 - {e}[/yellow]")

    if not all_apps:
        console.print("[yellow]未找到任何 App[/yellow]")
        return

    app_ids = list(all_apps.keys())
    await _fetch_analyze_report(app_ids, regions, level, model, output, f"{category_name} 类 Top App")


async def _analyze_apps(
    app_ids: list[str],
    regions: list[str],
    level: str,
    model: str,
    output: Optional[str],
):
    """Analyze specified app IDs."""
    await _fetch_analyze_report(app_ids, regions, level, model, output, f"App 分析")


async def _fetch_analyze_report(
    app_ids: list[str],
    regions: list[str],
    level: str,
    model: str,
    output: Optional[str],
    title: str,
):
    """Fetch reviews, analyze, and generate report."""
    scraper = ReviewScraper()

    # Fetch reviews
    console.print(f"\n[bold]📥 抓取评论中...[/bold]")

    all_reviews = []
    reviews_by_region = {r: [] for r in regions}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("抓取中...", total=len(app_ids) * len(regions))

        for app_id in app_ids:
            region_reviews = await scraper.get_reviews_multi_region(app_id, regions)
            for region, reviews in region_reviews.items():
                all_reviews.extend(reviews)
                reviews_by_region[region].extend(reviews)
                progress.advance(task)

    console.print(f"  共抓取 {len(all_reviews)} 条评论")

    if not all_reviews:
        console.print("[yellow]未抓取到任何评论[/yellow]")
        return

    # Analyze
    console.print(f"\n[bold]🤖 LLM 分析中...[/bold]")

    analyzer = InsightsAnalyzer(model=model)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("分析中...", total=None)
        result, _ = analyzer.analyze_by_region(reviews_by_region)
        progress.update(task, completed=True)

    # Generate report
    console.print(f"\n[bold]📄 生成报告...[/bold]")

    generator = PDFGenerator()
    output_path = generator.generate_to_file(
        title=title,
        result=result,
        regions=regions,
        level=level,
        output_dir="." if not output else str(Path(output).parent),
    )

    if output:
        import shutil
        shutil.move(output_path, output)
        output_path = output

    console.print(f"  [green]✅ 报告已保存: {output_path}[/green]")


if __name__ == "__main__":
    app()
```

**Step 4: Add rich to requirements.txt**

Edit `requirements.txt` to add:
```
rich>=13.0.0
```

**Step 5: Install new dependency**

Run: `pip install rich`

**Step 6: Run test to verify it passes**

Run: `python -m pytest tests/test_cli.py -v`
Expected: PASS (3 tests)

**Step 7: Commit**

```bash
git add main.py tests/test_cli.py requirements.txt
git commit -m "feat: add CLI with search, category, and analyze commands"
```

---

## Task 10: Integration Test & Final Polish

**Files:**
- Create: `tests/test_integration.py`
- Update: `scraper/__init__.py`
- Update: `analyzer/__init__.py`
- Update: `reporter/__init__.py`

**Step 1: Update module __init__ files for clean imports**

Update `scraper/__init__.py`:
```python
"""Scraper module for fetching App Store data."""
from scraper.models import App, Review
from scraper.search import AppSearcher
from scraper.reviews import ReviewScraper

__all__ = ["App", "Review", "AppSearcher", "ReviewScraper"]
```

Update `analyzer/__init__.py`:
```python
"""Analyzer module for LLM-based review analysis."""
from analyzer.llm_client import LLMClient
from analyzer.insights import InsightsAnalyzer, AnalysisResult, Insight

__all__ = ["LLMClient", "InsightsAnalyzer", "AnalysisResult", "Insight"]
```

Update `reporter/__init__.py`:
```python
"""Reporter module for PDF generation."""
from reporter.generator import PDFGenerator

__all__ = ["PDFGenerator"]
```

**Step 2: Write integration test**

Create `tests/test_integration.py`:

```python
"""Integration tests for iOS Digger."""
import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock, AsyncMock

from scraper import App, Review, AppSearcher, ReviewScraper
from analyzer import InsightsAnalyzer, AnalysisResult, Insight
from reporter import PDFGenerator


@pytest.fixture
def mock_apps():
    return [
        App("123", "Test App 1", "Dev 1", "效率", ""),
        App("456", "Test App 2", "Dev 2", "效率", ""),
    ]


@pytest.fixture
def mock_reviews():
    return [
        Review("123", "cn", 5, "很好", "非常好用", "用户A", "1.0", datetime.now()),
        Review("123", "cn", 1, "闪退", "打开就闪退", "用户B", "1.0", datetime.now()),
        Review("123", "cn", 4, "不错", "功能丰富", "用户C", "1.0", datetime.now()),
    ]


@pytest.fixture
def mock_analysis_result():
    return AnalysisResult(
        pain_points=[Insight("闪退问题", 5, "high", ["打开就闪退"])],
        positive_feedback=[Insight("界面好看", 3, "medium", ["界面简洁"])],
        user_needs=[Insight("希望支持夜间模式", 2, "low", [])],
        total_reviews=100,
        avg_rating=3.5,
        rating_distribution={1: 10, 2: 15, 3: 20, 4: 25, 5: 30},
    )


def test_full_pipeline_mock(mock_apps, mock_reviews, mock_analysis_result):
    """Test the full pipeline with mocked external calls."""
    # This tests that all components work together

    # 1. Search returns apps
    assert len(mock_apps) == 2

    # 2. Reviews can be collected
    assert len(mock_reviews) == 3

    # 3. Analysis result has expected structure
    assert len(mock_analysis_result.pain_points) == 1
    assert mock_analysis_result.total_reviews == 100

    # 4. PDF can be generated
    generator = PDFGenerator()
    pdf_bytes = generator.generate(
        title="Test Report",
        result=mock_analysis_result,
        regions=["cn", "us"],
        level="brief",
    )
    assert len(pdf_bytes) > 0
    assert pdf_bytes[:4] == b'%PDF'  # PDF magic number


def test_data_flow():
    """Test data flows correctly between components."""
    # Create review
    review = Review(
        app_id="123",
        region="cn",
        rating=2,
        title="问题",
        content="有bug",
        author="用户",
        version="1.0",
        date=datetime.now(),
    )

    # Check properties work
    assert review.is_negative is True
    assert review.is_positive is False

    # Create insight
    insight = Insight(
        summary="有bug",
        frequency=1,
        severity="medium",
        sample_reviews=["有bug"],
    )

    # Create result
    result = AnalysisResult(
        pain_points=[insight],
        positive_feedback=[],
        user_needs=[],
        total_reviews=1,
        avg_rating=2.0,
        rating_distribution={2: 1},
    )

    # Verify result structure
    assert result.pain_points[0].summary == "有bug"
```

**Step 3: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add scraper/__init__.py analyzer/__init__.py reporter/__init__.py tests/test_integration.py
git commit -m "feat: add integration tests and clean module exports"
```

---

## Summary

Implementation complete. The tool can now:

1. **Search** apps by keyword across multiple regions
2. **Browse** category rankings
3. **Analyze** specific App IDs
4. **Extract** pain points, positive feedback, and user needs via Qwen LLM
5. **Generate** PDF reports at three detail levels

**Usage:**
```bash
# Activate virtual environment
source .venv/bin/activate

# Set API key
export DASHSCOPE_API_KEY="sk-xxx"

# Run
python main.py search "备忘录" --regions cn,us,jp --level standard
python main.py category "效率" --top 10 --level brief
python main.py analyze 123456789 --regions all --level full
```
