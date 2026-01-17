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
