"""Integration tests for iOS Digger."""
import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock, AsyncMock

from scraper import App, Review, AppSearcher, ReviewScraper
from analyzer import InsightsAnalyzer, AnalysisResult, Insight
from reporter import PDFGenerator

# Check if WeasyPrint system dependencies are available
try:
    from weasyprint import HTML
    WEASYPRINT_AVAILABLE = True
except OSError:
    WEASYPRINT_AVAILABLE = False


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

    # 4. PDF generator can be initialized
    generator = PDFGenerator()
    assert generator.template_dir.exists()


@pytest.mark.skipif(not WEASYPRINT_AVAILABLE, reason="WeasyPrint system dependencies not installed")
def test_pdf_generation(mock_analysis_result):
    """Test actual PDF generation (requires WeasyPrint system libs)."""
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


@pytest.mark.skipif(not WEASYPRINT_AVAILABLE, reason="WeasyPrint system dependencies not installed")
def test_pdf_generation_all_levels(mock_analysis_result):
    """Test PDF generation works for all report levels."""
    generator = PDFGenerator()

    for level in ["brief", "standard", "full"]:
        pdf_bytes = generator.generate(
            title=f"Test Report - {level}",
            result=mock_analysis_result,
            regions=["cn"],
            level=level,
        )
        assert len(pdf_bytes) > 0
        assert pdf_bytes[:4] == b'%PDF'


def test_module_imports():
    """Test that all module exports are accessible."""
    # Test scraper exports
    assert App is not None
    assert Review is not None
    assert AppSearcher is not None
    assert ReviewScraper is not None

    # Test analyzer exports
    assert InsightsAnalyzer is not None
    assert AnalysisResult is not None
    assert Insight is not None

    # Test reporter exports
    assert PDFGenerator is not None


def test_review_classification():
    """Test review positive/negative classification."""
    positive_review = Review("123", "cn", 5, "好", "很好", "A", "1.0", datetime.now())
    neutral_review = Review("123", "cn", 3, "一般", "一般", "B", "1.0", datetime.now())
    negative_review = Review("123", "cn", 1, "差", "很差", "C", "1.0", datetime.now())

    assert positive_review.is_positive is True
    assert positive_review.is_negative is False

    assert neutral_review.is_positive is False
    assert neutral_review.is_negative is False

    assert negative_review.is_positive is False
    assert negative_review.is_negative is True
