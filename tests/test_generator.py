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
