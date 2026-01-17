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
