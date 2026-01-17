"""PDF report generator using WeasyPrint."""
from datetime import datetime
from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader

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

        # Import WeasyPrint at runtime to allow testing without system dependencies
        from weasyprint import HTML

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
