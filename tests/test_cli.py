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
