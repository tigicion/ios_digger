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


def test_llm_client_missing_api_key():
    """Test that missing API key raises ValueError."""
    with patch.dict(os.environ, {}, clear=True):
        # Need to also clear any loaded dotenv values
        with patch('analyzer.llm_client.os.getenv', return_value=None):
            with pytest.raises(ValueError) as exc_info:
                LLMClient()
            assert "DASHSCOPE_API_KEY" in str(exc_info.value)


def test_analyze_reviews():
    """Test analyzing reviews with mocked API response."""
    with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"}):
        with patch('analyzer.llm_client.OpenAI') as mock_openai:
            # Setup mock response
            mock_completion = MagicMock()
            mock_completion.choices = [MagicMock()]
            mock_completion.choices[0].message.content = '''{"pain_points": [{"summary": "闪退问题", "severity": "high", "sample_quotes": ["经常闪退"]}], "positive_feedback": [], "user_needs": []}'''
            mock_openai.return_value.chat.completions.create.return_value = mock_completion

            client = LLMClient()
            result = client.analyze_reviews("1. [1★] 闪退 - 经常闪退")

            assert "pain_points" in result
            assert len(result["pain_points"]) == 1
            assert result["pain_points"][0]["summary"] == "闪退问题"
