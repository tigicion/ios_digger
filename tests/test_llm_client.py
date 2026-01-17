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
