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
