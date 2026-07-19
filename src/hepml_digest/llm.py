from __future__ import annotations

import os
from typing import Protocol, TypeVar

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .models import Paper, Review, Screening


T = TypeVar("T", bound=BaseModel)


class Analyzer(Protocol):
    def screen(self, paper: Paper) -> Screening: ...

    def review(self, paper: Paper, screening: Screening) -> Review: ...


class DeepSeekAnalyzer:
    def __init__(
        self,
        screening_model: str,
        review_model: str,
        screen_prompt: str,
        review_prompt: str,
        timeout: float = 90.0,
    ) -> None:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is missing. Use --demo for an offline test."
            )
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
            timeout=timeout,
        )
        self.screening_model = screening_model
        self.review_model = review_model
        self.screen_prompt = screen_prompt
        self.review_prompt = review_prompt

    @retry(
        retry=retry_if_exception_type(
            (
                ValueError,
                TypeError,
                APIConnectionError,
                APITimeoutError,
                InternalServerError,
                RateLimitError,
            )
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=20),
        reraise=True,
    )
    def _request_json(
        self,
        model: str,
        system: str,
        user: str,
        response_model: type[T],
        thinking: bool,
        max_tokens: int,
    ) -> T:
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            max_tokens=max_tokens,
            stream=False,
            extra_body={
                "thinking": {"type": "enabled" if thinking else "disabled"}
            },
        )
        content = response.choices[0].message.content
        if not content or not content.strip():
            raise ValueError("Model returned empty content")
        return response_model.model_validate_json(content)

    def screen(self, paper: Paper) -> Screening:
        user = (
            f"标题：{paper.title}\n"
            f"分类：{', '.join(paper.categories)}\n"
            f"摘要：{paper.abstract}"
        )
        return self._request_json(
            self.screening_model,
            self.screen_prompt,
            user,
            Screening,
            thinking=False,
            max_tokens=900,
        )

    def review(self, paper: Paper, screening: Screening) -> Review:
        user = (
            f"标题：{paper.title}\n"
            f"分类：{', '.join(paper.categories)}\n"
            f"初筛方法：{screening.method}\n"
            f"初筛理由：{screening.reason}\n"
            f"摘要：{paper.abstract}"
        )
        return self._request_json(
            self.review_model,
            self.review_prompt,
            user,
            Review,
            thinking=True,
            max_tokens=2200,
        )


class DemoAnalyzer:
    """Deterministic analyzer for installation checks; never calls an API."""

    def screen(self, paper: Paper) -> Screening:
        direct = "collider" in paper.abstract.lower()
        return Screening(
            relevance=0.91 if direct else 0.78,
            method="条件密度估计与不确定度校准",
            hep_tasks=["simulation_based_inference", "calibration"],
            reason="方法可用于不可解析似然下的参数推断与覆盖率检查",
            needs_deep_review=True,
            evidence_level="direct" if direct else "transferable",
        )

    def review(self, paper: Paper, screening: Screening) -> Review:
        return Review(
            summary_cn=(
                "论文研究条件密度估计及其校准。该方法可能用于基于模拟的"
                "参数推断，但迁移到真实实验前需要检验覆盖率和模拟偏差。"
            ),
            paper_claims=["演示数据上改善了条件密度估计的校准"],
            hep_opportunities=["近似不可解析似然", "构造校准的置信区间"],
            transfer_risks=["simulation-to-data shift", "nuisance 参数覆盖不足"],
            validation_plan="在公开 HEP toy model 上与直方图似然和 flow 基线比较。",
            evidence_level=screening.evidence_level
            if screening.evidence_level != "irrelevant"
            else "speculative",
            confidence=0.82,
        )
