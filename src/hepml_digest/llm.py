from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Callable, Protocol, TypeVar
from urllib.parse import urlsplit

from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import (
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_REVIEW_MAX_TOKENS,
    DEFAULT_REVIEW_REASONING_EFFORT,
    Settings,
)
from .models import Paper, Review, Screening


T = TypeVar("T", bound=BaseModel)
SCREEN_TOKENS = 900
TERMINAL_API_FAILURES = frozenset({"authentication", "configuration", "output_limit"})


@dataclass(slots=True)
class StageUsage:
    calls: int = 0
    prompt_tokens: int = 0
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    last_finish_reason: str = ""

    def add(self, usage) -> None:
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
        hit = int(getattr(usage, "prompt_cache_hit_tokens", 0) or 0)
        miss = int(getattr(usage, "prompt_cache_miss_tokens", 0) or 0)
        if hit == 0 and miss == 0:
            details = getattr(usage, "prompt_tokens_details", None)
            hit = int(getattr(details, "cached_tokens", 0) or 0)
        self.prompt_tokens += prompt
        self.cache_hit_tokens += hit
        self.cache_miss_tokens += miss + max(0, prompt - hit - miss)
        self.completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
        details = getattr(usage, "completion_tokens_details", None)
        self.reasoning_tokens += int(getattr(details, "reasoning_tokens", 0) or 0)

    def cost(self, cached: float, uncached: float, output: float) -> float:
        return (
            self.cache_hit_tokens * cached + self.cache_miss_tokens * uncached
            + self.completion_tokens * output
        ) / 1_000_000


@dataclass(slots=True)
class AnalysisUsage:
    screening: StageUsage = field(default_factory=StageUsage)
    review: StageUsage = field(default_factory=StageUsage)
    failed_calls: int = 0
    last_failure_kind: str = ""
    unmetered_calls: int = 0

    @property
    def prompt_tokens(self) -> int:
        return self.screening.prompt_tokens + self.review.prompt_tokens

    @property
    def completion_tokens(self) -> int:
        return self.screening.completion_tokens + self.review.completion_tokens

    def cost(self, settings: Settings) -> float:
        return self.screening.cost(
            settings.screening_cache_hit_price_per_million_rmb,
            settings.screening_input_price_per_million_rmb,
            settings.screening_output_price_per_million_rmb,
        ) + self.review.cost(
            settings.review_cache_hit_price_per_million_rmb,
            settings.review_input_price_per_million_rmb,
            settings.review_output_price_per_million_rmb,
        )


def _analysis_fingerprint(
    model: str, prompt: str, service: str, revision: str,
    response_model: type[BaseModel], thinking: bool, max_tokens: int,
    reasoning_effort: str | None = None,
) -> str:
    contract = [
        "json-object-v2", service, revision, model, prompt,
        response_model.model_json_schema(), thinking, max_tokens,
    ]
    if reasoning_effort is not None:
        contract.append(reasoning_effort)
    payload = json.dumps(contract, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return sha256(payload).hexdigest()


class OutputLimitExceeded(RuntimeError):
    """A truncated response needs a configuration change, not an identical retry."""


def failure_kind(error: Exception) -> str:
    if isinstance(error, OutputLimitExceeded):
        return "output_limit"
    if isinstance(error, (AuthenticationError, PermissionDeniedError)):
        return "authentication"
    if isinstance(error, (BadRequestError, NotFoundError)):
        return "configuration"
    if isinstance(error, (ValueError, TypeError)):
        return "invalid_response"
    return "transient"


class APICallBudgetExceeded(RuntimeError):
    pass


class Analyzer(Protocol):
    screening_fingerprint: str
    review_fingerprint: str
    usage: AnalysisUsage
    request_guard: Callable[[], bool] | None

    def screen(self, paper: Paper) -> Screening: ...

    def review(self, paper: Paper, screening: Screening) -> Review: ...


class DeepSeekAnalyzer:
    def __init__(
        self,
        screening_model: str,
        review_model: str,
        screen_prompt: str,
        review_prompt: str,
        timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        review_max_tokens: int = DEFAULT_REVIEW_MAX_TOKENS,
        review_reasoning_effort: str = DEFAULT_REVIEW_REASONING_EFFORT,
    ) -> None:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is missing. Use --demo for an offline test."
            )
        base_url = (
            os.getenv("DEEPSEEK_BASE_URL", "").strip().rstrip("/")
            or "https://api.deepseek.com"
        )
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=0,
        )
        self.screening_model = screening_model
        self.review_model = review_model
        self.review_max_tokens = review_max_tokens
        self.review_reasoning_effort = review_reasoning_effort
        self.screen_prompt = screen_prompt
        self.review_prompt = review_prompt
        address = urlsplit(str(self.client.base_url))
        service = address._replace(
            netloc=address.netloc.rsplit("@", 1)[-1], query="", fragment="",
        ).geturl().rstrip("/")
        revision = os.getenv("ANALYSIS_REVISION", "1")
        self.screening_fingerprint = _analysis_fingerprint(
            screening_model, screen_prompt, service, revision,
            Screening, False, SCREEN_TOKENS,
        )
        self.review_fingerprint = _analysis_fingerprint(
            review_model, review_prompt, service, revision,
            Review, True, review_max_tokens, review_reasoning_effort,
        )
        self.usage = AnalysisUsage()
        self.request_guard: Callable[[], bool] | None = None

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
        stage: str,
    ) -> T:
        if self.request_guard is not None and not self.request_guard():
            raise APICallBudgetExceeded("API call budget exhausted")
        stage_usage = self.usage.screening if stage == "screening" else self.usage.review
        stage_usage.calls += 1
        try:
            options = {"reasoning_effort": self.review_reasoning_effort} if thinking else {}
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
                    "thinking": {
                        "type": "enabled" if thinking else "disabled"
                    }
                },
                **options,
            )
            if (
                getattr(response.usage, "prompt_tokens", None) is None
                or getattr(response.usage, "completion_tokens", None) is None
            ):
                self.usage.unmetered_calls += 1
            stage_usage.add(response.usage)
            choice = response.choices[0]
            finish = getattr(choice, "finish_reason", None)
            stage_usage.last_finish_reason = finish if finish in {
                "stop", "length", "content_filter", "tool_calls",
                "insufficient_system_resource", "aborted",
            } else "unknown"
            if finish == "length":
                raise OutputLimitExceeded("Completion stopped with finish_reason=length")
            content = choice.message.content
            if not content or not content.strip():
                raise ValueError("Model returned empty content")
            return response_model.model_validate_json(content)
        except Exception as error:
            self.usage.failed_calls += 1
            self.usage.last_failure_kind = failure_kind(error)
            raise

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
            max_tokens=SCREEN_TOKENS,
            stage="screening",
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
            max_tokens=self.review_max_tokens,
            stage="review",
        )


class DemoAnalyzer:
    """Deterministic analyzer for installation checks; never calls an API."""

    screening_fingerprint = "demo-screen-v1"
    review_fingerprint = "demo-review-v1"

    def __init__(self) -> None:
        self.usage = AnalysisUsage()
        self.request_guard: Callable[[], bool] | None = None

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
