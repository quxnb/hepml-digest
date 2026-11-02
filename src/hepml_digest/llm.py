from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Callable, Protocol, TypeVar
from urllib.parse import urlsplit

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    InternalServerError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import (
    DEFAULT_LLM_API_STYLE,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_REVIEW_MAX_TOKENS,
    DEFAULT_REVIEW_REASONING_EFFORT,
)
from .models import Paper, Review, Screening


T = TypeVar("T", bound=BaseModel)
LOGGER = logging.getLogger(__name__)
SCREEN_TOKENS = 900
OPENAI_SCREEN_TOKENS = 4096
TERMINAL_API_FAILURES = frozenset({"authentication", "configuration", "output_limit"})


@dataclass(slots=True)
class StageUsage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    last_finish_reason: str = ""

    def add(self, usage) -> bool:
        prompt = getattr(usage, "prompt_tokens", None)
        completion = getattr(usage, "completion_tokens", None)
        self.prompt_tokens += int(prompt or 0)
        self.completion_tokens += int(completion or 0)
        details = getattr(usage, "completion_tokens_details", None)
        self.reasoning_tokens += int(getattr(details, "reasoning_tokens", 0) or 0)
        return prompt is not None and completion is not None


@dataclass(slots=True)
class AnalysisUsage:
    screening: StageUsage = field(default_factory=StageUsage)
    review: StageUsage = field(default_factory=StageUsage)
    failed_calls: int = 0
    last_failure_kind: str = ""
    missing_usage_calls: int = 0

    @property
    def prompt_tokens(self) -> int:
        return self.screening.prompt_tokens + self.review.prompt_tokens

    @property
    def completion_tokens(self) -> int:
        return self.screening.completion_tokens + self.review.completion_tokens


def _analysis_fingerprint(
    model: str, prompt: str, service: str, revision: str,
    response_model: type[BaseModel], thinking: bool, max_tokens: int,
    reasoning_effort: str | None = None,
    api_style: str = DEFAULT_LLM_API_STYLE,
) -> str:
    contract = [
        "json-object-v2", service, revision, model, prompt,
        response_model.model_json_schema(), thinking, max_tokens,
    ]
    if reasoning_effort is not None:
        contract.append(reasoning_effort)
    if api_style != "deepseek":
        contract.append({"api_style": api_style})
    payload = json.dumps(contract, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return sha256(payload).hexdigest()


class OutputLimitExceeded(RuntimeError):
    """A truncated response needs a configuration change, not an identical retry."""


def failure_kind(error: Exception) -> str:
    if isinstance(error, OutputLimitExceeded):
        return "output_limit"
    if isinstance(error, (AuthenticationError, PermissionDeniedError)):
        return "authentication"
    if (
        isinstance(error, APIStatusError)
        and 400 <= error.status_code < 500
        and error.status_code not in {408, 409, 429}
    ):
        return "configuration"
    if isinstance(error, (ValueError, TypeError)):
        return "invalid_response"
    return "transient"


class APICallBudgetExceeded(RuntimeError):
    pass


def _connection_hint(error: Exception) -> str:
    """Inspect transport messages locally; return only a fixed, non-sensitive tag."""
    if not isinstance(error, APIConnectionError):
        return "not_applicable"
    markers = {
        "dns": ("name or service not known", "temporary failure in name resolution", "getaddrinfo failed"),
        "tls_certificate": ("certificate verify failed", "certificate_verify_failed"),
        "tls_protocol": ("wrong version number", "eof occurred in violation of protocol"),
        "connection_refused": ("connection refused",),
        "network_unreachable": ("network is unreachable", "no route to host"),
        "connection_reset": ("connection reset", "connection aborted"),
        "timeout": ("timed out",),
    }
    cause: BaseException | None = error
    for _ in range(6):
        if cause is None:
            break
        message = str(cause).lower()
        for tag, phrases in markers.items():
            if any(phrase in message for phrase in phrases):
                return tag
        cause = cause.__cause__ or cause.__context__
    return "unknown"


class Analyzer(Protocol):
    screening_fingerprint: str
    review_fingerprint: str
    usage: AnalysisUsage
    request_guard: Callable[[], bool] | None

    def screen(self, paper: Paper) -> Screening: ...

    def review(self, paper: Paper, screening: Screening) -> Review: ...


class DeepSeekAnalyzer:
    """Chat Completions client; the historical name and key variables are retained."""

    def __init__(
        self,
        screening_model: str,
        review_model: str,
        screen_prompt: str,
        review_prompt: str,
        timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        review_max_tokens: int = DEFAULT_REVIEW_MAX_TOKENS,
        review_reasoning_effort: str = DEFAULT_REVIEW_REASONING_EFFORT,
        api_style: str = DEFAULT_LLM_API_STYLE,
    ) -> None:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is missing. Use --demo for an offline test."
            )
        self.api_style = api_style
        is_openai = api_style == "openai"
        self.screening_max_tokens = OPENAI_SCREEN_TOKENS if is_openai else SCREEN_TOKENS
        default_url = "https://api.openai.com/v1" if is_openai else "https://api.deepseek.com"
        base_url = (
            os.getenv("DEEPSEEK_BASE_URL", "").strip().rstrip("/")
            or default_url
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
            Screening, is_openai, self.screening_max_tokens,
            reasoning_effort="low" if is_openai else None, api_style=api_style,
        )
        self.review_fingerprint = _analysis_fingerprint(
            review_model, review_prompt, service, revision,
            Review, True, review_max_tokens, review_reasoning_effort,
            api_style=api_style,
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
            if self.api_style == "openai":
                options = {
                    "max_completion_tokens": max_tokens,
                    "reasoning_effort": self.review_reasoning_effort if thinking else "low",
                }
            else:
                options = {
                    "max_tokens": max_tokens,
                    "extra_body": {"thinking": {"type": "enabled" if thinking else "disabled"}},
                }
                if thinking:
                    options["reasoning_effort"] = self.review_reasoning_effort
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                stream=False,
                **options,
            )
            if not stage_usage.add(response.usage):
                self.usage.missing_usage_calls += 1
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
            causes = []
            cause = error.__cause__ or error.__context__
            for _ in range(3):
                if cause is None:
                    break
                causes.append(type(cause).__name__)
                cause = cause.__cause__ or cause.__context__
            LOGGER.warning(
                "API request failed: stage=%s attempt=%d exception_type=%s http_status=%s "
                "cause_types=%s connection_hint=%s",
                stage, stage_usage.calls, type(error).__name__,
                error.status_code if isinstance(error, APIStatusError) else "none",
                "/".join(causes) or "none",
                _connection_hint(error),
            )
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
            max_tokens=self.screening_max_tokens,
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
