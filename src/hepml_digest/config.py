from __future__ import annotations

import os
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path


DEFAULT_CATEGORIES = (
    "stat.ML",
    "cs.LG",
    "physics.data-an",
    "hep-ex",
)


def _int_env(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _float_env(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


@dataclass(slots=True)
class Settings:
    categories: tuple[str, ...] = field(default_factory=lambda: DEFAULT_CATEGORIES)
    screening_model: str = "deepseek-v4-flash"
    review_model: str = "deepseek-v4-pro"
    site_url: str = "http://127.0.0.1:8000"
    feed_title: str = "Daily HEP-ML Digest"
    state_file: Path = Path("data/state.json")
    output_dir: Path = Path("public")
    prompt_dir: Path = Path("prompts")
    max_candidates: int = 60
    bootstrap_results: int = 120
    method_candidate_slots: int = 40
    hep_application_slots: int = 10
    discovery_slots: int = 10
    max_deep_reviews: int = 5
    min_deep_reviews: int = 3
    publish_threshold: float = 0.55
    review_threshold: float = 0.45
    feed_max_items: int = 300
    state_retention_days: int = 365
    request_timeout_seconds: float = 90.0
    api_time_budget_seconds: float = 1440.0
    max_consecutive_api_failures: int = 5
    max_prompt_tokens: int = 300_000
    reanalysis_slots: int = 5
    checkpoint_interval: int = 5
    screening_cache_hit_price_per_million_rmb: float = 0.02
    screening_input_price_per_million_rmb: float = 1.0
    screening_output_price_per_million_rmb: float = 2.0
    review_cache_hit_price_per_million_rmb: float = 0.025
    review_input_price_per_million_rmb: float = 3.0
    review_output_price_per_million_rmb: float = 6.0
    feedback_repository: str = ""
    user_agent: str = (
        "hepml-digest/0.1 (+personal research digest; "
        "contact configured by repository owner)"
    )

    def __post_init__(self) -> None:
        nonnegative_values = {
            "max_candidates": self.max_candidates,
            "bootstrap_results": self.bootstrap_results,
            "method_candidate_slots": self.method_candidate_slots,
            "hep_application_slots": self.hep_application_slots,
            "discovery_slots": self.discovery_slots,
            "max_deep_reviews": self.max_deep_reviews,
            "min_deep_reviews": self.min_deep_reviews,
            "feed_max_items": self.feed_max_items,
            "state_retention_days": self.state_retention_days,
            "api_time_budget_seconds": self.api_time_budget_seconds,
            "max_prompt_tokens": self.max_prompt_tokens,
            "reanalysis_slots": self.reanalysis_slots,
            "screening_cache_hit_price_per_million_rmb": (
                self.screening_cache_hit_price_per_million_rmb
            ),
            "screening_input_price_per_million_rmb": (
                self.screening_input_price_per_million_rmb
            ),
            "screening_output_price_per_million_rmb": (
                self.screening_output_price_per_million_rmb
            ),
            "review_cache_hit_price_per_million_rmb": (
                self.review_cache_hit_price_per_million_rmb
            ),
            "review_input_price_per_million_rmb": (
                self.review_input_price_per_million_rmb
            ),
            "review_output_price_per_million_rmb": (
                self.review_output_price_per_million_rmb
            ),
        }
        for name, value in nonnegative_values.items():
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name, value in (
            ("publish_threshold", self.publish_threshold),
            ("review_threshold", self.review_threshold),
        ):
            if not isfinite(value) or not 0 <= value <= 1:
                raise ValueError(
                    f"{name} must be finite and between 0 and 1"
                )
        if self.min_deep_reviews > self.max_deep_reviews:
            raise ValueError(
                "min_deep_reviews cannot exceed max_deep_reviews"
            )
        if (
            not isfinite(self.request_timeout_seconds)
            or self.request_timeout_seconds <= 0
        ):
            raise ValueError(
                "request_timeout_seconds must be finite and positive"
            )
        if self.max_consecutive_api_failures <= 0:
            raise ValueError("max_consecutive_api_failures must be positive")
        if self.checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive")

    @classmethod
    def from_env(cls) -> "Settings":
        raw_categories = os.getenv("ARXIV_CATEGORIES", "")
        categories = (
            tuple(x.strip() for x in raw_categories.split(",") if x.strip())
            if raw_categories
            else DEFAULT_CATEGORIES
        )
        return cls(
            categories=categories,
            screening_model=os.getenv(
                "SCREENING_MODEL", "deepseek-v4-flash"
            ),
            review_model=os.getenv("REVIEW_MODEL", "deepseek-v4-pro"),
            site_url=os.getenv(
                "SITE_URL", "http://127.0.0.1:8000"
            ).rstrip("/"),
            feed_title=os.getenv("FEED_TITLE", "Daily HEP-ML Digest"),
            state_file=Path(os.getenv("STATE_FILE", "data/state.json")),
            output_dir=Path(os.getenv("OUTPUT_DIR", "public")),
            prompt_dir=Path(os.getenv("PROMPT_DIR", "prompts")),
            max_candidates=_int_env("MAX_CANDIDATES", 60),
            bootstrap_results=_int_env("BOOTSTRAP_RESULTS", 120),
            method_candidate_slots=_int_env("METHOD_CANDIDATE_SLOTS", 40),
            hep_application_slots=_int_env("HEP_APPLICATION_SLOTS", 10),
            discovery_slots=_int_env("DISCOVERY_SLOTS", 10),
            max_deep_reviews=_int_env("MAX_DEEP_REVIEWS", 5),
            min_deep_reviews=_int_env("MIN_DEEP_REVIEWS", 3),
            publish_threshold=_float_env("PUBLISH_THRESHOLD", 0.55),
            review_threshold=_float_env("REVIEW_THRESHOLD", 0.45),
            feed_max_items=_int_env("FEED_MAX_ITEMS", 300),
            state_retention_days=_int_env("STATE_RETENTION_DAYS", 365),
            request_timeout_seconds=_float_env(
                "REQUEST_TIMEOUT_SECONDS", 90.0
            ),
            api_time_budget_seconds=_float_env(
                "API_TIME_BUDGET_SECONDS", 1440.0
            ),
            max_consecutive_api_failures=_int_env(
                "MAX_CONSECUTIVE_API_FAILURES", 5
            ),
            max_prompt_tokens=_int_env("MAX_PROMPT_TOKENS", 300_000),
            reanalysis_slots=_int_env("REANALYSIS_SLOTS", 5),
            checkpoint_interval=_int_env("CHECKPOINT_INTERVAL", 5),
            screening_cache_hit_price_per_million_rmb=_float_env(
                "SCREENING_CACHE_HIT_PRICE_PER_MILLION_RMB", 0.02
            ),
            screening_input_price_per_million_rmb=_float_env(
                "SCREENING_INPUT_PRICE_PER_MILLION_RMB", 1.0
            ),
            screening_output_price_per_million_rmb=_float_env(
                "SCREENING_OUTPUT_PRICE_PER_MILLION_RMB", 2.0
            ),
            review_cache_hit_price_per_million_rmb=_float_env(
                "REVIEW_CACHE_HIT_PRICE_PER_MILLION_RMB", 0.025
            ),
            review_input_price_per_million_rmb=_float_env(
                "REVIEW_INPUT_PRICE_PER_MILLION_RMB", 3.0
            ),
            review_output_price_per_million_rmb=_float_env(
                "REVIEW_OUTPUT_PRICE_PER_MILLION_RMB", 6.0
            ),
            feedback_repository=os.getenv("FEEDBACK_REPOSITORY", "").strip(
                "/"
            ),
            user_agent=os.getenv(
                "ARXIV_USER_AGENT",
                "hepml-digest/0.1 (+personal research digest)",
            ),
        )

    @property
    def feed_urls(self) -> tuple[str, ...]:
        return tuple(
            f"https://rss.arxiv.org/rss/{category}"
            for category in self.categories
        )
