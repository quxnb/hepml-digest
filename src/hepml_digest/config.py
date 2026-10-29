from __future__ import annotations

import os
from dataclasses import dataclass, fields
from math import isfinite
from pathlib import Path


DEFAULT_CATEGORIES = (
    "stat.ML",
    "cs.LG",
    "physics.data-an",
    "hep-ex",
)
DEFAULT_REVIEW_MAX_TOKENS = 8192
DEFAULT_REVIEW_REASONING_EFFORT = "low"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 120.0


@dataclass(slots=True)
class Settings:
    categories: tuple[str, ...] = DEFAULT_CATEGORIES
    screening_model: str = "deepseek-v4-flash"
    review_model: str = "deepseek-v4-pro"
    review_max_tokens: int = DEFAULT_REVIEW_MAX_TOKENS
    review_reasoning_effort: str = DEFAULT_REVIEW_REASONING_EFFORT
    site_url: str = "http://127.0.0.1:8000"
    feed_title: str = "Daily HEP-ML Digest"
    state_file: Path = Path("data/state.json")
    output_dir: Path = Path("public")
    prompt_dir: Path = Path("prompts")
    max_candidates: int = 60
    bootstrap_results: int = 120
    backfill_lookback_days: int = 7
    backfill_max_pages: int = 25
    method_candidate_slots: int = 40
    hep_application_slots: int = 10
    discovery_slots: int = 10
    max_deep_reviews: int = 5
    publish_threshold: float = 0.55
    review_threshold: float = 0.45
    feed_max_items: int = 300
    state_retention_days: int = 365
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS
    api_time_budget_seconds: float = 1440.0
    screening_budget_fraction: float = 0.7
    max_consecutive_api_failures: int = 5
    max_prompt_tokens: int = 300_000
    reanalysis_slots: int = 5
    checkpoint_interval: int = 5
    aging_slots: int = 5
    max_paper_failures: int = 3
    alert_after_runs: int = 3
    alert_backlog_days: int = 7
    max_estimated_cost_rmb: float = 1.0
    screening_cache_hit_price_per_million_rmb: float = 0.02
    screening_input_price_per_million_rmb: float = 1.0
    screening_output_price_per_million_rmb: float = 4.0
    review_cache_hit_price_per_million_rmb: float = 0.15
    review_input_price_per_million_rmb: float = 4.5
    review_output_price_per_million_rmb: float = 13.5
    feedback_repository: str = ""
    user_agent: str = (
        "hepml-digest/0.1 (+personal research digest; "
        "contact configured by repository owner)"
    )

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, (int, float)) and (not isfinite(value) or value < 0):
                raise ValueError(f"{item.name} must be finite and nonnegative")
        for name in (
            "backfill_lookback_days", "backfill_max_pages", "request_timeout_seconds",
            "screening_budget_fraction", "max_consecutive_api_failures",
            "checkpoint_interval", "max_paper_failures", "alert_after_runs",
            "alert_backlog_days",
            "review_max_tokens",
        ):
            if getattr(self, name) == 0:
                raise ValueError(f"{name} must be positive")
        for name in ("publish_threshold", "review_threshold", "screening_budget_fraction"):
            if getattr(self, name) > 1:
                raise ValueError(f"{name} cannot exceed 1")
        if self.review_reasoning_effort not in {"low", "high", "max"}:
            raise ValueError("review_reasoning_effort must be low, high or max")

    @classmethod
    def from_env(cls) -> "Settings":
        defaults = cls()
        aliases = {"categories": "ARXIV_CATEGORIES", "user_agent": "ARXIV_USER_AGENT"}
        values = {}
        for item in fields(cls):
            raw = os.getenv(aliases.get(item.name, item.name.upper()))
            if raw is None:
                continue
            if item.name == "categories":
                if raw:
                    values[item.name] = tuple(x.strip() for x in raw.split(",") if x.strip())
            else:
                values[item.name] = type(getattr(defaults, item.name))(raw)
        values["site_url"] = values.get("site_url", defaults.site_url).rstrip("/")
        values["feedback_repository"] = values.get("feedback_repository", "").strip("/")
        return cls(**values)

    @property
    def feed_urls(self) -> tuple[str, ...]:
        return tuple(
            f"https://rss.arxiv.org/rss/{category}"
            for category in self.categories
        )
