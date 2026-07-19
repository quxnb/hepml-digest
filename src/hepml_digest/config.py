from __future__ import annotations

import os
from dataclasses import dataclass, field
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
    discovery_slots: int = 5
    max_deep_reviews: int = 5
    publish_threshold: float = 0.55
    review_threshold: float = 0.72
    feed_max_items: int = 300
    state_retention_days: int = 365
    request_timeout_seconds: float = 90.0
    user_agent: str = (
        "hepml-digest/0.1 (+personal research digest; "
        "contact configured by repository owner)"
    )

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
            discovery_slots=_int_env("DISCOVERY_SLOTS", 5),
            max_deep_reviews=_int_env("MAX_DEEP_REVIEWS", 5),
            publish_threshold=_float_env("PUBLISH_THRESHOLD", 0.55),
            review_threshold=_float_env("REVIEW_THRESHOLD", 0.72),
            feed_max_items=_int_env("FEED_MAX_ITEMS", 300),
            state_retention_days=_int_env("STATE_RETENTION_DAYS", 365),
            request_timeout_seconds=_float_env(
                "REQUEST_TIMEOUT_SECONDS", 90.0
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
