from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, Field


Evidence = Literal["direct", "transferable", "speculative", "irrelevant"]
DigestTrack = Literal["method_radar", "hep_application"]


def _with_timezone(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


UTCDateTime = Annotated[datetime, AfterValidator(_with_timezone)]

HEP_APPLICATION_CATEGORIES = frozenset(
    {"hep-ex", "physics.ins-det", "nucl-ex"}
)


def digest_track_for_categories(categories: list[str]) -> DigestTrack:
    if HEP_APPLICATION_CATEGORIES.intersection(categories):
        return "hep_application"
    return "method_radar"


class Paper(BaseModel):
    arxiv_id: str
    version: int = Field(default=1, ge=1)
    title: str
    abstract: str
    authors: list[str] = Field(default_factory=list)
    link: str
    published: UTCDateTime
    updated: UTCDateTime
    categories: list[str] = Field(default_factory=list)

    @property
    def version_key(self) -> str:
        return f"{self.arxiv_id}:v{self.version}"

    @property
    def digest_track(self) -> DigestTrack:
        return digest_track_for_categories(self.categories)


class Screening(BaseModel):
    relevance: float = Field(ge=0, le=1)
    method: str
    hep_tasks: list[str] = Field(default_factory=list)
    reason: str
    needs_deep_review: bool
    evidence_level: Evidence


class Review(BaseModel):
    summary_cn: str
    paper_claims: list[str] = Field(default_factory=list)
    hep_opportunities: list[str] = Field(default_factory=list)
    transfer_risks: list[str] = Field(default_factory=list)
    validation_plan: str
    evidence_level: Literal["direct", "transferable", "speculative"]
    confidence: float = Field(ge=0, le=1)


class Record(BaseModel):
    paper: Paper
    screening: Screening
    review: Review | None = None
    review_status: Literal["not_selected", "pending", "complete"]
    processed_at: UTCDateTime
    screening_model: str
    review_model: str | None = None
    screening_fingerprint: str = ""
    review_fingerprint: str | None = None


class WorkState(BaseModel):
    first_seen_at: UTCDateTime
    fingerprint: str
    failures: int = Field(default=0, ge=0)
    last_failure: Literal["", "invalid_response", "transient"] = ""
    next_retry_at: UTCDateTime | None = None
    quarantined: bool = False

    def ready(self, now: datetime) -> bool:
        return not self.quarantined and (
            self.next_retry_at is None or self.next_retry_at <= now
        )


class RunHealth(BaseModel):
    source_failure_runs: int = 0
    analysis_failure_runs: int = 0
    queue_growth_runs: int = 0
    previous_queue_size: int = 0
    last_source_success_at: UTCDateTime | None = None
    last_analysis_success_at: UTCDateTime | None = None


class State(BaseModel):
    schema_version: int = 4
    records: dict[str, Record] = Field(default_factory=dict)
    pending_papers: dict[str, Paper] = Field(default_factory=dict)
    last_backfill_at: UTCDateTime | None = None
    pending_backfill_since: UTCDateTime | None = None
    work: dict[str, WorkState] = Field(default_factory=dict)
    health: RunHealth = Field(default_factory=RunHealth)
