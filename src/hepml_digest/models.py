from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, Field, field_validator, model_validator


Evidence = Literal["direct", "transferable", "speculative", "irrelevant"]
DigestTrack = Literal["method_radar", "hep_application"]


def _with_timezone(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


UTCDateTime = Annotated[datetime, AfterValidator(_with_timezone)]

HEP_APPLICATION_CATEGORIES = frozenset(
    {"hep-ex", "physics.ins-det", "nucl-ex", "inspire:Experiment-HEP",
     "inspire:Experiment-Nucl", "inspire:Instrumentation"}
)


class Paper(BaseModel):
    arxiv_id: str | None = None
    version: int | None = Field(default=1, ge=1)
    doi: str | None = None
    inspire_id: str | None = None
    journal_ref: str | None = None
    abstract_source: str | None = None
    # Assigned only when enrichment would otherwise change an existing ID.
    identity: str | None = None
    feed_identity: str | None = None
    title: str
    abstract: str
    authors: list[str] = Field(default_factory=list)
    link: str
    published: UTCDateTime
    updated: UTCDateTime
    categories: list[str] = Field(default_factory=list)

    @field_validator("doi")
    @classmethod
    def normalize_doi(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", value.strip(), flags=re.I)
        return normalized.strip().lower() or None

    @model_validator(mode="after")
    def require_identifier(self) -> "Paper":
        if not (self.arxiv_id or self.doi or self.inspire_id):
            raise ValueError("Paper requires an arXiv ID, DOI or INSPIRE ID")
        if not self.arxiv_id:
            self.version = None
        return self

    @property
    def version_key(self) -> str:
        if self.identity:
            return self.identity
        if self.arxiv_id:
            return f"{self.arxiv_id}:v{self.version}" if self.version is not None else f"arxiv:{self.arxiv_id}"
        return f"doi:{self.doi}" if self.doi else f"inspire:{self.inspire_id}"

    @property
    def feed_key(self) -> str:
        if self.feed_identity:
            return self.feed_identity
        if self.arxiv_id:
            return f"arxiv:{self.arxiv_id}:hepml-v1"
        return f"{self.version_key}:hepml-v1"

    @property
    def aliases(self) -> list[str]:
        return [f"{kind}:{value}" for kind, value in (
            ("arxiv", self.arxiv_id), ("doi", self.doi), ("inspire", self.inspire_id),
        ) if value]

    @property
    def identifier(self) -> str:
        if self.arxiv_id:
            return f"arXiv:{self.arxiv_id}"
        return f"DOI:{self.doi}" if self.doi else f"INSPIRE:{self.inspire_id}"

    @property
    def digest_track(self) -> DigestTrack:
        if HEP_APPLICATION_CATEGORIES.intersection(self.categories):
            return "hep_application"
        return "method_radar"


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
    schema_version: int = 5
    records: dict[str, Record] = Field(default_factory=dict)
    pending_papers: dict[str, Paper] = Field(default_factory=dict)
    last_backfill_at: UTCDateTime | None = None
    pending_backfill_since: UTCDateTime | None = None
    last_inspire_fetch_at: UTCDateTime | None = None
    pending_inspire_since: UTCDateTime | None = None
    work: dict[str, WorkState] = Field(default_factory=dict)
    health: RunHealth = Field(default_factory=RunHealth)
