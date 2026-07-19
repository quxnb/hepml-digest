from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, field_validator


Evidence = Literal["direct", "transferable", "speculative", "irrelevant"]


class Paper(BaseModel):
    arxiv_id: str
    version: int = Field(default=1, ge=1)
    title: str
    abstract: str
    authors: list[str] = Field(default_factory=list)
    link: str
    published: datetime
    updated: datetime
    categories: list[str] = Field(default_factory=list)

    @field_validator("published", "updated")
    @classmethod
    def ensure_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    @property
    def version_key(self) -> str:
        return f"{self.arxiv_id}:v{self.version}"


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
    processed_at: datetime
    screening_model: str
    review_model: str | None = None


class State(BaseModel):
    schema_version: int = 1
    records: dict[str, Record] = Field(default_factory=dict)
