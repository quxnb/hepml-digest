from __future__ import annotations

from hashlib import sha256
from itertools import chain, zip_longest
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, StrictBool, model_validator

from .models import Paper, Review, Screening, State
from .publish import is_publishable


class EvaluationItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper: Paper
    screening: Screening | None = None
    review: Review | None = None
    screening_model: str | None = None
    review_model: str | None = None
    screening_fingerprint: str | None = None
    review_fingerprint: str | None = None
    human_keep: StrictBool | None = None
    hep_mapping_overstated: StrictBool | None = None
    notes: str = ""


class EvaluationSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    items: list[EvaluationItem]

    @model_validator(mode="after")
    def unique_versions(self) -> "EvaluationSet":
        keys = [item.paper.version_key for item in self.items]
        if len(keys) != len(set(keys)):
            raise ValueError("Evaluation contains duplicate paper versions")
        return self


def _latest_papers(state: State) -> dict[str, Paper]:
    latest: dict[str, Paper] = {}
    for paper in chain(
        (record.paper for record in state.records.values()), state.pending_papers.values(),
    ):
        previous = latest.get(paper.arxiv_id)
        if previous is None or paper.version > previous.version:
            latest[paper.arxiv_id] = paper
    return latest


def export_evaluation(
    state: State, path: Path, publish_threshold: float, limit: int = 100,
) -> EvaluationSet:
    """Interleave recommended, screened-out and unprocessed papers for annotation."""
    if limit <= 0:
        raise ValueError("Evaluation sample limit must be positive")
    groups: list[list[EvaluationItem]] = [[], [], []]
    papers = sorted(
        _latest_papers(state).values(),
        key=lambda paper: sha256(paper.version_key.encode("utf-8")).hexdigest(),
    )
    for paper in papers:
        record = state.records.get(paper.version_key)
        item = EvaluationItem(paper=paper)
        if record is None:
            group = 2
        else:
            item.screening = record.screening
            item.review = record.review
            item.screening_model = record.screening_model
            item.review_model = record.review_model
            item.screening_fingerprint = record.screening_fingerprint
            item.review_fingerprint = record.review_fingerprint
            group = 0 if is_publishable(record, publish_threshold) else 1
        groups[group].append(item)
    items = [item for row in zip_longest(*groups) for item in row if item is not None]
    dataset = EvaluationSet(items=items[:limit]).model_copy(deep=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Never overwrite an annotated evaluation file, including the state file.
    with path.open("x", encoding="utf-8") as handle:
        handle.write(dataset.model_dump_json(indent=2) + "\n")
    return dataset


def evaluate(
    state: State, dataset: EvaluationSet, publish_threshold: float,
) -> dict[str, int | float | None]:
    latest = _latest_papers(state)
    tp = fp = fn = tn = 0
    missing = superseded = unprocessed_relevant = 0
    mapping_labeled = overstated = stale_mapping_labels = 0
    for item in dataset.items:
        paper = latest.get(item.paper.arxiv_id)
        if paper is None:
            missing += 1
            continue
        if paper.version_key != item.paper.version_key:
            superseded += 1
            continue
        record = state.records.get(paper.version_key)
        predicted_keep = record is not None and is_publishable(record, publish_threshold)
        if item.human_keep is not None:
            if item.human_keep:
                tp += int(predicted_keep)
                fn += int(not predicted_keep)
                unprocessed_relevant += int(record is None)
            else:
                fp += int(predicted_keep)
                tn += int(not predicted_keep)
        if item.hep_mapping_overstated is not None:
            if (
                record is None or record.screening != item.screening
                or record.review != item.review
                or record.screening_fingerprint != item.screening_fingerprint
                or record.review_fingerprint != item.review_fingerprint
            ):
                stale_mapping_labels += 1
                continue
            mapping_labeled += 1
            overstated += int(item.hep_mapping_overstated)
    return {
        "sample_size": len(dataset.items),
        "labeled": tp + fp + fn + tn,
        "unlabeled": sum(item.human_keep is None for item in dataset.items),
        "missing": missing,
        "superseded": superseded,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "sample_recall": tp / (tp + fn) if tp + fn else None,
        "unprocessed_relevant": unprocessed_relevant,
        "mapping_labeled": mapping_labeled,
        "stale_mapping_labels": stale_mapping_labels,
        "overstated": overstated,
        "overstatement_rate": overstated / mapping_labeled if mapping_labeled else None,
    }
