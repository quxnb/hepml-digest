from __future__ import annotations

import logging
from datetime import datetime, timezone

from .config import Settings
from .fetch import fetch_feeds, fetch_recent, merge_duplicates, select_candidates
from .llm import Analyzer
from .models import Paper, Record
from .publish import publish
from .store import load_state, prune_state, save_state


LOGGER = logging.getLogger(__name__)


def run_pipeline(
    settings: Settings,
    analyzer: Analyzer,
    papers: list[Paper] | None = None,
) -> dict[str, int]:
    state = load_state(settings.state_file)
    prune_state(state, settings.state_retention_days)
    bootstrap_fetched = 0

    if papers is None:
        papers = fetch_feeds(
            settings.feed_urls,
            settings.user_agent,
            timeout=min(settings.request_timeout_seconds, 60.0),
        )
        if not state.records and settings.bootstrap_results > 0:
            bootstrap_papers = fetch_recent(
                settings.categories,
                settings.bootstrap_results,
                settings.user_agent,
                timeout=min(settings.request_timeout_seconds, 60.0),
            )
            bootstrap_fetched = len(bootstrap_papers)
            papers = merge_duplicates([*papers, *bootstrap_papers])

    unseen = [
        paper for paper in papers if paper.version_key not in state.records
    ]
    candidates = select_candidates(
        unseen,
        settings.max_candidates,
        settings.discovery_slots,
        settings.method_candidate_slots,
        settings.hep_application_slots,
    )
    LOGGER.info(
        "Fetched %d unique papers; %d unseen; screening %d",
        len(papers),
        len(unseen),
        len(candidates),
    )

    screened = 0
    failed_screening = 0
    new_records: list[Record] = []
    now = datetime.now(timezone.utc)
    for paper in candidates:
        try:
            screening = analyzer.screen(paper)
        except Exception:
            failed_screening += 1
            LOGGER.exception("Screening failed for %s", paper.version_key)
            continue
        review_status = (
            "pending"
            if screening.needs_deep_review
            and screening.relevance >= settings.review_threshold
            else "not_selected"
        )
        record = Record(
            paper=paper,
            screening=screening,
            review_status=review_status,
            processed_at=now,
            screening_model=settings.screening_model,
        )
        state.records[paper.version_key] = record
        new_records.append(record)
        screened += 1

    # The model's needs_deep_review flag remains the main signal. If it yields
    # too few papers, promote the strongest non-irrelevant papers from today's
    # method radar first, so that promising transfer ideas are actually tested.
    review_target = min(
        max(settings.min_deep_reviews, 0),
        max(settings.max_deep_reviews, 0),
    )
    already_selected = sum(
        record.review_status == "pending" for record in new_records
    )
    promotion_candidates = sorted(
        (
            record
            for record in new_records
            if record.review_status == "not_selected"
            and record.screening.evidence_level != "irrelevant"
        ),
        key=lambda record: (
            record.paper.digest_track == "method_radar",
            record.screening.relevance,
            record.processed_at,
        ),
        reverse=True,
    )
    promoted_reviews = 0
    promotion_limit = max(0, review_target - already_selected)
    for record in promotion_candidates[:promotion_limit]:
        record.review_status = "pending"
        promoted_reviews += 1

    pending = sorted(
        (
            record
            for record in state.records.values()
            if record.review_status == "pending"
        ),
        key=lambda record: (
            record.screening.relevance,
            record.processed_at,
        ),
        reverse=True,
    )[: settings.max_deep_reviews]

    reviewed = 0
    failed_reviews = 0
    for record in pending:
        try:
            record.review = analyzer.review(record.paper, record.screening)
            record.review_status = "complete"
            record.review_model = settings.review_model
            reviewed += 1
        except Exception:
            failed_reviews += 1
            LOGGER.exception("Review failed for %s", record.paper.version_key)

    published = publish(
        state,
        settings.output_dir,
        settings.site_url,
        settings.feed_title,
        settings.publish_threshold,
        settings.feed_max_items,
        settings.feedback_repository,
    )
    save_state(settings.state_file, state)
    return {
        "fetched": len(papers),
        "bootstrap_fetched": bootstrap_fetched,
        "unseen": len(unseen),
        "screened": screened,
        "reviewed": reviewed,
        "promoted_reviews": promoted_reviews,
        "published": published,
        "failed_screening": failed_screening,
        "failed_reviews": failed_reviews,
    }
