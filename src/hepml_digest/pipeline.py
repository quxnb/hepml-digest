from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from .config import Settings
from .fetch import fetch_feeds, fetch_recent, merge_duplicates, select_candidates
from .llm import Analyzer
from .models import Paper, Record
from .publish import publish
from .store import load_state, prune_state, save_state, state_lock


LOGGER = logging.getLogger(__name__)


def run_pipeline(
    settings: Settings,
    analyzer: Analyzer,
    papers: list[Paper] | None = None,
) -> dict[str, int | float]:
    with state_lock(settings.state_file):
        return _run_pipeline(settings, analyzer, papers)


def _run_pipeline(
    settings: Settings,
    analyzer: Analyzer,
    papers: list[Paper] | None = None,
) -> dict[str, int | float]:
    state = load_state(settings.state_file)
    prune_state(state, settings.state_retention_days)
    state.schema_version = 2
    bootstrap_fetched = 0
    started_at = time.monotonic()
    consecutive_api_failures = 0
    api_calls_stopped = False
    changes_since_checkpoint = 0

    def checkpoint() -> None:
        nonlocal changes_since_checkpoint
        changes_since_checkpoint += 1
        if changes_since_checkpoint >= settings.checkpoint_interval:
            save_state(settings.state_file, state)
            changes_since_checkpoint = 0
    screening_fingerprint = getattr(
        analyzer,
        "screening_fingerprint",
        f"model:{settings.screening_model}",
    )
    review_fingerprint = getattr(
        analyzer,
        "review_fingerprint",
        f"model:{settings.review_model}",
    )

    for record in state.records.values():
        if (
            not record.screening_fingerprint
            and record.screening_model == settings.screening_model
        ):
            record.screening_fingerprint = screening_fingerprint
        if record.screening_fingerprint != screening_fingerprint:
            continue
        if (
            record.review_status == "complete"
            and not record.review_fingerprint
            and record.review_model == settings.review_model
        ):
            record.review_fingerprint = review_fingerprint
        if (
            record.review_status == "complete"
            and record.review_fingerprint != review_fingerprint
        ):
            record.review = None
            record.review_status = "pending"
            record.review_model = None
            record.review_fingerprint = None

    def api_calls_allowed() -> bool:
        nonlocal api_calls_stopped
        if api_calls_stopped:
            return False
        elapsed = time.monotonic() - started_at
        usage = getattr(analyzer, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        if elapsed >= settings.api_time_budget_seconds:
            api_calls_stopped = True
            LOGGER.warning(
                "Stopping API calls after reaching the %.0f second budget",
                settings.api_time_budget_seconds,
            )
        elif prompt_tokens >= settings.max_prompt_tokens:
            api_calls_stopped = True
            LOGGER.warning(
                "Stopping API calls after %d prompt tokens", prompt_tokens
            )
        elif (
            consecutive_api_failures
            >= settings.max_consecutive_api_failures
        ):
            api_calls_stopped = True
            LOGGER.warning(
                "Stopping API calls after %d consecutive failures",
                consecutive_api_failures,
            )
        return not api_calls_stopped

    set_request_guard = getattr(analyzer, "set_request_guard", None)
    if callable(set_request_guard):
        set_request_guard(api_calls_allowed)
    analyzer_tracks_attempt_failures = False
    set_failure_callback = getattr(
        analyzer, "set_request_failure_callback", None
    )
    if callable(set_failure_callback):
        analyzer_tracks_attempt_failures = True

        def record_attempt_failure() -> None:
            nonlocal consecutive_api_failures
            consecutive_api_failures += 1
            api_calls_allowed()

        set_failure_callback(record_attempt_failure)

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

    new_fetched_papers = [
        paper
        for paper in papers
        if paper.version_key not in state.records
    ]
    stale_fetched_papers = [
        paper
        for paper in papers
        if paper.version_key in state.records
        and state.records[paper.version_key].screening_fingerprint
        != screening_fingerprint
    ]
    fetched_keys = {paper.version_key for paper in papers}
    historical_stale_records = sorted(
        (
            record
            for record in state.records.values()
            if record.screening_fingerprint != screening_fingerprint
            and record.paper.version_key not in fetched_keys
        ),
        key=lambda record: (record.processed_at, record.paper.updated),
        reverse=True,
    )
    reserved_reanalysis_slots = min(
        settings.reanalysis_slots,
        len(stale_fetched_papers) + len(historical_stale_records),
        settings.max_candidates,
    )
    fresh_limit = settings.max_candidates - reserved_reanalysis_slots
    candidates = select_candidates(
        new_fetched_papers,
        fresh_limit,
        settings.discovery_slots,
        settings.method_candidate_slots,
        settings.hep_application_slots,
    )
    remaining_slots = max(0, settings.max_candidates - len(candidates))
    stale_candidates = select_candidates(
        [
            *stale_fetched_papers,
            *(record.paper for record in historical_stale_records),
        ],
        remaining_slots,
        settings.discovery_slots,
        settings.method_candidate_slots,
        settings.hep_application_slots,
    )
    candidates.extend(stale_candidates)
    requiring_analysis = (
        len(new_fetched_papers)
        + len(stale_fetched_papers)
        + len(historical_stale_records)
    )
    LOGGER.info(
        "Fetched %d unique papers; %d require analysis; screening %d",
        len(papers),
        requiring_analysis,
        len(candidates),
    )

    screened = 0
    failed_screening = 0
    new_records: list[Record] = []
    now = datetime.now(timezone.utc)
    for paper in candidates:
        if not api_calls_allowed():
            break
        try:
            screening = analyzer.screen(paper)
        except Exception:
            failed_screening += 1
            if not analyzer_tracks_attempt_failures:
                consecutive_api_failures += 1
            LOGGER.exception("Screening failed for %s", paper.version_key)
            api_calls_allowed()
            continue
        consecutive_api_failures = 0
        review_status = (
            "pending"
            if screening.needs_deep_review
            and screening.relevance >= settings.review_threshold
            and screening.evidence_level != "irrelevant"
            else "not_selected"
        )
        record = Record(
            paper=paper,
            screening=screening,
            review_status=review_status,
            processed_at=now,
            screening_model=settings.screening_model,
            screening_fingerprint=screening_fingerprint,
        )
        state.records[paper.version_key] = record
        new_records.append(record)
        screened += 1
        checkpoint()
        api_calls_allowed()

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
    if promoted_reviews:
        save_state(settings.state_file, state)
        changes_since_checkpoint = 0

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
        if not api_calls_allowed():
            break
        try:
            record.review = analyzer.review(record.paper, record.screening)
            record.review_status = "complete"
            record.review_model = settings.review_model
            record.review_fingerprint = review_fingerprint
            reviewed += 1
            consecutive_api_failures = 0
            checkpoint()
            api_calls_allowed()
        except Exception:
            failed_reviews += 1
            if not analyzer_tracks_attempt_failures:
                consecutive_api_failures += 1
            LOGGER.exception("Review failed for %s", record.paper.version_key)
            api_calls_allowed()

    save_state(settings.state_file, state)
    published = publish(
        state,
        settings.output_dir,
        settings.site_url,
        settings.feed_title,
        settings.publish_threshold,
        settings.feed_max_items,
        settings.feedback_repository,
    )
    usage = getattr(analyzer, "usage", None)
    screening_calls = int(getattr(usage, "screening_calls", 0) or 0)
    review_calls = int(getattr(usage, "review_calls", 0) or 0)
    screening_prompt_tokens = int(
        getattr(usage, "screening_prompt_tokens", 0) or 0
    )
    screening_cache_hit_tokens = int(
        getattr(usage, "screening_cache_hit_tokens", 0) or 0
    )
    screening_cache_miss_tokens = int(
        getattr(usage, "screening_cache_miss_tokens", 0) or 0
    )
    screening_cache_miss_tokens += max(
        0,
        screening_prompt_tokens
        - screening_cache_hit_tokens
        - screening_cache_miss_tokens,
    )
    screening_completion_tokens = int(
        getattr(usage, "screening_completion_tokens", 0) or 0
    )
    review_prompt_tokens = int(
        getattr(usage, "review_prompt_tokens", 0) or 0
    )
    review_cache_hit_tokens = int(
        getattr(usage, "review_cache_hit_tokens", 0) or 0
    )
    review_cache_miss_tokens = int(
        getattr(usage, "review_cache_miss_tokens", 0) or 0
    )
    review_cache_miss_tokens += max(
        0,
        review_prompt_tokens
        - review_cache_hit_tokens
        - review_cache_miss_tokens,
    )
    review_completion_tokens = int(
        getattr(usage, "review_completion_tokens", 0) or 0
    )
    prompt_tokens = screening_prompt_tokens + review_prompt_tokens
    completion_tokens = (
        screening_completion_tokens + review_completion_tokens
    )
    estimated_cost_rmb = (
        screening_cache_hit_tokens
        * settings.screening_cache_hit_price_per_million_rmb
        + screening_cache_miss_tokens
        * settings.screening_input_price_per_million_rmb
        + screening_completion_tokens
        * settings.screening_output_price_per_million_rmb
        + review_cache_hit_tokens
        * settings.review_cache_hit_price_per_million_rmb
        + review_cache_miss_tokens
        * settings.review_input_price_per_million_rmb
        + review_completion_tokens
        * settings.review_output_price_per_million_rmb
    ) / 1_000_000
    return {
        "fetched": len(papers),
        "bootstrap_fetched": bootstrap_fetched,
        "unseen": requiring_analysis,
        "screened": screened,
        "reviewed": reviewed,
        "promoted_reviews": promoted_reviews,
        "published": published,
        "failed_screening": failed_screening,
        "failed_reviews": failed_reviews,
        "api_calls_stopped": int(api_calls_stopped),
        "screening_api_calls": screening_calls,
        "review_api_calls": review_calls,
        "prompt_tokens": prompt_tokens,
        "prompt_cache_hit_tokens": (
            screening_cache_hit_tokens + review_cache_hit_tokens
        ),
        "prompt_cache_miss_tokens": (
            screening_cache_miss_tokens + review_cache_miss_tokens
        ),
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "estimated_cost_rmb": round(estimated_cost_rmb, 6),
    }
