from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Callable, TypeVar

from .config import Settings
from .fetch import fetch_feeds, fetch_recent, fetch_updates, merge_duplicates, select_candidates
from .llm import APICallBudgetExceeded, TERMINAL_API_FAILURES, Analyzer, failure_kind
from .models import Paper, Record, WorkState
from .publish import is_publishable, publish
from .store import load_state, prune_state, save_state, state_lock


LOGGER = logging.getLogger(__name__)
T = TypeVar("T")
RunResult = dict[str, int | float | str | list[str]]


def run_pipeline(
    settings: Settings,
    analyzer: Analyzer,
    papers: list[Paper] | None = None,
) -> RunResult:
    with state_lock(settings.state_file):
        return _run_pipeline(settings, analyzer, papers)


def _run_pipeline(
    settings: Settings,
    analyzer: Analyzer,
    papers: list[Paper] | None = None,
) -> RunResult:
    state = load_state(settings.state_file)
    prune_state(state, settings.state_retention_days)
    state.schema_version = 4
    bootstrap_fetched = 0
    backfill_fetched = 0
    fetch_errors = 0
    usage = analyzer.usage
    failures_at_last_success = usage.failed_calls
    stop_reason = ""
    screening_budget_exhausted = False
    stage = "screening"
    failed_papers = {"screening": 0, "review": 0}
    changes_since_checkpoint = 0
    now = datetime.now(timezone.utc)

    def checkpoint() -> None:
        nonlocal changes_since_checkpoint
        changes_since_checkpoint += 1
        if changes_since_checkpoint >= settings.checkpoint_interval:
            save_state(settings.state_file, state)
            changes_since_checkpoint = 0

    screening_fingerprint = analyzer.screening_fingerprint
    review_fingerprint = analyzer.review_fingerprint
    review_floor = max(settings.review_threshold, settings.publish_threshold)

    def work_for(stage: str, paper: Paper) -> WorkState:
        key = f"{stage}:{paper.version_key}"
        fingerprint = screening_fingerprint if stage == "screening" else review_fingerprint
        work = state.work.get(key)
        if work is None or work.fingerprint != fingerprint:
            work = WorkState(
                first_seen_at=work.first_seen_at if work else now, fingerprint=fingerprint,
            )
            state.work[key] = work
        return work

    for record in state.records.values():
        if (
            record.screening_fingerprint == screening_fingerprint
            and record.review_status == "complete"
            and record.review_fingerprint != review_fingerprint
        ):
            record.review = None
            record.review_status = "pending"
            record.review_model = None
            record.review_fingerprint = None
        if record.review_status == "pending" and not (
            record.screening.needs_deep_review and is_publishable(record, review_floor)
        ):
            record.review_status = "not_selected"
            state.work.pop(f"review:{record.paper.version_key}", None)

    def api_calls_allowed() -> bool:
        nonlocal stop_reason, screening_budget_exhausted
        if stop_reason:
            return False
        elapsed = time.monotonic() - started_at
        if elapsed >= settings.api_time_budget_seconds:
            stop_reason = "time_budget"
        elif usage.prompt_tokens >= settings.max_prompt_tokens:
            stop_reason = "token_budget"
        elif usage.cost(settings) >= settings.max_estimated_cost_rmb:
            stop_reason = "cost_budget"
        elif usage.unmetered_calls:
            stop_reason = "unmetered_usage"
        elif usage.failed_calls - failures_at_last_success >= settings.max_consecutive_api_failures:
            stop_reason = "consecutive_failures"
        if stop_reason:
            LOGGER.warning("Stopping API calls: %s", stop_reason)
            return False
        if stage == "screening" and settings.max_deep_reviews > 0:
            fraction = settings.screening_budget_fraction
            if (
                elapsed >= settings.api_time_budget_seconds * fraction
                or usage.prompt_tokens >= settings.max_prompt_tokens * fraction
            ):
                if not screening_budget_exhausted:
                    LOGGER.info("Stopping screening to reserve review budget")
                screening_budget_exhausted = True
                return False
        return True

    analyzer.request_guard = api_calls_allowed

    def attempt(stage: str, paper: Paper, operation: Callable[[], T]) -> T | None:
        nonlocal stop_reason, failures_at_last_success
        before = usage.failed_calls
        try:
            result = operation()
        except Exception as error:
            if isinstance(error, APICallBudgetExceeded) and usage.failed_calls == before:
                return None
            failed_papers[stage] += 1
            if not isinstance(error, APICallBudgetExceeded):
                if usage.failed_calls == before:
                    usage.failed_calls += 1
                usage.last_failure_kind = failure_kind(error)
            kind = usage.last_failure_kind
            if kind in TERMINAL_API_FAILURES:
                stop_reason = kind
            else:
                work = work_for(stage, paper)
                if work.last_failure != kind:
                    work.failures = 0
                work.last_failure = kind
                work.failures += 1
                work.quarantined = (
                    kind == "invalid_response" and work.failures >= settings.max_paper_failures
                )
                work.next_retry_at = datetime.now(timezone.utc) + timedelta(
                    hours=min(24, 2 ** min(work.failures - 1, 5)),
                )
            LOGGER.warning("%s failed for %s: %s", stage, paper.version_key, kind)
            checkpoint()
            return None
        failures_at_last_success = usage.failed_calls
        state.work.pop(f"{stage}:{paper.version_key}", None)
        return result

    def enqueue(incoming: list[Paper]) -> None:
        queued = merge_duplicates([*state.pending_papers.values(), *incoming])
        state.pending_papers = {
            paper.version_key: paper
            for paper in queued
            if paper.version_key not in state.records
        }
        for paper in state.pending_papers.values():
            work_for("screening", paper)
        # Persist raw candidates before any model call or subsequent fetch.
        save_state(settings.state_file, state)

    source_ok = None
    if papers is None:
        first_run = (
            not state.records
            and not state.pending_papers
            and state.last_backfill_at is None
            and state.pending_backfill_since is None
        )
        fetched: list[Paper] = []
        failed_urls: list[str] = []
        timeout = min(settings.request_timeout_seconds, 60.0)
        try:
            fetched = fetch_feeds(
                settings.feed_urls, settings.user_agent, timeout=timeout,
                failed_urls=failed_urls,
            )
            source_ok = True
        except (OSError, ValueError, RuntimeError):
            if not failed_urls:
                fetch_errors += 1
            LOGGER.exception("RSS fetch failed; trying backfill and saved candidates")
            source_ok = False
        fetch_errors += len(failed_urls)
        enqueue(fetched)
        backfill_until = datetime.now(timezone.utc)
        try:
            if first_run and settings.bootstrap_results > 0:
                batch = fetch_recent(
                    settings.categories, settings.bootstrap_results,
                    settings.user_agent, timeout=timeout,
                )
                bootstrap_fetched = len(batch)
                fetched.extend(batch)
                enqueue(batch)
            else:
                since = state.pending_backfill_since or (
                    (state.last_backfill_at or backfill_until)
                    - timedelta(days=settings.backfill_lookback_days)
                )
                state.pending_backfill_since = since
                save_state(settings.state_file, state)
                for batch in fetch_updates(
                    settings.categories, since, backfill_until,
                    settings.user_agent, timeout=timeout,
                    max_pages=settings.backfill_max_pages,
                ):
                    backfill_fetched += len(batch)
                    fetched.extend(batch)
                    enqueue(batch)
        except (OSError, ValueError, RuntimeError):
            fetch_errors += 1
            LOGGER.exception("Backfill incomplete; keeping its previous cursor")
        else:
            source_ok = True
            state.last_backfill_at = backfill_until
            state.pending_backfill_since = None
            save_state(settings.state_file, state)
        papers = merge_duplicates(fetched)
        if fetch_errors and not papers and not state.records and not state.pending_papers:
            raise RuntimeError("No papers available after source failures")
    else:
        papers = merge_duplicates(papers)
        enqueue(papers)

    # Source timeouts do not consume the separate model-call time budget.
    started_at = time.monotonic()

    def ready(papers: list[Paper], stage: str) -> list[Paper]:
        return [paper for paper in papers if work_for(stage, paper).ready(now)]

    def schedule(papers: list[Paper], limit: int, stage: str) -> list[Paper]:
        if stage == "screening":
            ranked = select_candidates(
                papers, limit, settings.discovery_slots,
                settings.method_candidate_slots, settings.hep_application_slots,
            )
        else:
            ranked = sorted(papers, key=lambda paper: (
                state.records[paper.version_key].screening.relevance,
                state.records[paper.version_key].processed_at,
            ), reverse=True)[:limit]
        oldest = sorted(
            (paper for paper in papers if work_for(stage, paper).first_seen_at < now),
            key=lambda paper: work_for(stage, paper).first_seen_at,
        )[:min(settings.aging_slots, (limit + 4) // 5)]
        return list({paper.version_key: paper for paper in [*oldest, *ranked]}.values())[:limit]

    stale_papers = [record.paper for record in state.records.values()
                    if record.screening_fingerprint != screening_fingerprint]
    fresh = ready(list(state.pending_papers.values()), "screening")
    stale = ready(stale_papers, "screening")
    reserved_reanalysis_slots = min(
        settings.reanalysis_slots, len(stale), settings.max_candidates,
    )
    candidates = schedule(fresh, settings.max_candidates - reserved_reanalysis_slots, "screening")
    candidates.extend(schedule(stale, settings.max_candidates - len(candidates), "screening"))
    requiring_analysis = len(state.pending_papers) + len(stale_papers)
    LOGGER.info(
        "Fetched %d unique papers; %d require analysis; screening %d",
        len(papers),
        requiring_analysis,
        len(candidates),
    )

    screened = 0
    for paper in candidates:
        if not api_calls_allowed():
            break
        screening = attempt("screening", paper, lambda: analyzer.screen(paper))
        if screening is None:
            continue
        record = Record(
            paper=paper,
            screening=screening,
            review_status="not_selected",
            processed_at=now,
            screening_model=settings.screening_model,
            screening_fingerprint=screening_fingerprint,
        )
        if screening.needs_deep_review and is_publishable(record, review_floor):
            record.review_status = "pending"
        state.records[paper.version_key] = record
        state.pending_papers.pop(paper.version_key, None)
        state.work.pop(f"review:{paper.version_key}", None)
        screened += 1
        checkpoint()
        api_calls_allowed()

    # Preserve screening progress before the more expensive review stage.
    save_state(settings.state_file, state)
    changes_since_checkpoint = 0

    pending = ready([
        record.paper for record in state.records.values()
        if record.review_status == "pending"
        and record.screening_fingerprint == screening_fingerprint
    ], "review")

    reviewed = 0
    stage = "review"
    for paper in schedule(pending, settings.max_deep_reviews, "review"):
        if not api_calls_allowed():
            break
        record = state.records[paper.version_key]
        review = attempt("review", paper, lambda: analyzer.review(paper, record.screening))
        if review is None:
            continue
        record.review = review
        record.review_status = "complete"
        record.review_model = settings.review_model
        record.review_fingerprint = review_fingerprint
        reviewed += 1
        checkpoint()
        api_calls_allowed()

    health = state.health
    finished_at = datetime.now(timezone.utc)
    completed = screened + reviewed
    failed_total = sum(failed_papers.values())
    if source_ok is not None:
        health.source_failure_runs = 0 if source_ok else health.source_failure_runs + 1
        if source_ok:
            health.last_source_success_at = finished_at
    if completed:
        health.last_analysis_success_at = finished_at
    if failed_total or usage.unmetered_calls:
        health.analysis_failure_runs += 1
    elif completed:
        health.analysis_failure_runs = 0
    queued = len(state.pending_papers)
    health.queue_growth_runs = (
        health.queue_growth_runs + 1 if queued > health.previous_queue_size else 0
    )
    health.previous_queue_size = queued
    oldest_wait_days = max(
        ((finished_at - work.first_seen_at).total_seconds() / 86400
         for work in state.work.values()),
        default=0.0,
    )
    quarantined = sum(work.quarantined for work in state.work.values())
    alerts = []
    if failed_total:
        alerts.append(
            f"Analysis incomplete: {failed_papers['screening']} screening failures, "
            f"{failed_papers['review']} review failures."
        )
    if stop_reason in TERMINAL_API_FAILURES or stop_reason == "unmetered_usage":
        alerts.append(
            f"Model service stopped: {stop_reason}; check endpoint, model, output limits and billing usage."
        )
    if source_ok is not None and state.pending_backfill_since is not None:
        alerts.append("Backfill incomplete; inspect source errors and BACKFILL_MAX_PAGES.")
    for label, count in (
        ("Source failures", health.source_failure_runs),
        ("Analysis failures", health.analysis_failure_runs),
        ("Growing candidate queue", health.queue_growth_runs),
    ):
        if count >= settings.alert_after_runs:
            alerts.append(f"{label}: {count} consecutive runs.")
    if oldest_wait_days >= settings.alert_backlog_days:
        alerts.append(f"Oldest waiting task: {oldest_wait_days:.1f} days.")
    if quarantined:
        alerts.append(f"Quarantined tasks: {quarantined}; inspect or reset their work metadata.")
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
    return {
        "fetched": len(papers),
        "bootstrap_fetched": bootstrap_fetched,
        "backfill_fetched": backfill_fetched,
        "fetch_errors": fetch_errors,
        "queued_papers": len(state.pending_papers),
        "pending_reviews": sum(
            record.review_status == "pending" for record in state.records.values()
        ),
        "unseen": requiring_analysis,
        "screened": screened,
        "reviewed": reviewed,
        "published": published,
        "failed_screening": failed_papers["screening"],
        "failed_reviews": failed_papers["review"],
        "api_calls_stopped": int(bool(stop_reason)),
        "stop_reason": stop_reason,
        "status": "degraded" if alerts else "ok",
        "alerts": alerts,
        "quarantined_tasks": quarantined,
        "oldest_wait_days": round(oldest_wait_days, 2),
        "source_failure_runs": health.source_failure_runs,
        "analysis_failure_runs": health.analysis_failure_runs,
        "queue_growth_runs": health.queue_growth_runs,
        "screening_budget_exhausted": int(screening_budget_exhausted),
        "screening_api_calls": usage.screening.calls,
        "review_api_calls": usage.review.calls,
        "screening_finish_reason": usage.screening.last_finish_reason,
        "review_finish_reason": usage.review.last_finish_reason,
        "reasoning_tokens": usage.screening.reasoning_tokens + usage.review.reasoning_tokens,
        "prompt_tokens": usage.prompt_tokens,
        "prompt_cache_hit_tokens": usage.screening.cache_hit_tokens + usage.review.cache_hit_tokens,
        "prompt_cache_miss_tokens": (
            usage.screening.cache_miss_tokens + usage.review.cache_miss_tokens
        ),
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.prompt_tokens + usage.completion_tokens,
        "estimated_cost_rmb": round(usage.cost(settings), 6),
    }
