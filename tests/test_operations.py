from datetime import datetime, timedelta, timezone

import httpx
import pytest
from openai import AuthenticationError, BadRequestError, NotFoundError, PermissionDeniedError

from hepml_digest.cli import _demo_papers, main
from hepml_digest.config import Settings
from hepml_digest.llm import APICallBudgetExceeded, DemoAnalyzer, OutputLimitExceeded
from hepml_digest.pipeline import run_pipeline
from hepml_digest.store import load_state, save_state


def settings_for(tmp_path, **kwargs):
    return Settings(
        state_file=tmp_path / "state.json", output_dir=tmp_path / "public",
        **kwargs,
    )


def allow_retry(settings):
    state = load_state(settings.state_file)
    for work in state.work.values():
        work.next_retry_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    save_state(settings.state_file, state)


def test_aging_slot_prevents_a_low_keyword_candidate_from_starving(tmp_path):
    settings = settings_for(tmp_path, max_candidates=0, max_deep_reviews=0, aging_slots=1)
    old, fresh = _demo_papers()
    old.categories = fresh.categories = ["cs.LG"]
    old.abstract = "An unranked method."
    fresh.abstract = "simulation-based inference normalizing flow calibration"
    run_pipeline(settings, DemoAnalyzer(), [old])
    state = load_state(settings.state_file)
    state.work[f"screening:{old.version_key}"].first_seen_at -= timedelta(days=2)
    save_state(settings.state_file, state)
    settings.max_candidates = 1
    run_pipeline(settings, DemoAnalyzer(), [fresh])
    state = load_state(settings.state_file)
    assert old.version_key in state.records
    assert fresh.version_key in state.pending_papers


def test_review_aging_slot_prevents_old_reviews_from_starving(tmp_path):
    settings = settings_for(tmp_path, max_deep_reviews=0, aging_slots=1)
    old, fresh = _demo_papers()
    run_pipeline(settings, DemoAnalyzer(), [old, fresh])
    state = load_state(settings.state_file)
    state.records[old.version_key].screening.relevance = 0.6
    state.records[fresh.version_key].screening.relevance = 0.99
    state.work[f"review:{old.version_key}"].first_seen_at -= timedelta(days=2)
    save_state(settings.state_file, state)
    settings.max_deep_reviews = 1
    run_pipeline(settings, DemoAnalyzer(), [])
    state = load_state(settings.state_file)
    assert state.records[old.version_key].review_status == "complete"
    assert state.records[fresh.version_key].review_status == "pending"


@pytest.mark.parametrize("stage", ["screening", "review"])
def test_bad_responses_are_cooled_down_quarantined_and_unlocked_by_revision(tmp_path, stage):
    class InvalidAnalyzer(DemoAnalyzer):
        def screen(self, paper):
            if stage == "screening":
                raise ValueError("invalid JSON")
            return super().screen(paper)

        def review(self, paper, screening):
            raise ValueError("invalid JSON")

    settings = settings_for(tmp_path, max_deep_reviews=1, max_paper_failures=3)
    paper = _demo_papers()[0]
    for index in range(3):
        result = run_pipeline(settings, InvalidAnalyzer(), [paper] if index == 0 else [])
        work = load_state(settings.state_file).work[f"{stage}:{paper.version_key}"]
        assert work.failures == index + 1
        assert work.quarantined == (index == 2)
        if index == 0:
            skipped = run_pipeline(settings, InvalidAnalyzer(), [])
            assert skipped["failed_screening"] == skipped["failed_reviews"] == 0
        allow_retry(settings)
    assert result["quarantined_tasks"] == 1
    assert run_pipeline(settings, DemoAnalyzer(), [])["reviewed"] == 0
    fixed = DemoAnalyzer()
    setattr(fixed, f"{stage}_fingerprint", "new-revision")
    result = run_pipeline(settings, fixed, [])
    assert result["reviewed"] == 1
    assert result["quarantined_tasks"] == 0


def test_transient_outages_do_not_quarantine_a_paper(tmp_path):
    class OfflineAnalyzer(DemoAnalyzer):
        def screen(self, paper):
            raise RuntimeError("temporary outage")

    settings = settings_for(tmp_path, max_deep_reviews=0, max_paper_failures=1)
    for index in range(3):
        result = run_pipeline(settings, OfflineAnalyzer(), _demo_papers()[:1] if index == 0 else [])
        assert result["quarantined_tasks"] == 0
        allow_retry(settings)


@pytest.mark.parametrize(("error_type", "status", "kind"), [
    (AuthenticationError, 401, "authentication"),
    (PermissionDeniedError, 403, "authentication"),
    (BadRequestError, 400, "configuration"),
    (NotFoundError, 404, "configuration"),
])
def test_service_errors_stop_immediately_without_blame_or_secret_logging(tmp_path, caplog, error_type, status, kind):
    class BrokenService(DemoAnalyzer):
        calls = 0

        def screen(self, paper):
            self.calls += 1
            raise error_type(
                "sensitive-provider-body", body=None,
                response=httpx.Response(status, request=httpx.Request("POST", "https://relay.example/v1")),
            )

    settings = settings_for(tmp_path, max_deep_reviews=0)
    analyzer = BrokenService()
    result = run_pipeline(settings, analyzer, _demo_papers())
    assert analyzer.calls == 1
    assert result["stop_reason"] == kind
    assert result["status"] == "degraded"
    assert result["queued_papers"] == 2
    assert all(work.failures == 0 for work in load_state(settings.state_file).work.values())
    assert "sensitive-provider-body" not in caplog.text
    assert "sensitive-provider-body" not in settings.state_file.read_text()


def test_output_tokens_alone_can_stop_the_cost_budget(tmp_path):
    class ExpensiveAnalyzer(DemoAnalyzer):
        def screen(self, paper):
            self.usage.screening.completion_tokens += 250000
            return super().screen(paper)

    result = run_pipeline(
        settings_for(tmp_path, max_deep_reviews=0, max_estimated_cost_rmb=1),
        ExpensiveAnalyzer(), _demo_papers(),
    )
    assert result["screened"] == 1
    assert result["queued_papers"] == 1
    assert result["stop_reason"] == "cost_budget"
    assert result["estimated_cost_rmb"] == 1


def test_zero_cost_budget_blocks_all_model_calls(tmp_path):
    result = run_pipeline(
        settings_for(tmp_path, max_deep_reviews=0, max_estimated_cost_rmb=0),
        DemoAnalyzer(), _demo_papers(),
    )
    assert result["screened"] == 0
    assert result["queued_papers"] == 2
    assert result["stop_reason"] == "cost_budget"


def test_budget_stop_between_loop_and_request_is_not_an_analysis_failure(tmp_path):
    settings = settings_for(tmp_path, max_deep_reviews=0)

    class BudgetRaceAnalyzer(DemoAnalyzer):
        def screen(self, paper):
            settings.max_prompt_tokens = 0
            if not self.request_guard():
                raise APICallBudgetExceeded("budget exhausted before request")

    result = run_pipeline(settings, BudgetRaceAnalyzer(), _demo_papers())
    assert result["stop_reason"] == "token_budget"
    assert result["failed_screening"] == 0
    assert result["analysis_failure_runs"] == 0
    assert result["queued_papers"] == 2


def test_missing_billing_usage_stops_further_calls_and_warns(tmp_path):
    class UnmeteredAnalyzer(DemoAnalyzer):
        def screen(self, paper):
            self.usage.unmetered_calls += 1
            return super().screen(paper)

    result = run_pipeline(settings_for(tmp_path), UnmeteredAnalyzer(), _demo_papers())
    assert result["screened"] == 1
    assert result["reviewed"] == 0
    assert result["stop_reason"] == "unmetered_usage"
    assert result["status"] == "degraded"


def test_source_failure_alerts_persist_and_reset_on_recovery(tmp_path, monkeypatch):
    settings = settings_for(tmp_path, max_deep_reviews=0)
    run_pipeline(settings, DemoAnalyzer(), _demo_papers())

    def offline(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr("hepml_digest.pipeline.fetch_feeds", offline)
    monkeypatch.setattr("hepml_digest.pipeline.fetch_updates", offline)
    for index in range(3):
        result = run_pipeline(settings, DemoAnalyzer())
        assert result["source_failure_runs"] == index + 1
    assert any("Source failures: 3" in alert for alert in result["alerts"])
    monkeypatch.setattr("hepml_digest.pipeline.fetch_feeds", lambda *a, **k: [])
    monkeypatch.setattr("hepml_digest.pipeline.fetch_updates", lambda *a, **k: iter([]))
    recovered = run_pipeline(settings, DemoAnalyzer())
    assert recovered["source_failure_runs"] == 0
    assert recovered["analysis_failure_runs"] == 0
    assert recovered["alerts"] == []
    assert load_state(settings.state_file).health.last_source_success_at is not None


def test_queue_growth_and_oldest_wait_alerts_are_not_silent(tmp_path):
    settings = settings_for(tmp_path, max_candidates=0, max_deep_reviews=0)
    for index in range(3):
        paper = _demo_papers()[0]
        paper.arxiv_id = f"2607.{index:05d}"
        result = run_pipeline(settings, DemoAnalyzer(), [paper])
    assert any("Growing candidate queue: 3" in alert for alert in result["alerts"])
    state = load_state(settings.state_file)
    next(iter(state.work.values())).first_seen_at -= timedelta(days=8)
    save_state(settings.state_file, state)
    result = run_pipeline(settings, DemoAnalyzer(), [])
    assert result["queue_growth_runs"] == 0
    assert any("Oldest waiting task" in alert for alert in result["alerts"])


@pytest.mark.parametrize("kind", ["authentication", "output_limit"])
def test_fatal_cli_error_preserves_state_and_writes_actions_diagnostics(tmp_path, monkeypatch, capsys, kind):
    class BrokenService(DemoAnalyzer):
        def screen(self, paper):
            if kind == "output_limit":
                raise OutputLimitExceeded("private-error-body")
            raise AuthenticationError(
                "private-error-body", body=None,
                response=httpx.Response(401, request=httpx.Request("POST", "https://relay.example")),
            )

    state = tmp_path / "state.json"
    summary = tmp_path / "summary.md"
    monkeypatch.setattr("hepml_digest.cli.DemoAnalyzer", BrokenService)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setattr("sys.argv", [
        "hepml-digest", "--demo", "--state-file", str(state),
        "--output-dir", str(tmp_path / "public"),
    ])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 1
    assert len(load_state(state).pending_papers) == 2
    output = capsys.readouterr().out
    assert "::error::" in output
    assert "private-error-body" not in output
    assert kind in summary.read_text()


@pytest.mark.parametrize(("score", "needs_review", "review_threshold", "publish_threshold", "expected"), [
    (0.18, False, 0.45, 0.55, 0),
    (0.50, True, 0.45, 0.55, 0),
    (0.60, False, 0.45, 0.55, 0),
    (0.70, True, 0.80, 0.55, 0),
    (0.80, False, 0.80, 0.55, 0),
    (0.60, True, 0.45, 0.55, 1),
    (0.80, True, 0.80, 0.55, 1),
    (0.55, True, 0.45, 0.55, 1),
])
def test_reviews_require_recommendation_and_both_thresholds(tmp_path, score, needs_review, review_threshold, publish_threshold, expected):
    class ScoredAnalyzer(DemoAnalyzer):
        def screen(self, paper):
            screening = super().screen(paper)
            screening.relevance = score
            screening.needs_deep_review = needs_review
            return screening

        def review(self, paper, screening):
            assert expected == 1, "Below-threshold papers must not consume review calls"
            return super().review(paper, screening)

    settings = Settings(
        state_file=tmp_path / "state.json", output_dir=tmp_path / "public",
        max_deep_reviews=1,
        review_threshold=review_threshold, publish_threshold=publish_threshold,
    )
    result = run_pipeline(settings, ScoredAnalyzer(), _demo_papers()[:1])
    assert result["reviewed"] == expected
    assert result["failed_reviews"] == 0
    assert result["pending_reviews"] == 0


@pytest.mark.parametrize(("score", "needs_review"), [(0.18, True), (0.95, False)])
def test_old_ineligible_pending_review_is_retired_without_losing_paper(tmp_path, score, needs_review):
    settings = settings_for(tmp_path, max_deep_reviews=0)
    paper = _demo_papers()[0]
    run_pipeline(settings, DemoAnalyzer(), [paper])
    state = load_state(settings.state_file)
    state.records[paper.version_key].screening.relevance = score
    state.records[paper.version_key].screening.needs_deep_review = needs_review
    assert state.records[paper.version_key].review_status == "pending"
    assert f"review:{paper.version_key}" in state.work
    save_state(settings.state_file, state)
    settings.max_deep_reviews = 1
    result = run_pipeline(settings, DemoAnalyzer(), [])
    saved = load_state(settings.state_file)
    assert result["reviewed"] == result["failed_reviews"] == result["pending_reviews"] == 0
    assert saved.records[paper.version_key].review_status == "not_selected"
    assert f"review:{paper.version_key}" not in saved.work


@pytest.mark.parametrize("failed_stage", ["screening", "review"])
def test_partial_analysis_failure_warns_and_counts_despite_other_successes(tmp_path, failed_stage):
    papers = _demo_papers()

    class PartiallyBrokenAnalyzer(DemoAnalyzer):
        def screen(self, paper):
            if failed_stage == "screening" and paper.version_key == papers[0].version_key:
                raise RuntimeError("temporary outage")
            return super().screen(paper)

        def review(self, paper, screening):
            if failed_stage == "review" and paper.version_key == papers[0].version_key:
                raise RuntimeError("temporary outage")
            return super().review(paper, screening)

    settings = settings_for(tmp_path, max_deep_reviews=2)
    result = run_pipeline(settings, PartiallyBrokenAnalyzer(), papers)
    assert result["screened"] + result["reviewed"] > 0
    assert result["status"] == "degraded"
    assert result["analysis_failure_runs"] == 1
    assert any("Analysis incomplete" in alert for alert in result["alerts"])
    allow_retry(settings)
    recovered = run_pipeline(settings, DemoAnalyzer(), [])
    assert recovered["analysis_failure_runs"] == 0
    assert recovered["status"] == "ok"


def test_output_limit_stops_review_stage_without_quarantining_a_paper(tmp_path):
    class TruncatedAnalyzer(DemoAnalyzer):
        calls = 0

        def review(self, paper, screening):
            self.calls += 1
            raise OutputLimitExceeded("finish_reason=length")

    settings = settings_for(tmp_path, max_deep_reviews=2)
    analyzer = TruncatedAnalyzer()
    result = run_pipeline(settings, analyzer, _demo_papers())
    assert analyzer.calls == 1
    assert result["stop_reason"] == "output_limit"
    assert result["status"] == "degraded"
    assert result["analysis_failure_runs"] == 1
    assert result["pending_reviews"] == 2
    assert result["quarantined_tasks"] == 0
    assert all(work.failures == 0 for work in load_state(settings.state_file).work.values())
