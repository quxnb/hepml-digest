from pathlib import Path
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from xml.etree import ElementTree

import pytest

from hepml_digest.cli import _demo_papers
from hepml_digest.config import Settings
from hepml_digest.llm import DeepSeekAnalyzer, DemoAnalyzer
from hepml_digest.models import Review, Screening
from hepml_digest.pipeline import run_pipeline
from hepml_digest.store import load_state, save_state


def test_demo_pipeline_is_idempotent(tmp_path: Path):
    settings = Settings(
        state_file=tmp_path / "state.json",
        output_dir=tmp_path / "public",
        site_url="https://example.test/hepml",
        max_candidates=10,
        max_deep_reviews=2,
        feedback_repository="example/hepml",
    )
    first = run_pipeline(settings, DemoAnalyzer(), _demo_papers())
    second = run_pipeline(settings, DemoAnalyzer(), _demo_papers())

    assert first["screened"] == 2
    assert first["reviewed"] == 2
    assert second["screened"] == 0
    assert second["reviewed"] == 0
    assert (settings.output_dir / "atom.xml").exists()
    assert (settings.output_dir / "rss.xml").exists()
    assert (settings.output_dir / "index.html").exists()
    assert (settings.output_dir / "methods.xml").exists()
    assert (settings.output_dir / "hep-applications.xml").exists()
    assert len(load_state(settings.state_file).records) == 2

    rss_root = ElementTree.parse(settings.output_dir / "rss.xml").getroot()
    assert rss_root.findtext("./channel/link") == settings.site_url
    atom_link = rss_root.find(
        "./channel/{http://www.w3.org/2005/Atom}link"
    )
    assert atom_link is not None
    assert atom_link.attrib["href"] == f"{settings.site_url}/rss.xml"

    methods_root = ElementTree.parse(
        settings.output_dir / "methods.xml"
    ).getroot()
    applications_root = ElementTree.parse(
        settings.output_dir / "hep-applications.xml"
    ).getroot()
    assert len(methods_root.findall("./channel/item")) == 1
    assert len(applications_root.findall("./channel/item")) == 1
    assert "github.com/example/hepml/issues/new" in (
        settings.output_dir / "index.html"
    ).read_text(encoding="utf-8")
    assert "方法雷达" in (
        settings.output_dir / "index.html"
    ).read_text(encoding="utf-8")
    assert "HEP 直接应用" in (
        settings.output_dir / "index.html"
    ).read_text(encoding="utf-8")


class ConservativeAnalyzer(DemoAnalyzer):
    def screen(self, paper):
        return Screening(
            relevance=0.35,
            method="待深评方法",
            hep_tasks=["uncertainty_quantification"],
            reason="初筛证据不足，但存在可检验的迁移路径",
            needs_deep_review=False,
            evidence_level="speculative",
        )


class QualifiedConservativeAnalyzer(ConservativeAnalyzer):
    def screen(self, paper):
        screening = super().screen(paper)
        screening.relevance = 0.8
        return screening


@pytest.mark.parametrize("analyzer_type", [ConservativeAnalyzer, QualifiedConservativeAnalyzer])
def test_pipeline_does_not_review_candidates_declined_by_model(tmp_path: Path, analyzer_type):
    papers = []
    for index, source in enumerate(_demo_papers() * 2, start=1):
        paper = source.model_copy(deep=True)
        paper.arxiv_id = f"2607.1{index:04d}"
        paper.link = f"https://arxiv.org/abs/{paper.arxiv_id}"
        paper.categories = ["cs.LG"]
        papers.append(paper)

    settings = Settings(
        state_file=tmp_path / "state.json",
        output_dir=tmp_path / "public",
        max_candidates=10,
        max_deep_reviews=5,
        review_threshold=0.72,
    )
    result = run_pipeline(settings, analyzer_type(), papers)

    assert result["reviewed"] == 0
    assert result["pending_reviews"] == 0
    records = load_state(settings.state_file).records.values()
    assert all(record.review_status == "not_selected" for record in records)


class InvalidXmlAnalyzer(DemoAnalyzer):
    def screen(self, paper):
        screening = super().screen(paper)
        screening.method = "条件密度\x00估计"
        screening.hep_tasks = ["calibration\x0btask"]
        screening.reason = "可迁移\x08，但需要验证"
        return screening

    def review(self, paper, screening):
        return Review(
            summary_cn="总结\x0c含非法字符",
            paper_claims=["结论\x1f一"],
            hep_opportunities=["机会\ufffe一"],
            transfer_risks=["风险\uffff一"],
            validation_plan="验证\x00方案",
            evidence_level="direct",
            confidence=0.8,
        )


def test_pipeline_removes_invalid_xml_characters_from_feeds(tmp_path: Path):
    paper = _demo_papers()[0]
    paper.title = "论文\x00标题"
    paper.authors = ["作者\x01甲"]
    paper.categories.append("hep-ex\x08invalid")
    settings = Settings(
        state_file=tmp_path / "state.json",
        output_dir=tmp_path / "public",
        site_url="https://example.test/hepml",
        max_deep_reviews=1,
    )

    result = run_pipeline(settings, InvalidXmlAnalyzer(), [paper])

    assert result["published"] == 1
    for name in (
        "atom.xml",
        "rss.xml",
        "methods.xml",
        "hep-applications.xml",
    ):
        ElementTree.parse(settings.output_dir / name)
    index = (settings.output_dir / "index.html").read_text(encoding="utf-8")
    invalid_characters = (
        "\x00",
        "\x01",
        "\x08",
        "\x0b",
        "\x0c",
        "\x1f",
        "\ufffe",
        "\uffff",
    )
    for character in invalid_characters:
        assert character not in index


class VersionAnalyzer(DemoAnalyzer):
    def screen(self, paper):
        return Screening(
            relevance=0.9 if paper.version == 1 else 0.1,
            method="版本测试",
            hep_tasks=[],
            reason="测试最新版本替换",
            needs_deep_review=False,
            evidence_level="transferable" if paper.version == 1 else "irrelevant",
        )


def test_latest_version_is_filtered_before_publishing(tmp_path: Path):
    version_one = _demo_papers()[0]
    version_two = version_one.model_copy(deep=True)
    version_two.version = 2
    settings = Settings(
        state_file=tmp_path / "state.json",
        output_dir=tmp_path / "public",
        max_deep_reviews=0,
    )

    result = run_pipeline(
        settings, VersionAnalyzer(), [version_one, version_two]
    )

    assert result["published"] == 0
    rss_root = ElementTree.parse(settings.output_dir / "rss.xml").getroot()
    assert rss_root.findall("./channel/item") == []


class IrrelevantAnalyzer(DemoAnalyzer):
    def screen(self, paper):
        return Screening(
            relevance=0.95,
            method="无关方法",
            hep_tasks=[],
            reason="与项目无关",
            needs_deep_review=True,
            evidence_level="irrelevant",
        )

    def review(self, paper, screening):
        raise AssertionError("irrelevant records must not be reviewed")


def test_irrelevant_record_does_not_enter_review_queue(tmp_path: Path):
    settings = Settings(
        state_file=tmp_path / "state.json",
        output_dir=tmp_path / "public",
        max_deep_reviews=1,
    )

    result = run_pipeline(settings, IrrelevantAnalyzer(), [_demo_papers()[0]])

    assert result["reviewed"] == 0
    assert result["failed_reviews"] == 0
    assert result["published"] == 0
    for name in ("rss.xml", "methods.xml", "hep-applications.xml"):
        root = ElementTree.parse(settings.output_dir / name).getroot()
        assert root.findall("./channel/item") == []
    atom = ElementTree.parse(settings.output_dir / "atom.xml").getroot()
    assert atom.findall("{http://www.w3.org/2005/Atom}entry") == []
    assert _demo_papers()[0].title not in (
        settings.output_dir / "index.html"
    ).read_text(encoding="utf-8")
    record = next(iter(load_state(settings.state_file).records.values()))
    assert record.review_status == "not_selected"


class FailingAnalyzer(DemoAnalyzer):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def screen(self, paper):
        self.calls += 1
        raise RuntimeError("API unavailable")


def test_pipeline_stops_after_consecutive_api_failures(tmp_path: Path):
    papers = []
    for index, source in enumerate(_demo_papers() * 3, start=1):
        paper = source.model_copy(deep=True)
        paper.arxiv_id = f"2607.2{index:04d}"
        papers.append(paper)
    analyzer = FailingAnalyzer()
    settings = Settings(
        state_file=tmp_path / "state.json",
        output_dir=tmp_path / "public",
        max_candidates=6,
        max_deep_reviews=0,
        max_consecutive_api_failures=2,
    )

    result = run_pipeline(settings, analyzer, papers)

    assert analyzer.calls == 2
    assert result["failed_screening"] == 2
    assert result["api_calls_stopped"] == 1


def test_zero_api_time_budget_skips_calls_and_still_publishes(tmp_path: Path):
    analyzer = FailingAnalyzer()
    settings = Settings(
        state_file=tmp_path / "state.json",
        output_dir=tmp_path / "public",
        max_deep_reviews=0,
        api_time_budget_seconds=0,
    )

    result = run_pipeline(settings, analyzer, [_demo_papers()[0]])

    assert analyzer.calls == 0
    assert result["api_calls_stopped"] == 1
    assert (settings.output_dir / "atom.xml").exists()


class ChangedScreenPromptAnalyzer(DemoAnalyzer):
    screening_fingerprint = "changed-screen-prompt"


def test_screening_fingerprint_change_reprocesses_record(tmp_path: Path):
    paper = _demo_papers()[0]
    settings = Settings(
        state_file=tmp_path / "state.json",
        output_dir=tmp_path / "public",
        max_deep_reviews=1,
    )
    run_pipeline(settings, DemoAnalyzer(), [paper])

    result = run_pipeline(settings, ChangedScreenPromptAnalyzer(), [paper])

    assert result["screened"] == 1
    record = load_state(settings.state_file).records[paper.version_key]
    assert record.screening_fingerprint == "changed-screen-prompt"


class ChangedReviewPromptAnalyzer(DemoAnalyzer):
    review_fingerprint = "changed-review-prompt"


def test_review_fingerprint_change_reprocesses_completed_review(
    tmp_path: Path,
):
    paper = _demo_papers()[0]
    settings = Settings(
        state_file=tmp_path / "state.json",
        output_dir=tmp_path / "public",
        max_deep_reviews=1,
    )
    run_pipeline(settings, DemoAnalyzer(), [paper])

    result = run_pipeline(settings, ChangedReviewPromptAnalyzer(), [paper])

    assert result["screened"] == 0
    assert result["reviewed"] == 1
    record = load_state(settings.state_file).records[paper.version_key]
    assert record.review_fingerprint == "changed-review-prompt"


class TokenBudgetAnalyzer(DemoAnalyzer):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def screen(self, paper):
        self.calls += 1
        self.usage.screening.calls += 1
        self.usage.screening.prompt_tokens += 10
        self.usage.screening.cache_miss_tokens += 10
        return super().screen(paper)


def test_prompt_token_budget_stops_additional_calls(tmp_path: Path):
    analyzer = TokenBudgetAnalyzer()
    settings = Settings(
        state_file=tmp_path / "state.json",
        output_dir=tmp_path / "public",
        max_deep_reviews=0,
        max_prompt_tokens=10,
    )

    result = run_pipeline(settings, analyzer, _demo_papers())

    assert analyzer.calls == 1
    assert result["prompt_tokens"] == 10
    assert result["estimated_cost_rmb"] == 0.00001
    assert result["api_calls_stopped"] == 1


def test_legacy_records_without_provenance_are_reanalyzed(
    tmp_path: Path,
):
    paper = _demo_papers()[0]
    settings = Settings(
        state_file=tmp_path / "state.json",
        output_dir=tmp_path / "public",
        max_deep_reviews=1,
    )
    run_pipeline(settings, DemoAnalyzer(), [paper])
    state = load_state(settings.state_file)
    state.schema_version = 1
    record = state.records[paper.version_key]
    record.screening_fingerprint = ""
    record.review_fingerprint = None
    save_state(settings.state_file, state)

    result = run_pipeline(settings, DemoAnalyzer(), [paper])

    assert result["screened"] == 1
    assert result["reviewed"] == 1
    migrated = load_state(settings.state_file)
    assert migrated.schema_version == 4
    assert migrated.records[paper.version_key].screening_fingerprint
    assert migrated.records[paper.version_key].review_fingerprint


def test_historical_reanalysis_keeps_reserved_candidate_slot(tmp_path: Path):
    historical = _demo_papers()[0]
    initial_settings = Settings(
        state_file=tmp_path / "state.json",
        output_dir=tmp_path / "public",
        max_candidates=2,
        max_deep_reviews=0,
    )
    run_pipeline(initial_settings, DemoAnalyzer(), [historical])
    fresh_papers = []
    for index, source in enumerate(_demo_papers(), start=1):
        paper = source.model_copy(deep=True)
        paper.arxiv_id = f"2607.3{index:04d}"
        fresh_papers.append(paper)
    changed_settings = Settings(
        state_file=tmp_path / "state.json",
        output_dir=tmp_path / "public",
        max_candidates=2,
        max_deep_reviews=0,
        reanalysis_slots=1,
    )

    result = run_pipeline(
        changed_settings, ChangedScreenPromptAnalyzer(), fresh_papers
    )

    assert result["screened"] == 2
    state = load_state(changed_settings.state_file)
    assert (
        state.records[historical.version_key].screening_fingerprint
        == "changed-screen-prompt"
    )


def test_stale_feed_record_uses_reserved_reanalysis_slot(tmp_path: Path):
    stale_paper = _demo_papers()[0]
    settings = Settings(
        state_file=tmp_path / "state.json",
        output_dir=tmp_path / "public",
        max_candidates=2,
        max_deep_reviews=0,
        reanalysis_slots=1,
    )
    run_pipeline(settings, DemoAnalyzer(), [stale_paper])
    fresh_papers = []
    for index, source in enumerate(_demo_papers(), start=1):
        paper = source.model_copy(deep=True)
        paper.arxiv_id = f"2607.4{index:04d}"
        fresh_papers.append(paper)

    result = run_pipeline(
        settings,
        ChangedScreenPromptAnalyzer(),
        [stale_paper, *fresh_papers],
    )

    assert result["screened"] == 2
    state = load_state(settings.state_file)
    assert (
        state.records[stale_paper.version_key].screening_fingerprint
        == "changed-screen-prompt"
    )


class InterruptingAnalyzer(DemoAnalyzer):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def screen(self, paper):
        self.calls += 1
        if self.calls == 2:
            raise KeyboardInterrupt
        return super().screen(paper)


class InterruptingReviewAnalyzer(DemoAnalyzer):
    def review(self, paper, screening):
        raise KeyboardInterrupt


def test_checkpoint_preserves_progress_after_interruption(tmp_path: Path):
    settings = Settings(
        state_file=tmp_path / "state.json",
        output_dir=tmp_path / "public",
        max_candidates=2,
        max_deep_reviews=0,
        checkpoint_interval=1,
    )

    with pytest.raises(KeyboardInterrupt):
        run_pipeline(settings, InterruptingAnalyzer(), _demo_papers())

    assert len(load_state(settings.state_file).records) == 1


def test_pending_review_survives_interruption_and_resumes(tmp_path: Path):
    paper = _demo_papers()[0]
    settings = Settings(
        state_file=tmp_path / "state.json",
        output_dir=tmp_path / "public",
        max_candidates=1,
        max_deep_reviews=1,
        checkpoint_interval=10,
    )

    with pytest.raises(KeyboardInterrupt):
        run_pipeline(settings, InterruptingReviewAnalyzer(), [paper])

    interrupted = load_state(settings.state_file)
    assert interrupted.records[paper.version_key].review_status == "pending"

    result = run_pipeline(settings, DemoAnalyzer(), [paper])

    assert result["screened"] == 0
    assert result["reviewed"] == 1
    resumed = load_state(settings.state_file)
    assert resumed.records[paper.version_key].review_status == "complete"


def test_state_is_saved_before_publish_failure(
    tmp_path: Path, monkeypatch
):
    settings = Settings(
        state_file=tmp_path / "state.json",
        output_dir=tmp_path / "public",
        max_candidates=2,
        max_deep_reviews=0,
        checkpoint_interval=100,
    )

    def fail_publish(*args, **kwargs):
        raise RuntimeError("publishing failed")

    monkeypatch.setattr("hepml_digest.pipeline.publish", fail_publish)

    with pytest.raises(RuntimeError, match="publishing failed"):
        run_pipeline(settings, DemoAnalyzer(), _demo_papers())

    assert len(load_state(settings.state_file).records) == 2


class SerializationFailureFeed:
    rss_calls = 0

    def atom_str(self, pretty=True):
        return b"<feed/>"

    def rss_str(self, pretty=True):
        type(self).rss_calls += 1
        if type(self).rss_calls == 2:
            raise ValueError("feed serialization failed")
        return b"<rss><channel><link>old</link></channel></rss>"


def test_serialization_failure_does_not_replace_existing_site(
    tmp_path: Path, monkeypatch
):
    settings = Settings(
        state_file=tmp_path / "state.json",
        output_dir=tmp_path / "public",
        max_deep_reviews=0,
    )
    settings.output_dir.mkdir()
    output_names = (
        "atom.xml",
        "rss.xml",
        "methods.xml",
        "hep-applications.xml",
        "index.html",
    )
    for name in output_names:
        (settings.output_dir / name).write_bytes(b"previous site")
    SerializationFailureFeed.rss_calls = 0
    monkeypatch.setattr(
        "hepml_digest.publish._build_feed",
        lambda *args, **kwargs: SerializationFailureFeed(),
    )

    with pytest.raises(ValueError, match="feed serialization failed"):
        run_pipeline(settings, DemoAnalyzer(), _demo_papers())

    for name in output_names:
        assert (settings.output_dir / name).read_bytes() == b"previous site"


def test_candidate_overflow_is_drained_after_leaving_rss(tmp_path: Path):
    settings = Settings(
        state_file=tmp_path / "state.json", output_dir=tmp_path / "public",
        max_candidates=1, max_deep_reviews=0,
    )
    first = run_pipeline(settings, DemoAnalyzer(), _demo_papers())
    assert first["queued_papers"] == 1
    second = run_pipeline(settings, DemoAnalyzer(), [])
    assert second["screened"] == 1
    assert second["queued_papers"] == 0
    assert len(load_state(settings.state_file).records) == 2


def test_failed_screening_is_retried_from_saved_queue(tmp_path: Path):
    settings = Settings(
        state_file=tmp_path / "state.json", output_dir=tmp_path / "public",
        max_deep_reviews=0,
    )
    first = run_pipeline(settings, FailingAnalyzer(), _demo_papers())
    assert first["queued_papers"] == 2
    assert run_pipeline(settings, DemoAnalyzer(), [])["screened"] == 0
    state = load_state(settings.state_file)
    for work in state.work.values():
        work.next_retry_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    save_state(settings.state_file, state)
    second = run_pipeline(settings, DemoAnalyzer(), [])
    assert second["screened"] == 2
    assert second["queued_papers"] == 0


def test_raw_candidates_survive_interruption_before_checkpoint(tmp_path: Path):
    settings = Settings(
        state_file=tmp_path / "state.json", output_dir=tmp_path / "public",
        max_deep_reviews=0, checkpoint_interval=100,
    )
    with pytest.raises(KeyboardInterrupt):
        run_pipeline(settings, InterruptingAnalyzer(), _demo_papers())
    assert len(load_state(settings.state_file).pending_papers) == 2
    result = run_pipeline(settings, DemoAnalyzer(), [])
    assert result["screened"] == 2


def test_partial_backfill_persists_pages_without_advancing_cursor(tmp_path: Path, monkeypatch):
    settings = Settings(
        state_file=tmp_path / "state.json", output_dir=tmp_path / "public",
        max_candidates=0, max_deep_reviews=0,
    )
    run_pipeline(settings, DemoAnalyzer(), [])
    state = load_state(settings.state_file)
    cursor = datetime.now(timezone.utc) - timedelta(days=2)
    state.last_backfill_at = cursor
    save_state(settings.state_file, state)
    monkeypatch.setattr("hepml_digest.pipeline.fetch_feeds", lambda *a, **k: [])

    def updates(categories, since, until, *args, **kwargs):
        assert since == cursor - timedelta(days=settings.backfill_lookback_days)
        yield _demo_papers()
        raise OSError("second page failed")

    monkeypatch.setattr("hepml_digest.pipeline.fetch_updates", updates)
    result = run_pipeline(settings, DemoAnalyzer())
    saved = load_state(settings.state_file)
    assert result["fetch_errors"] == 1
    assert len(saved.pending_papers) == 2
    assert saved.last_backfill_at == cursor


def test_backfill_rescues_all_rss_source_failure(tmp_path: Path, monkeypatch):
    settings = Settings(
        state_file=tmp_path / "state.json", output_dir=tmp_path / "public",
        bootstrap_results=0, max_deep_reviews=0,
    )

    def failed_feeds(urls, *args, failed_urls, **kwargs):
        failed_urls.extend(urls)
        raise RuntimeError("RSS unavailable")

    monkeypatch.setattr("hepml_digest.pipeline.fetch_feeds", failed_feeds)
    monkeypatch.setattr(
        "hepml_digest.pipeline.fetch_updates", lambda *a, **k: iter([_demo_papers()]),
    )
    result = run_pipeline(settings, DemoAnalyzer())
    assert result["screened"] == 2
    assert result["fetch_errors"] == len(settings.feed_urls)
    assert load_state(settings.state_file).last_backfill_at is not None


def test_saved_queue_is_used_when_all_sources_are_unavailable(tmp_path: Path, monkeypatch):
    settings = Settings(
        state_file=tmp_path / "state.json", output_dir=tmp_path / "public",
        max_candidates=0, max_deep_reviews=0,
    )
    run_pipeline(settings, DemoAnalyzer(), _demo_papers())

    def unavailable(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr("hepml_digest.pipeline.fetch_feeds", unavailable)
    monkeypatch.setattr("hepml_digest.pipeline.fetch_updates", unavailable)
    settings.max_candidates = 2
    result = run_pipeline(settings, DemoAnalyzer())
    assert result["screened"] == 2


def test_screening_leaves_prompt_budget_for_reviews(tmp_path: Path):
    settings = Settings(
        state_file=tmp_path / "state.json", output_dir=tmp_path / "public",
        max_prompt_tokens=20, screening_budget_fraction=0.5,
        max_deep_reviews=1,
    )
    result = run_pipeline(settings, TokenBudgetAnalyzer(), _demo_papers())
    assert result["screened"] == 1
    assert result["reviewed"] == 1
    assert result["screening_budget_exhausted"] == 1
    assert result["api_calls_stopped"] == 0
    assert result["queued_papers"] == 1


def test_screening_leaves_time_budget_for_reviews(tmp_path: Path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("hepml_digest.pipeline.time.monotonic", lambda: clock[0])

    class TimedAnalyzer(DemoAnalyzer):
        def screen(self, paper):
            clock[0] += 60
            return super().screen(paper)

    settings = Settings(
        state_file=tmp_path / "state.json", output_dir=tmp_path / "public",
        api_time_budget_seconds=100, screening_budget_fraction=0.5,
        max_deep_reviews=1,
    )
    result = run_pipeline(settings, TimedAnalyzer(), _demo_papers())
    assert result["screened"] == 1
    assert result["reviewed"] == 1
    assert result["screening_budget_exhausted"] == 1


@pytest.mark.parametrize("deep_reviews", [0, 1])
def test_analysis_basis_is_explicit_in_site_and_feeds(tmp_path: Path, deep_reviews):
    settings = Settings(
        state_file=tmp_path / "state.json", output_dir=tmp_path / "public",
        max_deep_reviews=deep_reviews,
    )
    run_pipeline(settings, DemoAnalyzer(), [_demo_papers()[0]])
    for name in ("index.html", "rss.xml", "atom.xml", "hep-applications.xml"):
        content = (settings.output_dir / name).read_text(encoding="utf-8")
        assert "仅标题和摘要；未读取全文" in content
        assert "论文明确支持的结论" not in content


def test_first_incomplete_backfill_keeps_its_window_until_success(tmp_path: Path, monkeypatch):
    settings = Settings(
        state_file=tmp_path / "state.json", output_dir=tmp_path / "public",
        bootstrap_results=0, max_candidates=0,
        max_deep_reviews=0,
    )
    monkeypatch.setattr("hepml_digest.pipeline.fetch_feeds", lambda *a, **k: [])

    def interrupted(*args, **kwargs):
        yield _demo_papers()
        raise OSError("partial scan")

    monkeypatch.setattr("hepml_digest.pipeline.fetch_updates", interrupted)
    run_pipeline(settings, DemoAnalyzer())
    saved = load_state(settings.state_file)
    assert saved.last_backfill_at is None
    assert saved.pending_backfill_since is not None

    def completed(categories, since, *args, **kwargs):
        assert since == saved.pending_backfill_since
        yield []

    monkeypatch.setattr("hepml_digest.pipeline.fetch_updates", completed)
    result = run_pipeline(settings, DemoAnalyzer())
    assert result["fetch_errors"] == 0
    completed_state = load_state(settings.state_file)
    assert completed_state.pending_backfill_since is None
    assert completed_state.last_backfill_at is not None


def test_internal_retry_cannot_consume_reserved_review_budget(tmp_path: Path, monkeypatch):
    settings = Settings(
        state_file=tmp_path / "state.json", output_dir=tmp_path / "public",
        max_deep_reviews=0, max_prompt_tokens=20,
        screening_budget_fraction=0.5,
    )
    papers = _demo_papers()
    run_pipeline(settings, DemoAnalyzer(), [papers[0]])
    settings.max_deep_reviews = 1
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    analyzer = DeepSeekAnalyzer(
        settings.screening_model, settings.review_model, "screen", "review",
    )
    analyzer.screening_fingerprint = DemoAnalyzer.screening_fingerprint
    analyzer.review_fingerprint = DemoAnalyzer.review_fingerprint
    monkeypatch.setattr(DeepSeekAnalyzer._request_json.retry, "wait", lambda _: 0)

    def create(**kwargs):
        screening = kwargs["model"] == settings.screening_model
        content = "invalid JSON" if screening else DemoAnalyzer().review(
            papers[0], DemoAnalyzer().screen(papers[0]),
        ).model_dump_json()
        return SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=10 if screening else 1, completion_tokens=0),
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        )

    analyzer.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    )
    result = run_pipeline(settings, analyzer, [papers[1]])
    assert result["screening_api_calls"] == 1
    assert result["review_api_calls"] == 1
    assert result["screening_budget_exhausted"] == 1
    assert result["reviewed"] == 1
    assert result["queued_papers"] == 1
