from pathlib import Path
from xml.etree import ElementTree

import pytest

from hepml_digest.cli import _demo_papers
from hepml_digest.config import Settings
from hepml_digest.llm import AnalysisUsage, DemoAnalyzer
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
        min_deep_reviews=2,
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


def test_pipeline_promotes_daily_minimum_review_target(tmp_path: Path):
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
        min_deep_reviews=3,
        review_threshold=0.72,
    )
    result = run_pipeline(settings, ConservativeAnalyzer(), papers)

    assert result["promoted_reviews"] == 3
    assert result["reviewed"] == 3
    records = load_state(settings.state_file).records.values()
    assert sum(record.review_status == "complete" for record in records) == 3


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
        min_deep_reviews=1,
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
            evidence_level="irrelevant",
        )


def test_latest_version_is_filtered_before_publishing(tmp_path: Path):
    version_one = _demo_papers()[0]
    version_two = version_one.model_copy(deep=True)
    version_two.version = 2
    settings = Settings(
        state_file=tmp_path / "state.json",
        output_dir=tmp_path / "public",
        max_deep_reviews=0,
        min_deep_reviews=0,
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
        min_deep_reviews=0,
    )

    result = run_pipeline(settings, IrrelevantAnalyzer(), [_demo_papers()[0]])

    assert result["reviewed"] == 0
    assert result["failed_reviews"] == 0
    record = next(iter(load_state(settings.state_file).records.values()))
    assert record.review_status == "not_selected"


class FailingAnalyzer(DemoAnalyzer):
    def __init__(self):
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
        min_deep_reviews=0,
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
        min_deep_reviews=0,
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
        min_deep_reviews=1,
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
        min_deep_reviews=1,
    )
    run_pipeline(settings, DemoAnalyzer(), [paper])

    result = run_pipeline(settings, ChangedReviewPromptAnalyzer(), [paper])

    assert result["screened"] == 0
    assert result["reviewed"] == 1
    record = load_state(settings.state_file).records[paper.version_key]
    assert record.review_fingerprint == "changed-review-prompt"


class TokenBudgetAnalyzer(DemoAnalyzer):
    def __init__(self):
        self.calls = 0
        self.usage = AnalysisUsage()

    def screen(self, paper):
        self.calls += 1
        self.usage.screening_calls += 1
        self.usage.screening_prompt_tokens += 10
        return super().screen(paper)


def test_prompt_token_budget_stops_additional_calls(tmp_path: Path):
    analyzer = TokenBudgetAnalyzer()
    settings = Settings(
        state_file=tmp_path / "state.json",
        output_dir=tmp_path / "public",
        max_deep_reviews=0,
        min_deep_reviews=0,
        max_prompt_tokens=10,
    )

    result = run_pipeline(settings, analyzer, _demo_papers())

    assert analyzer.calls == 1
    assert result["prompt_tokens"] == 10
    assert result["estimated_cost_rmb"] == 0.00001
    assert result["api_calls_stopped"] == 1


def test_legacy_record_fingerprints_are_backfilled_without_reanalysis(
    tmp_path: Path,
):
    paper = _demo_papers()[0]
    settings = Settings(
        state_file=tmp_path / "state.json",
        output_dir=tmp_path / "public",
        max_deep_reviews=1,
        min_deep_reviews=1,
    )
    run_pipeline(settings, DemoAnalyzer(), [paper])
    state = load_state(settings.state_file)
    state.schema_version = 1
    record = state.records[paper.version_key]
    record.screening_fingerprint = ""
    record.review_fingerprint = None
    save_state(settings.state_file, state)

    result = run_pipeline(settings, DemoAnalyzer(), [paper])

    assert result["screened"] == 0
    assert result["reviewed"] == 0
    migrated = load_state(settings.state_file)
    assert migrated.schema_version == 2
    assert migrated.records[paper.version_key].screening_fingerprint
    assert migrated.records[paper.version_key].review_fingerprint


def test_historical_reanalysis_keeps_reserved_candidate_slot(tmp_path: Path):
    historical = _demo_papers()[0]
    initial_settings = Settings(
        state_file=tmp_path / "state.json",
        output_dir=tmp_path / "public",
        max_candidates=2,
        max_deep_reviews=0,
        min_deep_reviews=0,
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
        min_deep_reviews=0,
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
        min_deep_reviews=0,
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
        self.calls = 0

    def screen(self, paper):
        self.calls += 1
        if self.calls == 2:
            raise KeyboardInterrupt
        return super().screen(paper)


class InterruptingPromotedReviewAnalyzer(ConservativeAnalyzer):
    def review(self, paper, screening):
        raise KeyboardInterrupt


def test_checkpoint_preserves_progress_after_interruption(tmp_path: Path):
    settings = Settings(
        state_file=tmp_path / "state.json",
        output_dir=tmp_path / "public",
        max_candidates=2,
        max_deep_reviews=0,
        min_deep_reviews=0,
        checkpoint_interval=1,
    )

    with pytest.raises(KeyboardInterrupt):
        run_pipeline(settings, InterruptingAnalyzer(), _demo_papers())

    assert len(load_state(settings.state_file).records) == 1


def test_promoted_review_survives_interruption_and_resumes(tmp_path: Path):
    paper = _demo_papers()[0]
    settings = Settings(
        state_file=tmp_path / "state.json",
        output_dir=tmp_path / "public",
        max_candidates=1,
        max_deep_reviews=1,
        min_deep_reviews=1,
        checkpoint_interval=10,
    )

    with pytest.raises(KeyboardInterrupt):
        run_pipeline(settings, InterruptingPromotedReviewAnalyzer(), [paper])

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
        min_deep_reviews=0,
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
        min_deep_reviews=0,
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
