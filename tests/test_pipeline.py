from pathlib import Path
from xml.etree import ElementTree

from hepml_digest.cli import _demo_papers
from hepml_digest.config import Settings
from hepml_digest.llm import DemoAnalyzer
from hepml_digest.models import Review, Screening
from hepml_digest.pipeline import run_pipeline
from hepml_digest.store import load_state


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
