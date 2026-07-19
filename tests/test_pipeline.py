from pathlib import Path
from xml.etree import ElementTree

from hepml_digest.cli import _demo_papers
from hepml_digest.config import Settings
from hepml_digest.llm import DemoAnalyzer
from hepml_digest.pipeline import run_pipeline
from hepml_digest.store import load_state


def test_demo_pipeline_is_idempotent(tmp_path: Path):
    settings = Settings(
        state_file=tmp_path / "state.json",
        output_dir=tmp_path / "public",
        site_url="https://example.test/hepml",
        max_candidates=10,
        max_deep_reviews=2,
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
    assert len(load_state(settings.state_file).records) == 2

    rss_root = ElementTree.parse(settings.output_dir / "rss.xml").getroot()
    assert rss_root.findtext("./channel/link") == settings.site_url
    atom_link = rss_root.find(
        "./channel/{http://www.w3.org/2005/Atom}link"
    )
    assert atom_link is not None
    assert atom_link.attrib["href"] == f"{settings.site_url}/rss.xml"
