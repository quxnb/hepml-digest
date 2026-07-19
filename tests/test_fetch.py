from datetime import datetime, timezone
from pathlib import Path

from hepml_digest.fetch import (
    keyword_score,
    merge_duplicates,
    parse_arxiv_identifier,
    parse_feed_bytes,
    select_candidates,
)
from hepml_digest.models import Paper


def test_parse_modern_and_legacy_ids():
    assert parse_arxiv_identifier(
        "https://arxiv.org/abs/2607.01234v2"
    ) == ("2607.01234", 2)
    assert parse_arxiv_identifier(
        "https://arxiv.org/pdf/hep-ex/0307015v3.pdf"
    ) == ("hep-ex/0307015", 3)


def test_parse_feed_and_clean_html():
    payload = Path("tests/fixtures/arxiv_rss.xml").read_bytes()
    papers = parse_feed_bytes(payload, "stat.ML")
    assert len(papers) == 1
    paper = papers[0]
    assert paper.arxiv_id == "2607.01234"
    assert paper.version == 2
    assert "<b>" not in paper.abstract
    assert keyword_score(paper) >= 8


def test_parse_arxiv_api_atom_without_synthetic_category():
    payload = Path("tests/fixtures/arxiv_atom.xml").read_bytes()
    papers = parse_feed_bytes(payload)
    assert len(papers) == 1
    assert papers[0].arxiv_id == "2607.04567"
    assert papers[0].categories == ["stat.ML"]


def test_merge_cross_list_categories():
    payload = Path("tests/fixtures/arxiv_rss.xml").read_bytes()
    first = parse_feed_bytes(payload, "stat.ML")[0]
    second = first.model_copy(deep=True)
    second.categories = ["cs.LG"]
    merged = merge_duplicates([first, second])
    assert len(merged) == 1
    assert merged[0].categories == ["cs.LG", "stat.ML"]


def test_candidate_selection_reserves_both_tracks_and_discovery():
    now = datetime.now(timezone.utc)

    def paper(index: int, category: str) -> Paper:
        arxiv_id = f"2607.{index:05d}"
        return Paper(
            arxiv_id=arxiv_id,
            title=f"Paper {index}",
            abstract="transformer method",
            link=f"https://arxiv.org/abs/{arxiv_id}",
            published=now,
            updated=now,
            categories=[category],
        )

    papers = [paper(index, "cs.LG") for index in range(50)]
    papers.extend(paper(100 + index, "hep-ex") for index in range(20))
    selected = select_candidates(
        papers,
        limit=60,
        discovery_slots=10,
        method_slots=40,
        hep_application_slots=10,
    )

    assert len(selected) == 60
    assert all(item.digest_track == "method_radar" for item in selected[:40])
    assert all(
        item.digest_track == "hep_application" for item in selected[40:50]
    )
    assert len({item.version_key for item in selected}) == 60
