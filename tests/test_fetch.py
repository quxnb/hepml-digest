from pathlib import Path

from hepml_digest.fetch import (
    keyword_score,
    merge_duplicates,
    parse_arxiv_identifier,
    parse_feed_bytes,
)


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
