from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from hepml_digest.fetch import (
    keyword_score,
    fetch_feeds,
    fetch_updates,
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


class FeedResponse:
    def __init__(self, payload: bytes):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return self.payload


def test_fetch_feeds_continues_after_one_source_fails(monkeypatch):
    payload = Path("tests/fixtures/arxiv_rss.xml").read_bytes()

    def urlopen(request, timeout):
        if "broken" in request.full_url:
            raise OSError("feed unavailable")
        return FeedResponse(payload)

    monkeypatch.setattr("hepml_digest.fetch.urllib.request.urlopen", urlopen)
    monkeypatch.setattr("hepml_digest.fetch.time.sleep", lambda _: None)

    papers = fetch_feeds(
        (
            "https://rss.arxiv.org/rss/broken",
            "https://rss.arxiv.org/rss/stat.ML",
        ),
        "test-agent",
    )

    assert len(papers) == 1


def test_fetch_feeds_fails_when_all_sources_fail(monkeypatch):
    def urlopen(request, timeout):
        raise OSError("feed unavailable")

    monkeypatch.setattr("hepml_digest.fetch.urllib.request.urlopen", urlopen)
    monkeypatch.setattr("hepml_digest.fetch.time.sleep", lambda _: None)

    try:
        fetch_feeds(("https://rss.arxiv.org/rss/broken",), "test-agent")
    except RuntimeError as exc:
        assert str(exc) == "All configured arXiv feeds failed"
    else:
        raise AssertionError("all-source failure must abort the fetch")


def test_update_backfill_paginates_and_builds_date_window(monkeypatch):
    payload = Path("tests/fixtures/arxiv_atom.xml").read_bytes()
    empty = payload[:payload.index(b"  <entry>")] + b"</feed>"
    requests = []

    def urlopen(request, timeout):
        requests.append(parse_qs(urlparse(request.full_url).query))
        return FeedResponse(payload if len(requests) == 1 else empty)

    monkeypatch.setattr("hepml_digest.fetch.urllib.request.urlopen", urlopen)
    monkeypatch.setattr("hepml_digest.fetch.time.sleep", lambda _: None)
    since = datetime(2026, 7, 18, tzinfo=timezone.utc)
    until = datetime(2026, 7, 19, tzinfo=timezone.utc)
    pages = list(fetch_updates(["stat.ML"], since, until, "test-agent", page_size=1))
    assert [len(page) for page in pages] == [1, 0]
    assert [query["start"] for query in requests] == [["0"], ["1"]]
    assert requests[0]["search_query"] == ["cat:stat.ML"]
    assert requests[0]["sortBy"] == ["lastUpdatedDate"]
    assert requests[0]["sortOrder"] == ["descending"]


def test_update_backfill_reports_truncation_instead_of_silent_success(monkeypatch):
    payload = Path("tests/fixtures/arxiv_atom.xml").read_bytes()
    monkeypatch.setattr(
        "hepml_digest.fetch.urllib.request.urlopen", lambda *a, **k: FeedResponse(payload),
    )
    since = datetime(2026, 7, 18, tzinfo=timezone.utc)
    until = datetime(2026, 7, 19, tzinfo=timezone.utc)
    pages = fetch_updates(["stat.ML"], since, until, "test-agent", page_size=1, max_pages=1)
    assert len(next(pages)) == 1
    with pytest.raises(RuntimeError, match="page limit reached"):
        next(pages)


def test_update_backfill_stops_before_scanning_older_history(monkeypatch):
    payload = Path("tests/fixtures/arxiv_atom.xml").read_bytes()
    monkeypatch.setattr(
        "hepml_digest.fetch.urllib.request.urlopen", lambda *a, **k: FeedResponse(payload),
    )
    since = datetime(2026, 7, 19, tzinfo=timezone.utc)
    pages = list(fetch_updates(["stat.ML"], since, since, "test-agent", page_size=1))
    assert pages == [[]]


def test_update_backfill_finds_revisions_of_old_submissions(monkeypatch):
    payload = Path("tests/fixtures/arxiv_atom.xml").read_bytes().replace(
        b"<published>2026-07-18", b"<published>2020-07-18",
    )
    monkeypatch.setattr(
        "hepml_digest.fetch.urllib.request.urlopen", lambda *a, **k: FeedResponse(payload),
    )
    since = datetime(2026, 7, 18, tzinfo=timezone.utc)
    until = datetime(2026, 7, 19, tzinfo=timezone.utc)
    pages = list(fetch_updates(["stat.ML"], since, until, "test-agent"))
    assert len(pages[0]) == 1
    assert pages[0][0].published.year == 2020


def test_update_backfill_does_not_treat_an_error_page_as_an_empty_window(monkeypatch):
    monkeypatch.setattr(
        "hepml_digest.fetch.urllib.request.urlopen",
        lambda *a, **k: FeedResponse(b"<html><body>Unavailable</body></html>"),
    )
    now = datetime.now(timezone.utc)
    with pytest.raises(ValueError):
        list(fetch_updates(["stat.ML"], now, now, "test-agent"))


def test_update_page_is_parsed_once(monkeypatch):
    import feedparser

    payload = Path("tests/fixtures/arxiv_atom.xml").read_bytes()
    original = feedparser.parse
    calls = []

    def parse(value):
        calls.append(value)
        return original(value)

    monkeypatch.setattr("hepml_digest.fetch.feedparser.parse", parse)
    monkeypatch.setattr(
        "hepml_digest.fetch.urllib.request.urlopen", lambda *a, **k: FeedResponse(payload),
    )
    since = datetime(2026, 7, 18, tzinfo=timezone.utc)
    until = datetime(2026, 7, 19, tzinfo=timezone.utc)
    assert len(list(fetch_updates(["stat.ML"], since, until, "test-agent"))) == 1
    assert len(calls) == 1
