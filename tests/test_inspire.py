from datetime import datetime, timezone
import json
from urllib.parse import parse_qs, urlparse
from xml.etree import ElementTree

import pytest
from pydantic import ValidationError

from hepml_digest.cli import _demo_papers
from hepml_digest.config import Settings
from hepml_digest.evaluation import evaluate, export_evaluation
from hepml_digest.fetch import fetch_inspire, merge_duplicates, parse_inspire_record
from hepml_digest.llm import DemoAnalyzer
from hepml_digest.models import Paper
from hepml_digest.pipeline import run_pipeline
from hepml_digest.store import load_state


def hit(number="100", arxiv=None, doi="10.1234/neutrino-test", abstract=True):
    metadata = {
        "titles": [{"title": "Example neutrino cross-section measurement"}],
        "authors": [{"full_name": "Example Collaboration"}],
        "publication_info": [{"journal_title": "Example Journal", "journal_volume": "1", "year": 2026, "artid": "001"}],
        "inspire_categories": [{"term": "Experiment-HEP"}],
    }
    if arxiv:
        metadata["arxiv_eprints"] = [{"value": arxiv, "categories": ["hep-ex"]}]
    if doi:
        metadata["dois"] = [{"value": doi}]
    if abstract:
        metadata["abstracts"] = [{"source": "publisher", "value": "Synthetic abstract for testing neutrino measurements and detector calibration."}]
    return {"id": number, "created": "2026-11-01T00:00:00Z", "updated": "2026-11-02T00:00:00Z", "metadata": metadata}


def settings_for(tmp_path, **kwargs):
    return Settings(state_file=tmp_path / "state.json", output_dir=tmp_path / "public", **kwargs)


class Response:
    def __init__(self, records, total=None):
        self.payload = json.dumps({"hits": {"hits": records, "total": len(records) if total is None else total}}).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.payload


def test_journal_record_without_arxiv_has_native_identity():
    paper = parse_inspire_record(hit(doi="https://doi.org/10.1234/NEUTRINO-TEST"))
    assert paper.arxiv_id is None
    assert paper.version is None
    assert paper.doi == "10.1234/neutrino-test"
    assert paper.version_key == "doi:10.1234/neutrino-test"
    assert paper.inspire_id == "100"
    assert paper.journal_ref == "Example Journal 1 (2026) 001"
    assert paper.abstract_source == "INSPIRE (publisher)"
    assert paper.digest_track == "hep_application"


def test_inspire_does_not_invent_an_arxiv_version_or_abstract():
    paper = parse_inspire_record(hit(arxiv="2610.12345", abstract=False))
    assert paper.version is None
    assert paper.version_key == "arxiv:2610.12345"
    assert paper.abstract == ""
    versioned = parse_inspire_record(hit(arxiv="2610.12345v4"))
    assert versioned.version == 4


def test_paper_requires_a_real_source_identifier():
    data = _demo_papers()[0].model_dump()
    data["arxiv_id"] = None
    with pytest.raises(ValidationError, match="requires an arXiv ID"):
        Paper.model_validate(data)


def test_dedup_enriches_known_arxiv_without_changing_ids():
    original = _demo_papers()[0]
    original.version = 3
    original_key, original_feed = original.version_key, original.feed_key
    incoming = parse_inspire_record(hit(arxiv=original.arxiv_id))
    merged = merge_duplicates([original, incoming], protected_keys={original_key})
    assert len(merged) == 1
    assert merged[0].version_key == original_key
    assert merged[0].feed_key == original_feed
    assert merged[0].version == 3
    assert merged[0].doi == incoming.doi
    assert merged[0].journal_ref
    assert merged[0].abstract == _demo_papers()[0].abstract


def test_inspire_first_identity_survives_doi_and_arxiv_metadata():
    original = parse_inspire_record(hit(doi=None))
    original_key, original_feed = original.version_key, original.feed_key
    with_doi = parse_inspire_record(hit())
    with_arxiv = parse_inspire_record(hit(arxiv="2610.12345v4"))
    merged = merge_duplicates([original, with_doi, with_arxiv])
    assert len(merged) == 1
    assert original.version_key == original_key == "inspire:100"
    assert original.feed_key == original_feed
    assert original.arxiv_id == "2610.12345"
    assert original.version == 4
    revision = with_arxiv.model_copy(deep=True)
    revision.identity = revision.feed_identity = None
    revision.version = 5
    editions = merge_duplicates([original, revision])
    assert len(editions) == 2
    assert editions[0].feed_key == editions[1].feed_key == original_feed


def test_known_arxiv_versions_remain_distinct_when_publication_arrives():
    first = _demo_papers()[0]
    second = first.model_copy(deep=True)
    second.version = 2
    result = merge_duplicates([first, second, parse_inspire_record(hit(arxiv=first.arxiv_id))])
    assert len(result) == 2
    assert {paper.version for paper in result} == {1, 2}
    assert second.journal_ref
    assert len({paper.feed_key for paper in result}) == 1


def test_conflicting_arxiv_ids_are_not_merged_by_doi_alone():
    first = parse_inspire_record(hit("100", "2610.11111", "10.1234/shared"))
    second = parse_inspire_record(hit("101", "2610.22222", "10.1234/shared"))
    assert len(merge_duplicates([first, second])) == 2


def test_inspire_fetch_uses_update_dates_and_pagination(monkeypatch):
    requests = []

    def urlopen(request, timeout):
        requests.append(parse_qs(urlparse(request.full_url).query))
        return Response([hit(str(len(requests)))], total=2)

    monkeypatch.setattr("hepml_digest.fetch.urllib.request.urlopen", urlopen)
    monkeypatch.setattr("hepml_digest.fetch.time.sleep", lambda _: None)
    since = datetime(2026, 10, 26, tzinfo=timezone.utc)
    until = datetime(2026, 11, 2, 12, tzinfo=timezone.utc)
    pages = list(fetch_inspire("neutrino OR test", since, until, "test-agent", page_size=1))
    assert len(pages) == 2
    assert requests[0]["q"] == ["(neutrino OR test) and du >= 2026-10-26 and du < 2026-11-03"]
    assert [item["page"] for item in requests] == [["1"], ["2"]]


def test_inspire_page_limit_and_malformed_entries_do_not_claim_completion(monkeypatch):
    monkeypatch.setattr("hepml_digest.fetch.urllib.request.urlopen", lambda *a, **k: Response([hit()], total=2))
    since = datetime(2026, 10, 26, tzinfo=timezone.utc)
    until = datetime(2026, 11, 3, tzinfo=timezone.utc)
    pages = fetch_inspire("neutrino", since, until, "test", page_size=1, max_pages=1)
    assert len(next(pages)) == 1
    with pytest.raises(RuntimeError, match="page limit"):
        next(pages)
    malformed = hit("101")
    del malformed["metadata"]["titles"]
    monkeypatch.setattr("hepml_digest.fetch.urllib.request.urlopen", lambda *a, **k: Response([hit(), malformed]))
    pages = fetch_inspire("neutrino", since, until, "test")
    assert len(next(pages)) == 1
    with pytest.raises(ValueError, match="malformed"):
        next(pages)


def test_publication_metadata_update_does_not_call_models_again(tmp_path):
    settings = settings_for(tmp_path)
    original = _demo_papers()[0]
    run_pipeline(settings, DemoAnalyzer(), [original])
    feed = ElementTree.parse(settings.output_dir / "rss.xml")
    guid = feed.findtext("./channel/item/guid")
    incoming = parse_inspire_record(hit(arxiv=original.arxiv_id))
    result = run_pipeline(settings, DemoAnalyzer(), [incoming])
    assert result["screened"] == result["reviewed"] == 0
    assert result["published"] == 1
    assert ElementTree.parse(settings.output_dir / "rss.xml").findtext("./channel/item/guid") == guid
    state = load_state(settings.state_file)
    assert list(state.records) == [original.version_key]
    assert state.records[original.version_key].paper.journal_ref


def test_late_identifier_bridge_coalesces_cached_records_and_keeps_first_guid(tmp_path):
    settings = settings_for(tmp_path)
    first = parse_inspire_record(hit("100"))
    run_pipeline(settings, DemoAnalyzer(), [first])
    guid = ElementTree.parse(settings.output_dir / "rss.xml").findtext("./channel/item/guid")
    second = parse_inspire_record(hit("101", arxiv="2610.12345v4", doi=None))
    second.abstract = "Synthetic collider example with an independently cached analysis."
    run_pipeline(settings, DemoAnalyzer(), [second])
    assert len(load_state(settings.state_file).records) == 2
    joined = parse_inspire_record(hit("100", arxiv="2610.12345v4"))
    result = run_pipeline(settings, DemoAnalyzer(), [joined])
    assert result["screened"] == result["reviewed"] == 0
    assert result["published"] == 1
    saved = load_state(settings.state_file)
    assert list(saved.records) == [first.version_key]
    assert saved.records[first.version_key].paper.abstract == second.abstract
    assert saved.records[first.version_key].review_status == "complete"
    assert ElementTree.parse(settings.output_dir / "rss.xml").findtext("./channel/item/guid") == guid


def test_journal_only_papers_publish_and_evaluate_independently(tmp_path):
    papers = [parse_inspire_record(hit("100")), parse_inspire_record(hit("101", doi=None))]
    settings = settings_for(tmp_path)
    result = run_pipeline(settings, DemoAnalyzer(), papers)
    assert result["published"] == result["screened"] == 2
    root = ElementTree.parse(settings.output_dir / "rss.xml").getroot()
    assert len({item.findtext("guid") for item in root.findall("./channel/item")}) == 2
    html = (settings.output_dir / "index.html").read_text()
    assert "Example Journal" in html and "INSPIRE 记录" in html
    assert "arXiv:None" not in html
    dataset = export_evaluation(load_state(settings.state_file), tmp_path / "labels.json", 0.55)
    for item in dataset.items:
        item.human_keep = True
    assert evaluate(load_state(settings.state_file), dataset, 0.55)["true_positives"] == 2


def test_missing_abstract_is_queued_without_model_calls_then_filled(tmp_path):
    settings = settings_for(tmp_path)
    missing = parse_inspire_record(hit(abstract=False))
    result = run_pipeline(settings, DemoAnalyzer(), [missing])
    assert result["screened"] == result["reviewed"] == 0
    assert result["missing_abstract_papers"] == 1
    filled = parse_inspire_record(hit())
    resumed = run_pipeline(settings, DemoAnalyzer(), [filled])
    assert resumed["screened"] == 1
    assert resumed["missing_abstract_papers"] == 0


def test_inspire_partial_failure_keeps_pages_and_source_cursor(tmp_path, monkeypatch):
    settings = settings_for(tmp_path, bootstrap_results=0, max_candidates=0, max_deep_reviews=0)
    monkeypatch.setattr("hepml_digest.pipeline.fetch_feeds", lambda *a, **k: [])
    monkeypatch.setattr("hepml_digest.pipeline.fetch_updates", lambda *a, **k: iter([]))

    def partial(*args, **kwargs):
        yield [parse_inspire_record(hit())]
        raise OSError("second page failed")

    monkeypatch.setattr("hepml_digest.pipeline.fetch_inspire", partial)
    first = run_pipeline(settings, DemoAnalyzer())
    saved = load_state(settings.state_file)
    assert first["fetch_errors"] == 1
    assert first["queued_papers"] == 1
    assert saved.last_backfill_at is not None
    assert saved.last_inspire_fetch_at is None
    anchor = saved.pending_inspire_since

    def complete(query, since, *args, **kwargs):
        assert since == anchor
        yield [parse_inspire_record(hit())]

    monkeypatch.setattr("hepml_digest.pipeline.fetch_inspire", complete)
    settings.max_candidates = 1
    resumed = run_pipeline(settings, DemoAnalyzer())
    assert resumed["screened"] == 1
    saved = load_state(settings.state_file)
    assert saved.pending_inspire_since is None
    assert saved.last_inspire_fetch_at is not None


def test_inspire_rescues_arxiv_failure_without_spending_a_second_quota(tmp_path, monkeypatch):
    settings = settings_for(tmp_path, bootstrap_results=0, max_candidates=1, max_deep_reviews=0)

    def unavailable(*args, **kwargs):
        raise OSError("arxiv offline")

    monkeypatch.setattr("hepml_digest.pipeline.fetch_feeds", unavailable)
    monkeypatch.setattr("hepml_digest.pipeline.fetch_updates", unavailable)
    monkeypatch.setattr("hepml_digest.pipeline.fetch_inspire", lambda *a, **k: iter([[
        parse_inspire_record(hit("100")), parse_inspire_record(hit("101", doi=None)),
    ]]))
    result = run_pipeline(settings, DemoAnalyzer())
    assert result["screened"] == result["queued_papers"] == 1
    assert result["source_failure_runs"] == 0
    saved = load_state(settings.state_file)
    assert saved.last_backfill_at is None
    assert saved.pending_backfill_since is not None
    assert saved.last_inspire_fetch_at is not None
    assert saved.pending_inspire_since is None


def test_sources_share_quota_and_cross_source_duplicates_are_screened_once(tmp_path, monkeypatch):
    settings = settings_for(tmp_path, bootstrap_results=0, max_candidates=1, max_deep_reviews=0)
    arxiv = _demo_papers()[0]
    monkeypatch.setattr("hepml_digest.pipeline.fetch_feeds", lambda *a, **k: [arxiv])
    monkeypatch.setattr("hepml_digest.pipeline.fetch_updates", lambda *a, **k: iter([]))
    monkeypatch.setattr("hepml_digest.pipeline.fetch_inspire", lambda *a, **k: iter([[
        parse_inspire_record(hit("100", arxiv=arxiv.arxiv_id)),
        parse_inspire_record(hit("101", doi=None)),
    ]]))
    result = run_pipeline(settings, DemoAnalyzer())
    assert result["fetched"] == 2
    assert result["screened"] == 1
    assert result["queued_papers"] == 1


def test_empty_inspire_page_with_nonzero_total_is_not_a_complete_window(monkeypatch):
    monkeypatch.setattr("hepml_digest.fetch.urllib.request.urlopen", lambda *a, **k: Response([], total=1))
    now = datetime.now(timezone.utc)
    pages = fetch_inspire("neutrino", now, now, "test")
    assert next(pages) == []
    with pytest.raises(RuntimeError, match="incomplete"):
        next(pages)
