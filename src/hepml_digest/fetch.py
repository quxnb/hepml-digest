from __future__ import annotations

import calendar
import hashlib
import html
import json
import logging
import re
import time
import urllib.request
from urllib.parse import urlencode
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Iterable, Iterator

import feedparser

from .models import Paper


LOGGER = logging.getLogger(__name__)
ARXIV_LINK_RE = re.compile(r"/(?:abs|pdf)/([^?#]+?)(?:\.pdf)?(?:[?#]|$)")
VERSION_RE = re.compile(r"v(\d+)$")
TAG_RE = re.compile(r"<[^>]+>")

TERMS: dict[str, int] = {
    "neutrino": 5,
    "oscillation": 3,
    "liquid argon": 3,
    "near detector": 3,
    "simulation-based inference": 5,
    "likelihood-free": 5,
    "uncertainty quantification": 4,
    "anomaly detection": 4,
    "domain adaptation": 4,
    "equivariant": 3,
    "calibration": 3,
    "density estimation": 3,
    "optimal transport": 2,
    "graph neural network": 2,
    "generative model": 2,
    "normalizing flow": 3,
    "diffusion": 2,
    "transformer": 1,
}


def parse_arxiv_identifier(url: str) -> tuple[str, int | None]:
    match = ARXIV_LINK_RE.search(url)
    if not match:
        raise ValueError(f"Cannot parse arXiv identifier from {url!r}")
    raw = match.group(1).strip("/")
    version_match = VERSION_RE.search(raw)
    if version_match:
        return raw[: version_match.start()], int(version_match.group(1))
    return raw, None


def _clean(value: str) -> str:
    return " ".join(html.unescape(TAG_RE.sub(" ", value or "")).split())


def _entry_datetime(entry: dict, name: str) -> datetime:
    if name == "updated" and "updated" not in entry:
        return _entry_datetime(entry, "published")
    parsed = entry.get(f"{name}_parsed")
    if parsed:
        return datetime.fromtimestamp(calendar.timegm(parsed), tz=timezone.utc)
    value = entry.get(name)
    if value:
        result = parsedate_to_datetime(value)
        return result if result.tzinfo else result.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


def _entry_link(entry: dict) -> str:
    candidates = [candidate for candidate in (entry.get("link"), entry.get("id"))
                  if candidate and ("arxiv.org/abs/" in candidate or "arxiv.org/pdf/" in candidate)]
    if not candidates:
        raise ValueError("RSS entry has no arXiv link")
    return max(candidates, key=lambda candidate: parse_arxiv_identifier(candidate)[1] or 0)


def parse_feed_bytes(
    payload: bytes, source_category: str | None = None
) -> list[Paper]:
    return _parse_entries(feedparser.parse(payload), source_category)


def _parse_entries(parsed, source_category: str | None = None) -> list[Paper]:
    if parsed.bozo and not parsed.entries:
        raise ValueError(
            f"Invalid feed for {source_category or 'arXiv API'}: "
            f"{parsed.bozo_exception}"
        )

    papers: list[Paper] = []
    for entry in parsed.entries:
        try:
            link = _entry_link(entry)
            base_id, version = parse_arxiv_identifier(link)
            authors = [
                item.get("name", "").strip()
                for item in entry.get("authors", [])
                if item.get("name", "").strip()
            ]
            tags = [
                item.get("term", "").strip()
                for item in entry.get("tags", [])
                if item.get("term", "").strip()
            ]
            if source_category is not None:
                tags.append(source_category)
            published = _entry_datetime(entry, "published")
            updated = _entry_datetime(entry, "updated")
            papers.append(
                Paper(
                    arxiv_id=base_id,
                    version=version,
                    title=_clean(entry.get("title", "")),
                    abstract=_clean(
                        entry.get("summary", entry.get("description", ""))
                    ),
                    authors=authors,
                    link=f"https://arxiv.org/abs/{base_id}",
                    published=published,
                    updated=updated,
                    categories=sorted(set(tags)),
                    doi=entry.get("arxiv_doi"),
                    journal_ref=_clean(entry.get("arxiv_journal_ref", "")) or None,
                    abstract_source="arXiv",
                )
            )
        except (ValueError, TypeError) as exc:
            LOGGER.warning("Skipping malformed RSS entry: %s", exc)
    return papers


def fetch_feeds(
    feed_urls: Iterable[str], user_agent: str, timeout: float = 30.0,
    *, failed_urls: list[str] | None = None,
) -> list[Paper]:
    feed_urls = tuple(feed_urls)
    papers: list[Paper] = []
    successful_feeds = 0
    for url in feed_urls:
        category = url.rstrip("/").rsplit("/", 1)[-1]
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": user_agent,
                "Accept": "application/rss+xml, application/xml, text/xml",
            },
        )
        for attempt in range(1, 3):
            try:
                LOGGER.info("Fetching %s (attempt %d/2)", url, attempt)
                with urllib.request.urlopen(
                    request, timeout=timeout
                ) as response:
                    papers.extend(
                        parse_feed_bytes(response.read(), category)
                    )
            except (OSError, ValueError) as exc:
                if attempt == 2:
                    LOGGER.error("Failed to fetch %s: %s", url, exc)
                    if failed_urls is not None:
                        failed_urls.append(url)
                else:
                    LOGGER.warning(
                        "Fetching %s failed; retrying: %s", url, exc
                    )
                    time.sleep(2)
            else:
                successful_feeds += 1
                break
    if feed_urls and successful_feeds == 0:
        raise RuntimeError("All configured arXiv feeds failed")
    return merge_duplicates(papers)


def fetch_recent(
    categories: Iterable[str],
    max_results: int,
    user_agent: str,
    timeout: float = 30.0,
) -> list[Paper]:
    """Fetch recent papers once to seed a new, otherwise-empty digest."""
    categories = tuple(categories)
    if not categories or max_results <= 0:
        return []
    query = " OR ".join(f"cat:{category}" for category in categories)
    parameters = urlencode(
        {
            "search_query": query,
            "start": 0,
            "max_results": min(max_results, 200),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
    )
    url = f"https://export.arxiv.org/api/query?{parameters}"
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": user_agent,
            "Accept": "application/atom+xml, application/xml",
        },
    )
    LOGGER.info("Bootstrapping from arXiv API with %d results", max_results)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return merge_duplicates(parse_feed_bytes(response.read()))


def fetch_updates(
    categories: Iterable[str],
    since: datetime,
    until: datetime,
    user_agent: str,
    timeout: float = 30.0,
    page_size: int = 200,
    max_pages: int = 25,
) -> Iterator[list[Paper]]:
    """Yield bounded update pages; exhaustion means the whole window was read."""
    categories = tuple(categories)
    if not categories:
        return
    if not 1 <= page_size <= 200 or max_pages <= 0:
        raise ValueError("Invalid arXiv update pagination limits")
    if since > until:
        raise ValueError("arXiv update window must not run backwards")
    query = " OR ".join(f"cat:{category}" for category in categories)
    for page_index in range(max_pages):
        if page_index:
            time.sleep(3)
        start = page_index * page_size
        parameters = urlencode({
            "search_query": query,
            "start": start,
            "max_results": page_size,
            "sortBy": "lastUpdatedDate",
            "sortOrder": "descending",
        })
        request = urllib.request.Request(
            f"https://export.arxiv.org/api/query?{parameters}",
            headers={"User-Agent": user_agent, "Accept": "application/atom+xml"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
        parsed = feedparser.parse(payload)
        papers = _parse_entries(parsed)
        if not parsed.feed.get("id"):
            raise ValueError("arXiv update response has no feed identifier")
        count = len(parsed.entries)
        # The API supports sorting, but not a lastUpdatedDate range filter.
        # Scan newest updates until crossing the overlap window's lower bound.
        yield [paper for paper in papers if since <= paper.updated <= until]
        if len(papers) != count:
            raise ValueError("arXiv update page contains malformed entries")
        total = parsed.feed.get("opensearch_totalresults")
        if (
            any(paper.updated < since for paper in papers)
            or count < page_size
            or (total is not None and start + count >= int(total))
        ):
            return
    raise RuntimeError(
        "arXiv update page limit reached; increase BACKFILL_MAX_PAGES. "
        "The backfill cursor has not advanced."
    )


def parse_inspire_record(hit: dict) -> Paper:
    metadata = hit["metadata"]
    title = _clean(metadata["titles"][0]["title"])
    if not title:
        raise ValueError("INSPIRE record has no title")
    eprint = next(iter(metadata.get("arxiv_eprints", [])), {})
    raw_id = eprint.get("value", "").removeprefix("arXiv:")
    arxiv_id, version = parse_arxiv_identifier(f"https://arxiv.org/abs/{raw_id}") if raw_id else (None, None)
    doi = next((item["value"] for item in metadata.get("dois", [])
                if item.get("value") and item.get("material") not in {"erratum", "addendum"}), None)
    publication = next((item for item in metadata.get("publication_info", [])
                        if item.get("journal_title") and item.get("material") not in {"erratum", "addendum"}), {})
    reference = " ".join(str(value) for value in (
        publication.get("journal_title"), publication.get("journal_volume"),
        f"({publication['year']})" if publication.get("year") else None,
        publication.get("artid") or publication.get("page_start"),
    ) if value)
    abstract = next((item for item in metadata.get("abstracts", []) if item.get("value")), {})
    inspire_id = str(hit.get("id") or metadata.get("control_number") or "") or None
    if inspire_id is None:
        raise ValueError("INSPIRE record has no identifier")
    return Paper(
        arxiv_id=arxiv_id, version=version, doi=doi, inspire_id=inspire_id,
        title=title, abstract=_clean(abstract.get("value", "")),
        abstract_source=f"INSPIRE ({abstract.get('source') or 'record'})" if abstract else None,
        authors=[item["full_name"] for item in metadata.get("authors", []) if item.get("full_name")],
        link=f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else f"https://inspirehep.net/literature/{inspire_id}",
        # INSPIRE timestamps refer to record availability, not journal publication dates.
        published=datetime.fromisoformat(hit["created"].replace("Z", "+00:00")),
        updated=datetime.fromisoformat(hit["updated"].replace("Z", "+00:00")),
        journal_ref=reference or None,
        categories=sorted(set(eprint.get("categories", [])) | {
            f"inspire:{item['term']}" for item in metadata.get("inspire_categories", []) if item.get("term")
        }),
    )


def fetch_inspire(
    query: str, since: datetime, until: datetime, user_agent: str,
    timeout: float = 30.0, page_size: int = 200, max_pages: int = 25,
) -> Iterator[list[Paper]]:
    """Read an update window, yielding each page before advancing a source cursor."""
    if not 1 <= page_size <= 200 or max_pages <= 0 or since > until:
        raise ValueError("Invalid INSPIRE window or pagination limits")
    search = f"({query}) and du >= {since:%Y-%m-%d} and du < {until + timedelta(days=1):%Y-%m-%d}"
    for page in range(1, max_pages + 1):
        if page > 1:
            time.sleep(1)
        parameters = urlencode({
            "q": search, "sort": "mostrecent", "page": page, "size": page_size,
            "fields": "titles,abstracts,authors.full_name,arxiv_eprints,dois,publication_info,inspire_categories",
        })
        request = urllib.request.Request(
            f"https://inspirehep.net/api/literature?{parameters}",
            headers={"User-Agent": user_agent, "Accept": "application/json"},
        )
        LOGGER.info("Fetching INSPIRE updates (page %d)", page)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read())
        hits = result["hits"]["hits"]
        papers = []
        for hit in hits:
            try:
                papers.append(parse_inspire_record(hit))
            except (KeyError, IndexError, ValueError, TypeError):
                LOGGER.warning("Malformed INSPIRE entry; update window will remain incomplete")
        yield [paper for paper in papers if since <= paper.updated <= until]
        if len(papers) != len(hits):
            raise ValueError("INSPIRE update page contains malformed entries")
        total = int(result["hits"]["total"])
        offset = (page - 1) * page_size
        if len(hits) < min(page_size, max(0, total - offset)):
            raise RuntimeError("INSPIRE returned an incomplete result page")
        if offset + len(hits) >= total:
            return
    raise RuntimeError("INSPIRE page limit reached; increase BACKFILL_MAX_PAGES")


def enrich_paper(current: Paper, incoming: Paper, *, preserve_content: bool = False) -> None:
    """Add bibliographic metadata without changing an assigned record or Feed ID."""
    identity, feed_identity = current.version_key, current.feed_key
    for name in ("arxiv_id", "doi", "inspire_id"):
        if not getattr(current, name):
            setattr(current, name, getattr(incoming, name))
    if current.version is None:
        current.version = incoming.version
    current.journal_ref = incoming.journal_ref or current.journal_ref
    current.categories = sorted(set(current.categories) | set(incoming.categories))
    current.published = min(current.published, incoming.published)
    current.updated = max(current.updated, incoming.updated)
    if not current.abstract.strip() or (
        not preserve_content and len(incoming.abstract) > len(current.abstract)
    ):
        current.abstract = incoming.abstract
        current.abstract_source = incoming.abstract_source
    if current.version_key != identity:
        current.identity = identity
    if current.feed_key != feed_identity:
        current.feed_identity = feed_identity


def merge_duplicates(
    papers: Iterable[Paper], *, protected_keys: set[str] | None = None,
) -> list[Paper]:
    """Merge exact identifiers, preserving known arXiv editions and first-seen IDs."""
    merged: dict[str, Paper] = {}
    index: dict[str, set[str]] = {}
    order: dict[str, int] = {}
    protected_keys = protected_keys or set()
    for paper in papers:
        original_key = paper.version_key
        matches = {key for alias in paper.aliases for key in index.get(alias, ()) if key in merged}
        candidates = [merged[key] for key in sorted(matches, key=order.__getitem__)
                      if not (paper.arxiv_id and merged[key].arxiv_id
                              and paper.arxiv_id != merged[key].arxiv_id)]
        if not paper.arxiv_id and len({item.arxiv_id for item in candidates if item.arxiv_id}) > 1:
            candidates = []  # Conflicting arXiv IDs are not resolved by guessing.
        if candidates:
            family = candidates[0].feed_key
            for item in candidates:
                item.feed_identity = family
            paper.feed_identity = family
        version = paper.version if paper.arxiv_id else None
        if version is None:
            version = max((item.version for item in candidates if item.version is not None), default=None)
        same = [item for item in candidates if item.version is None or item.version == version]
        if same:
            current = same[0]
            key = current.version_key
            enrich_paper(current, paper, preserve_content=key in protected_keys)
            for duplicate in same[1:]:
                duplicate_key = duplicate.version_key
                enrich_paper(current, duplicate, preserve_content=key in protected_keys)
                duplicate.identity = key
                merged.pop(duplicate_key)
                for aliases in index.values():
                    if duplicate_key in aliases:
                        aliases.remove(duplicate_key)
                        aliases.add(key)
            if paper is not current:
                enrich_paper(paper, current, preserve_content=original_key in protected_keys)
                paper.identity = key
        else:
            key = paper.version_key
            current = paper
            order.setdefault(key, len(order))
            merged[key] = current
        for alias in set(current.aliases) | set(paper.aliases):
            index.setdefault(alias, set()).add(key)
    return list(merged.values())


def keyword_score(paper: Paper) -> int:
    text = f"{paper.title} {paper.abstract}".lower()
    return sum(weight for term, weight in TERMS.items() if term in text)


def select_candidates(
    papers: Iterable[Paper],
    limit: int,
    discovery_slots: int,
    method_slots: int | None = None,
    hep_application_slots: int | None = None,
) -> list[Paper]:
    if limit <= 0:
        return []

    ranked = sorted(
        papers,
        key=lambda paper: (keyword_score(paper), paper.updated),
        reverse=True,
    )
    if len(ranked) <= limit:
        return ranked

    discovery_slots = max(0, min(discovery_slots, limit))
    if method_slots is None or hep_application_slots is None:
        primary = ranked[: limit - discovery_slots]
    else:
        primary_limit = limit - discovery_slots
        method_slots = max(0, min(method_slots, primary_limit))
        hep_application_slots = max(
            0, min(hep_application_slots, primary_limit - method_slots)
        )
        methods = [
            paper for paper in ranked if paper.digest_track == "method_radar"
        ]
        applications = [
            paper
            for paper in ranked
            if paper.digest_track == "hep_application"
        ]
        primary = [
            *methods[:method_slots],
            *applications[:hep_application_slots],
        ]

    selected_keys = {paper.version_key for paper in primary}
    remaining = [
        paper for paper in ranked if paper.version_key not in selected_keys
    ]
    discovery = sorted(
        remaining,
        key=lambda paper: hashlib.sha256(
            paper.version_key.encode("utf-8")
        ).hexdigest(),
    )[:discovery_slots]
    selected = [*primary, *discovery]

    # If one track has fewer papers than its quota, fill the unused capacity
    # from the globally ranked remainder instead of screening fewer papers.
    selected_keys = {paper.version_key for paper in selected}
    for paper in ranked:
        if len(selected) >= limit:
            break
        if paper.version_key not in selected_keys:
            selected.append(paper)
            selected_keys.add(paper.version_key)
    return selected
