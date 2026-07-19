from __future__ import annotations

import calendar
import hashlib
import html
import logging
import re
import urllib.request
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Iterable

import feedparser

from .models import Paper


LOGGER = logging.getLogger(__name__)
ARXIV_LINK_RE = re.compile(r"/(?:abs|pdf)/([^?#]+?)(?:\.pdf)?(?:[?#]|$)")
VERSION_RE = re.compile(r"v(\d+)$")
TAG_RE = re.compile(r"<[^>]+>")

TERMS: dict[str, int] = {
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


def parse_arxiv_identifier(url: str) -> tuple[str, int]:
    match = ARXIV_LINK_RE.search(url)
    if not match:
        raise ValueError(f"Cannot parse arXiv identifier from {url!r}")
    raw = match.group(1).strip("/")
    version_match = VERSION_RE.search(raw)
    if version_match:
        return raw[: version_match.start()], int(version_match.group(1))
    return raw, 1


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
    for candidate in (entry.get("link"), entry.get("id")):
        if candidate and ("arxiv.org/abs/" in candidate or "arxiv.org/pdf/" in candidate):
            return candidate
    raise ValueError("RSS entry has no arXiv link")


def parse_feed_bytes(payload: bytes, source_category: str) -> list[Paper]:
    parsed = feedparser.parse(payload)
    if parsed.bozo and not parsed.entries:
        raise ValueError(f"Invalid feed for {source_category}: {parsed.bozo_exception}")

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
            categories = sorted(set([source_category, *tags]))
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
                    categories=categories,
                )
            )
        except (ValueError, TypeError) as exc:
            LOGGER.warning("Skipping malformed RSS entry: %s", exc)
    return papers


def fetch_feeds(
    feed_urls: Iterable[str], user_agent: str, timeout: float = 30.0
) -> list[Paper]:
    papers: list[Paper] = []
    for url in feed_urls:
        category = url.rstrip("/").rsplit("/", 1)[-1]
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": user_agent,
                "Accept": "application/rss+xml, application/xml, text/xml",
            },
        )
        LOGGER.info("Fetching %s", url)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            papers.extend(parse_feed_bytes(response.read(), category))
    return merge_duplicates(papers)


def merge_duplicates(papers: Iterable[Paper]) -> list[Paper]:
    merged: dict[str, Paper] = {}
    for paper in papers:
        key = paper.version_key
        current = merged.get(key)
        if current is None:
            merged[key] = paper
            continue
        current.categories = sorted(
            set(current.categories).union(paper.categories)
        )
        if len(paper.abstract) > len(current.abstract):
            current.abstract = paper.abstract
    return list(merged.values())


def keyword_score(paper: Paper) -> int:
    text = f"{paper.title} {paper.abstract}".lower()
    return sum(weight for term, weight in TERMS.items() if term in text)


def select_candidates(
    papers: Iterable[Paper], limit: int, discovery_slots: int
) -> list[Paper]:
    ranked = sorted(
        papers,
        key=lambda paper: (keyword_score(paper), paper.updated),
        reverse=True,
    )
    if len(ranked) <= limit:
        return ranked

    discovery_slots = max(0, min(discovery_slots, limit))
    primary = ranked[: limit - discovery_slots]
    remaining = ranked[limit - discovery_slots :]
    discovery = sorted(
        remaining,
        key=lambda paper: hashlib.sha256(
            paper.version_key.encode("utf-8")
        ).hexdigest(),
    )[:discovery_slots]
    return [*primary, *discovery]
