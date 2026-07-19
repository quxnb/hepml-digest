from __future__ import annotations

import html
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from feedgen.feed import FeedGenerator

from .models import Record, State


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _list_block(title: str, values: list[str]) -> str:
    if not values:
        return ""
    items = "".join(f"<li>{html.escape(value)}</li>" for value in values)
    return f"<h4>{html.escape(title)}</h4><ul>{items}</ul>"


def record_html(record: Record) -> str:
    paper = record.paper
    screen = record.screening
    parts = [
        f"<p><strong>方法：</strong>{html.escape(screen.method)}</p>",
        f"<p><strong>相关性：</strong>{screen.relevance:.2f}；"
        f"<strong>证据等级：</strong>{html.escape(screen.evidence_level)}</p>",
        f"<p><strong>初筛理由：</strong>{html.escape(screen.reason)}</p>",
    ]
    if record.review:
        review = record.review
        parts.extend(
            [
                f"<p><strong>方法总结：</strong>{html.escape(review.summary_cn)}</p>",
                _list_block("论文明确支持的结论", review.paper_claims),
                _list_block("对高能实验物理的潜在帮助", review.hep_opportunities),
                _list_block("迁移风险", review.transfer_risks),
                f"<p><strong>最小验证方案：</strong>"
                f"{html.escape(review.validation_plan)}</p>",
            ]
        )
    else:
        parts.append("<p><em>本条目仅完成摘要级筛选。</em></p>")
    parts.append(
        f"<p><a href=\"{html.escape(paper.link, quote=True)}\">arXiv 页面</a>"
        f" · {html.escape(', '.join(paper.categories))}</p>"
    )
    return "".join(parts)


def _eligible_records(
    state: State, publish_threshold: float, max_items: int
) -> list[Record]:
    latest: dict[str, Record] = {}
    for record in state.records.values():
        if record.screening.relevance < publish_threshold:
            continue
        previous = latest.get(record.paper.arxiv_id)
        if previous is None or record.paper.version > previous.paper.version:
            latest[record.paper.arxiv_id] = record
    return sorted(
        latest.values(),
        key=lambda item: (item.processed_at, item.paper.updated),
        reverse=True,
    )[:max_items]


def _build_feed(
    records: list[Record], site_url: str, feed_title: str
) -> FeedGenerator:
    now = datetime.now(timezone.utc)
    feed = FeedGenerator()
    feed.id(site_url)
    feed.title(feed_title)
    feed.subtitle("统计与机器学习方法对高能实验物理的每日启发")
    feed.author({"name": "HEP-ML Digest Bot"})
    feed.link(href=site_url, rel="alternate")
    feed.link(href=f"{site_url}/atom.xml", rel="self")
    feed.language("zh-CN")
    feed.updated(now)

    for record in reversed(records):
        paper = record.paper
        entry = feed.add_entry()
        entry.id(f"arxiv:{paper.arxiv_id}:hepml-v1")
        entry.title(paper.title)
        entry.link(href=paper.link)
        entry.published(paper.published)
        entry.updated(max(paper.updated, record.processed_at))
        if paper.authors:
            entry.author({"name": ", ".join(paper.authors[:12])})
        for category in sorted(
            set([*paper.categories, *record.screening.hep_tasks])
        ):
            entry.category(term=category)
        body = record_html(record)
        entry.summary(record.screening.reason)
        entry.content(body, type="html")
    return feed


def _build_index(
    records: list[Record], site_url: str, feed_title: str
) -> str:
    cards = []
    for record in records:
        paper = record.paper
        authors = ", ".join(paper.authors[:8]) or "作者信息未提供"
        cards.append(
            "<article>"
            f"<h2><a href=\"{html.escape(paper.link, quote=True)}\">"
            f"{html.escape(paper.title)}</a></h2>"
            f"<p class=\"meta\">{html.escape(authors)} · "
            f"arXiv:{html.escape(paper.arxiv_id)} · "
            f"相关性 {record.screening.relevance:.2f}</p>"
            f"{record_html(record)}"
            "</article>"
        )
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{html.escape(feed_title)}</title>
  <link rel="alternate" type="application/atom+xml" href="{site_url}/atom.xml">
  <link rel="alternate" type="application/rss+xml" href="{site_url}/rss.xml">
  <style>
    :root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
    body {{ max-width: 920px; margin: 0 auto; padding: 2rem 1rem; line-height: 1.65; }}
    header {{ border-bottom: 1px solid #8885; margin-bottom: 2rem; }}
    article {{ border: 1px solid #8885; border-radius: 12px; padding: 1rem 1.25rem; margin: 1rem 0; }}
    h1, h2 {{ line-height: 1.3; }}
    h2 {{ font-size: 1.2rem; }}
    .meta {{ opacity: .72; font-size: .9rem; }}
    a {{ color: #2676c7; }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(feed_title)}</h1>
    <p>统计与机器学习方法对高能实验物理的每日启发。</p>
    <p><a href="{site_url}/atom.xml">Atom</a> · <a href="{site_url}/rss.xml">RSS</a></p>
  </header>
  <main>{''.join(cards) if cards else '<p>尚无可发布条目。</p>'}</main>
  <footer><p>最后生成：{generated}</p></footer>
</body>
</html>
"""


def publish(
    state: State,
    output_dir: Path,
    site_url: str,
    feed_title: str,
    publish_threshold: float,
    max_items: int,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = _eligible_records(state, publish_threshold, max_items)
    feed = _build_feed(records, site_url, feed_title)
    atom_payload = feed.atom_str(pretty=True)
    rss_payload = feed.rss_str(pretty=True)
    index_payload = _build_index(
        records, site_url, feed_title
    ).encode("utf-8")
    _atomic_write(output_dir / "atom.xml", atom_payload)
    _atomic_write(output_dir / "rss.xml", rss_payload)
    _atomic_write(
        output_dir / "index.html",
        index_payload,
    )
    return len(records)
