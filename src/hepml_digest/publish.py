from __future__ import annotations

import html
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from xml.etree import ElementTree

from feedgen.feed import FeedGenerator

from .models import DigestTrack, Record, State


TRACK_LABELS: dict[DigestTrack, str] = {
    "method_radar": "方法雷达",
    "hep_application": "HEP 直接应用",
}

XML_INVALID_CHARACTERS = re.compile(
    "[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]"
)


def _xml_text(value: str) -> str:
    return XML_INVALID_CHARACTERS.sub("", value)


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


def _feedback_url(record: Record, repository: str) -> str:
    paper = record.paper
    track = TRACK_LABELS[paper.digest_track]
    body = f"""## 论文反馈

- arXiv：[{paper.arxiv_id}]({paper.link})
- 栏目：{track}
- 当前相关性：{record.screening.relevance:.2f}
- 当前证据等级：{record.screening.evidence_level}

## 判断

- [ ] 推荐保留
- [ ] HEP 映射合理
- [ ] HEP 映射过度推测
- [ ] 应移至另一个栏目
- [ ] 建议精读全文

## 补充说明

"""
    query = urlencode(
        {
            "title": f"[论文反馈] {paper.arxiv_id} {paper.title[:80]}",
            "body": body,
        }
    )
    return f"https://github.com/{repository}/issues/new?{query}"


def record_html(record: Record, feedback_repository: str = "") -> str:
    paper = record.paper
    screen = record.screening
    parts = [
        f"<p><strong>栏目：</strong>{TRACK_LABELS[paper.digest_track]}</p>",
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
    links = [
        f'<a href="{html.escape(paper.link, quote=True)}">arXiv 页面</a>'
    ]
    if feedback_repository:
        feedback_url = _feedback_url(record, feedback_repository)
        links.append(
            f'<a href="{html.escape(feedback_url, quote=True)}">提交反馈</a>'
        )
    parts.append(
        f"<p>{' · '.join(links)} · "
        f"{html.escape(', '.join(paper.categories))}</p>"
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
    records: list[Record],
    site_url: str,
    feed_title: str,
    self_url: str,
    feedback_repository: str,
) -> FeedGenerator:
    now = datetime.now(timezone.utc)
    feed = FeedGenerator()
    feed.id(_xml_text(site_url))
    feed.title(_xml_text(feed_title))
    feed.subtitle("统计与机器学习方法对高能实验物理的每日启发")
    feed.author({"name": "HEP-ML Digest Bot"})
    feed.link(href=_xml_text(site_url), rel="alternate")
    feed.link(href=_xml_text(self_url), rel="self")
    feed.language("zh-CN")
    feed.updated(now)

    for record in reversed(records):
        paper = record.paper
        entry = feed.add_entry()
        entry.id(_xml_text(f"arxiv:{paper.arxiv_id}:hepml-v1"))
        entry.title(_xml_text(paper.title))
        entry.link(href=_xml_text(paper.link))
        entry.published(paper.published)
        entry.updated(max(paper.updated, record.processed_at))
        if paper.authors:
            entry.author(
                {"name": _xml_text(", ".join(paper.authors[:12]))}
            )
        for category in sorted(
            set(
                [
                    *paper.categories,
                    *record.screening.hep_tasks,
                    f"track:{paper.digest_track}",
                ]
            )
        ):
            entry.category(term=_xml_text(category))
        body = record_html(record, feedback_repository)
        entry.summary(_xml_text(record.screening.reason))
        entry.content(_xml_text(body), type="html")
    return feed


def _build_index(
    records: list[Record],
    site_url: str,
    feed_title: str,
    feedback_repository: str,
) -> str:
    def cards_for(track: DigestTrack) -> str:
        cards = []
        for record in records:
            if record.paper.digest_track != track:
                continue
            paper = record.paper
            authors = ", ".join(paper.authors[:8]) or "作者信息未提供"
            cards.append(
                "<article>"
                f"<h3><a href=\"{html.escape(paper.link, quote=True)}\">"
                f"{html.escape(paper.title)}</a></h3>"
                f"<p class=\"meta\">{html.escape(authors)} · "
                f"arXiv:{html.escape(paper.arxiv_id)} · "
                f"相关性 {record.screening.relevance:.2f}</p>"
                f"{record_html(record, feedback_repository)}"
                "</article>"
            )
        return "".join(cards) or '<p class="empty">本栏目尚无可发布条目。</p>'

    method_cards = cards_for("method_radar")
    application_cards = cards_for("hep_application")
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
    h1, h2, h3 {{ line-height: 1.3; }}
    h2 {{ margin-top: 2.4rem; }}
    h3 {{ font-size: 1.2rem; }}
    .meta {{ opacity: .72; font-size: .9rem; }}
    .empty {{ opacity: .72; }}
    a {{ color: #2676c7; }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(feed_title)}</h1>
    <p>统计与机器学习方法对高能实验物理的每日启发。</p>
    <p><a href="{site_url}/atom.xml">综合 Atom</a> ·
       <a href="{site_url}/rss.xml">综合 RSS</a> ·
       <a href="{site_url}/methods.xml">方法雷达 RSS</a> ·
       <a href="{site_url}/hep-applications.xml">HEP 应用 RSS</a></p>
  </header>
  <main>
    <section id="method-radar"><h2>方法雷达</h2>{method_cards}</section>
    <section id="hep-applications"><h2>HEP 直接应用</h2>{application_cards}</section>
  </main>
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
    feedback_repository: str = "",
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = _eligible_records(state, publish_threshold, max_items)
    atom_feed = _build_feed(
        records,
        site_url,
        feed_title,
        f"{site_url}/atom.xml",
        feedback_repository,
    )
    rss_feed = _build_feed(
        records,
        site_url,
        feed_title,
        f"{site_url}/rss.xml",
        feedback_repository,
    )
    method_records = [
        record
        for record in records
        if record.paper.digest_track == "method_radar"
    ]
    application_records = [
        record
        for record in records
        if record.paper.digest_track == "hep_application"
    ]
    method_feed = _build_feed(
        method_records,
        site_url,
        f"{feed_title} · 方法雷达",
        f"{site_url}/methods.xml",
        feedback_repository,
    )
    application_feed = _build_feed(
        application_records,
        site_url,
        f"{feed_title} · HEP 直接应用",
        f"{site_url}/hep-applications.xml",
        feedback_repository,
    )
    atom_payload = atom_feed.atom_str(pretty=True)
    rss_root = ElementTree.fromstring(rss_feed.rss_str(pretty=True))
    channel_link = rss_root.find("./channel/link")
    if channel_link is not None:
        channel_link.text = site_url
    rss_payload = ElementTree.tostring(
        rss_root, encoding="utf-8", xml_declaration=True
    )
    index_payload = _build_index(
        records, site_url, feed_title, feedback_repository
    ).encode("utf-8")
    _atomic_write(output_dir / "atom.xml", atom_payload)
    _atomic_write(output_dir / "rss.xml", rss_payload)
    for name, generated_feed in (
        ("methods.xml", method_feed),
        ("hep-applications.xml", application_feed),
    ):
        root = ElementTree.fromstring(generated_feed.rss_str(pretty=True))
        channel_link = root.find("./channel/link")
        if channel_link is not None:
            channel_link.text = site_url
        _atomic_write(
            output_dir / name,
            ElementTree.tostring(
                root, encoding="utf-8", xml_declaration=True
            ),
        )
    _atomic_write(
        output_dir / "index.html",
        index_payload,
    )
    return len(records)
