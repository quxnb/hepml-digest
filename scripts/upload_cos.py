"""Upload a static site to COS, then check public RSS accessibility."""

from __future__ import annotations

import argparse
import mimetypes
import os
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


def _content_type(path: Path) -> str:
    if path.name == "atom.xml":
        return "application/atom+xml; charset=utf-8"
    if path.suffix == ".xml":
        return "application/rss+xml; charset=utf-8"

    guessed, _ = mimetypes.guess_type(path.name)
    content_type = guessed or "application/octet-stream"
    if content_type.startswith("text/") or path.suffix in {".js", ".json"}:
        content_type += "; charset=utf-8"
    return content_type


def _verify_feed(public_site_url: str) -> None:
    url = f"{public_site_url.rstrip('/')}/rss.xml?verify={time.time_ns()}"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "hepml-digest-cos-deploy/1.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = response.read()
    if ET.fromstring(payload).tag != "rss":
        raise RuntimeError(f"Public URL did not return an RSS document: {url}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload static output to Tencent COS")
    parser.add_argument("source", type=Path)
    args = parser.parse_args()

    source = args.source.resolve()
    if not (source / "rss.xml").is_file():
        raise SystemExit(f"RSS file does not exist: {source / 'rss.xml'}")

    secret_id = _required_env("TENCENT_CLOUD_SECRET_ID")
    secret_key = _required_env("TENCENT_CLOUD_SECRET_KEY")
    bucket = _required_env("COS_BUCKET")
    region = _required_env("COS_REGION")
    public_site_url = _required_env("PUBLIC_SITE_URL")
    prefix = os.getenv("COS_PREFIX", "").strip("/")

    from qcloud_cos import CosConfig, CosS3Client  # type: ignore[import-not-found]

    client = CosS3Client(
        CosConfig(
            Region=region,
            SecretId=secret_id,
            SecretKey=secret_key,
            Scheme="https",
        )
    )

    files = sorted(path for path in source.rglob("*") if path.is_file())
    for path in files:
        relative_key = path.relative_to(source).as_posix()
        key = f"{prefix}/{relative_key}" if prefix else relative_key
        cache_control = (
            "public, max-age=300"
            if path.suffix in {".html", ".xml"}
            else "public, max-age=86400"
        )
        with path.open("rb") as body:
            client.put_object(
                Bucket=bucket,
                Key=key,
                Body=body,
                ContentType=_content_type(path),
                CacheControl=cache_control,
            )
        print(f"Uploaded {key}")

    _verify_feed(public_site_url)
    print(f"Public RSS is accessible: {public_site_url.rstrip('/')}/rss.xml")


if __name__ == "__main__":
    main()
