"""Step 1: fetch content.
  python -m ingest.crawl docs     # crawl olake.io/docs via sitemap -> data/raw/docs.jsonl
  python -m ingest.crawl repos    # shallow-clone the OLake repos -> data/repos/
"""
import json
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import httpx
import trafilatura

SITEMAP = "https://olake.io/sitemap.xml"
REPOS = ["olake", "olake-ui", "olake-helm", "olake-fusion", "olake-docs"]
RAW = Path("data/raw")
REPO_DIR = Path("data/repos")


def sitemap_urls(url: str) -> list[str]:
    response = httpx.get(url, timeout=30, follow_redirects=True)
    response.raise_for_status()
    xml = response.text
    locs = re.findall(r"<loc>\s*(.*?)\s*</loc>", xml)
    out: list[str] = []
    for loc in locs:
        out.extend(sitemap_urls(loc) if loc.endswith(".xml") else [loc])
    return out


def fetch(url: str) -> dict | None:
    try:
        html = trafilatura.fetch_url(url)
        if not html:
            return None
        text = trafilatura.extract(
            html, output_format="markdown", include_links=True,
            include_tables=True, include_comments=False,
        )
        meta = trafilatura.extract_metadata(html)
        if not text:
            return None
        return {"url": url, "title": (meta.title if meta and meta.title else url), "text": text,
                "fetched_at": datetime.now(timezone.utc).isoformat()}
    except Exception as e:  # noqa: BLE001
        print("fail", url, e)
        return None


def write_crawl(urls: list[str], records: list[dict | None], destination: Path) -> None:
    """Publish a sufficiently complete crawl, retaining old pages on fetch errors."""
    if not urls or len(urls) != len(records):
        raise RuntimeError("Empty or incomplete crawl; previous data is unchanged")
    previous = {}
    if destination.exists():
        previous = {record["url"]: record for line in destination.read_text(encoding="utf-8").splitlines()
                    if line.strip() for record in [json.loads(line)]}
    merged, failed = [], []
    for url, record in zip(urls, records):
        if record is not None:
            merged.append(record)
        else:
            failed.append(url)
            if url in previous:
                merged.append(previous[url])
    if len(merged) / len(urls) < 0.95:
        raise RuntimeError(f"Only {len(merged)}/{len(urls)} pages available; previous data is unchanged")
    if previous and len(merged) < len(previous) * 0.9:
        raise RuntimeError("Crawl coverage dropped more than 10%; previous data is unchanged")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent,
                                         prefix="docs-", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            for record in merged:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary.replace(destination)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()
    print(f"Saved {len(merged)} pages; {len(failed)} fetch failures (previous copies retained where available)")


def crawl_docs() -> None:
    urls = sorted({u for u in sitemap_urls(SITEMAP) if "/docs" in u})
    print(f"{len(urls)} doc URLs")
    with ThreadPoolExecutor(8) as executor:
        records = list(executor.map(fetch, urls))
    write_crawl(urls, records, RAW / "docs.jsonl")


def clone_repos() -> None:
    REPO_DIR.mkdir(parents=True, exist_ok=True)
    for name in REPOS:
        dest = REPO_DIR / name
        if dest.exists():
            dirty = subprocess.run(["git", "-C", str(dest), "status", "--porcelain"],
                                   check=True, capture_output=True, text=True).stdout.strip()
            if dirty:
                raise RuntimeError(f"{dest} has local changes; refusing to update it")
            branch = subprocess.run(["git", "-C", str(dest), "symbolic-ref", "--short", "HEAD"],
                                    check=True, capture_output=True, text=True).stdout.strip()
            subprocess.run(["git", "-C", str(dest), "fetch", "origin", branch], check=True)
            subprocess.run(["git", "-C", str(dest), "merge", "--ff-only", "FETCH_HEAD"], check=True)
            continue
        subprocess.run(
            ["git", "clone", "--depth", "1", f"https://github.com/datazip-inc/{name}", str(dest)],
            check=True,
        )


if __name__ == "__main__":
    {"docs": crawl_docs, "repos": clone_repos}[sys.argv[1]]()
