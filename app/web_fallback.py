import asyncio
import logging
import math
import threading
import time
from urllib.parse import urlsplit

import httpx
import trafilatura
from ddgs import DDGS

from .config import Settings
from .models import Chunk

log = logging.getLogger("olake.web")

# Cancellation cannot kill a Python thread. Keep its slot occupied until DDGS
# actually exits, so repeated timed-out requests cannot accumulate workers.
_SEARCH_GATES: dict[int, threading.BoundedSemaphore] = {}
_EXTRACT_GATES: dict[int, threading.BoundedSemaphore] = {}
_MAX_HTML_BYTES = 512_000


def _release_extractor(worker: asyncio.Task, gate: threading.BoundedSemaphore) -> None:
    gate.release()
    # A caller can already have timed out. Consume any later thread exception so
    # it does not become an unhandled task warning after cancellation.
    if not worker.cancelled():
        worker.exception()


def _allowed_url(url: str, trusted_domains: list[str]) -> bool:
    try:
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password:
            return False
        if parsed.port not in (None, 80, 443):
            return False
        host = (parsed.hostname or "").rstrip(".").encode("idna").decode("ascii").lower()
        for domain in trusted_domains:
            domain = domain.strip().rstrip(".").encode("idna").decode("ascii").lower()
            if domain and (host == domain or host.endswith("." + domain)):
                return True
    except (AttributeError, TypeError, ValueError, UnicodeError):
        pass
    return False


def _search_sync(
    query: str,
    s: Settings,
    cancelled: threading.Event,
    deadline: float,
) -> list[dict]:
    try:
        if cancelled.is_set() or time.monotonic() >= deadline:
            return []
        sites = " OR ".join(f"site:{d}" for d in s.trusted_domains)
        # Search engines do not enforce site: perfectly; check every result too.
        results = DDGS(timeout=max(1, math.ceil(deadline - time.monotonic()))).text(
            f"{query} ({sites})", max_results=4
        )
        if cancelled.is_set() or time.monotonic() >= deadline:
            return []
        return list(results or [])[:4]
    except Exception as exc:  # noqa: BLE001
        log.warning("web search failed: %s", exc)
        return []


async def _fetch_allowed(
    client: httpx.AsyncClient, url: str, s: Settings
) -> tuple[str, str] | None:
    """Follow a few redirects, validating the destination before every request."""
    for _ in range(5):
        if not _allowed_url(url, s.trusted_domains):
            return None
        async with client.stream("GET", url) as response:
            if not _allowed_url(str(response.url), s.trusted_domains):
                return None
            if response.is_redirect:
                location = response.headers.get("location")
                if not location:
                    return None
                url = str(response.url.join(location))
                continue
            response.raise_for_status()
            parts = []
            size = 0
            async for part in response.aiter_bytes():
                size += len(part)
                if size > _MAX_HTML_BYTES:
                    return None
                parts.append(part)
            html = b"".join(parts).decode(response.encoding or "utf-8", errors="replace")
            # Extraction can be CPU-heavy; keep it off the event loop and bound
            # workers even when a timed-out caller cannot stop its Python thread.
            slots = max(1, int(getattr(s, "web_concurrency", 2)))
            gate = _EXTRACT_GATES.setdefault(slots, threading.BoundedSemaphore(slots))
            if not gate.acquire(blocking=False):
                return None
            worker = asyncio.create_task(asyncio.to_thread(trafilatura.extract, html))
            try:
                extracted = await asyncio.shield(worker) or ""
            finally:
                worker.add_done_callback(lambda completed: _release_extractor(completed, gate))
            return str(response.url), extracted[:2500]
    return None


async def search(query: str, s: Settings) -> list[Chunk]:
    """Allowlisted external evidence, bounded by one deadline for all operations."""
    if not s.trusted_domains or s.web_timeout_s <= 0:
        return []
    slots = max(1, int(getattr(s, "web_concurrency", 2)))
    gate = _SEARCH_GATES.setdefault(slots, threading.BoundedSemaphore(slots))
    if not gate.acquire(blocking=False):
        return []
    cancelled = threading.Event()
    deadline = time.monotonic() + s.web_timeout_s
    worker = asyncio.create_task(asyncio.to_thread(_search_sync, query, s, cancelled, deadline))
    chunks: list[Chunk] = []
    try:
        async with asyncio.timeout(s.web_timeout_s):
            results = await asyncio.shield(worker)
            seen_urls: set[str] = set()
            async with httpx.AsyncClient(timeout=s.web_timeout_s, follow_redirects=False) as client:
                for result in results:
                    url = result.get("href", "")
                    if not _allowed_url(url, s.trusted_domains) or url in seen_urls:
                        continue
                    try:
                        fetched = await _fetch_allowed(client, url, s)
                    except (httpx.HTTPError, ValueError) as exc:
                        log.warning("external fetch failed %s: %s", url, exc)
                        continue
                    if not fetched:
                        continue
                    final_url, text = fetched
                    if not text.strip() or final_url in seen_urls:
                        continue
                    seen_urls.add(final_url)
                    chunks.append(
                        Chunk(id=0, url=final_url, title=str(result.get("title") or final_url),
                              breadcrumb="external", text=text, source_type="external")
                    )
                    if len(chunks) == 2:
                        break
            return chunks
    except Exception as exc:  # noqa: BLE001
        log.warning("web fallback ended (%s); retaining %d completed passages", type(exc).__name__, len(chunks))
        return chunks
    finally:
        cancelled.set()
        # Bound the whole fallback, including async fetches. If DDGS is still
        # running after cancellation, its slot remains held until it exits.
        worker.add_done_callback(lambda _: gate.release())
