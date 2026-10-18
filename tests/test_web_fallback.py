import asyncio
import threading
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from app.config import Settings
from app.web_fallback import _EXTRACT_GATES, _SEARCH_GATES, _allowed_url, _fetch_allowed, _search_sync, search


class AllowlistTests(unittest.TestCase):
    def test_only_real_allowed_hosts_and_subdomains_are_accepted(self):
        trusted = ["postgresql.org"]
        for url in ("https://postgresql.org/docs", "https://www.postgresql.org/docs", "https://POSTGRESQL.ORG/docs"):
            with self.subTest(url=url):
                self.assertTrue(_allowed_url(url, trusted))
        for url in (
            "https://postgresql.org.evil.test/docs", "https://fakepostgresql.org/docs",
            "https://postgresql.org@evil.test/docs", "https://evil.test/postgresql.org",
            "file://postgresql.org/path", "https://postgresql.org:999/docs", "not a url",
            "https://user:pass@postgresql.org/docs",
        ):
            with self.subTest(url=url):
                self.assertFalse(_allowed_url(url, trusted))

    def test_search_does_not_retry_unrestricted(self):
        settings = Settings(_env_file=None)
        fake = MagicMock()
        fake.text.return_value = []
        with patch("app.web_fallback.DDGS", return_value=fake):
            self.assertEqual(_search_sync("question", settings, threading.Event(), time.monotonic() + 1), [])
        fake.text.assert_called_once()
        self.assertIn("site:iceberg.apache.org", fake.text.call_args.args[0])


class WebFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_redirect_is_checked_before_second_request(self):
        visited = []

        def handler(request):
            visited.append(str(request.url))
            return httpx.Response(302, headers={"location": "https://evil.test/content"})

        settings = Settings(_env_file=None)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False) as client:
            result = await _fetch_allowed(client, "https://iceberg.apache.org/a", settings)
        self.assertIsNone(result)
        self.assertEqual(visited, ["https://iceberg.apache.org/a"])

    async def test_relative_trusted_redirect_keeps_final_citation(self):
        def handler(request):
            if request.url.path == "/a":
                return httpx.Response(302, headers={"location": "/b"})
            return httpx.Response(200, text="<html>contents</html>")

        settings = Settings(_env_file=None)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False) as client:
            with patch("app.web_fallback.trafilatura.extract", return_value="Relevant evidence"):
                result = await _fetch_allowed(client, "https://iceberg.apache.org/a", settings)
        self.assertEqual(result, ("https://iceberg.apache.org/b", "Relevant evidence"))

    async def test_search_rejects_untrusted_results(self):
        fake = MagicMock()
        fake.text.return_value = [{"href": "https://evil.test", "body": "fabricated"}]
        fetch = AsyncMock()
        with patch("app.web_fallback.DDGS", return_value=fake), patch("app.web_fallback._fetch_allowed", new=fetch):
            self.assertEqual(await search("question", Settings(_env_file=None)), [])
        fetch.assert_not_awaited()

    async def test_timeout_preserves_previously_fetched_passages(self):
        fake = MagicMock()
        fake.text.return_value = [
            {"href": "https://iceberg.apache.org/first", "title": "First source"},
            {"href": "https://iceberg.apache.org/slow", "title": "Slow source"},
        ]

        async def fetch(client, url, settings):
            if url.endswith("/slow"):
                await asyncio.sleep(1)
            return url, "Completed evidence"

        settings = Settings(_env_file=None, web_timeout_s=0.15)
        with patch("app.web_fallback.DDGS", return_value=fake), patch("app.web_fallback._fetch_allowed", side_effect=fetch):
            chunks = await search("question", settings)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].url, "https://iceberg.apache.org/first")
        self.assertEqual(chunks[0].text, "Completed evidence")

    async def test_extraction_runs_off_event_loop_and_holds_slot_after_timeout(self):
        started, release = threading.Event(), threading.Event()
        loop_thread = threading.get_ident()
        extract_threads = []

        def blocked_extract(html):
            extract_threads.append(threading.get_ident())
            started.set()
            release.wait(timeout=3)
            return "Evidence"

        settings = Settings(_env_file=None, web_concurrency=1)
        _EXTRACT_GATES.pop(1, None)
        transport = httpx.MockTransport(lambda request: httpx.Response(200, text="<html>contents</html>"))
        try:
            async with httpx.AsyncClient(transport=transport) as client:
                with patch("app.web_fallback.trafilatura.extract", side_effect=blocked_extract) as extract:
                    with self.assertRaises(asyncio.TimeoutError):
                        async with asyncio.timeout(0.06):
                            await _fetch_allowed(client, "https://iceberg.apache.org/a", settings)
                    self.assertTrue(started.is_set())
                    self.assertNotEqual(extract_threads[0], loop_thread)
                    self.assertIsNone(await _fetch_allowed(client, "https://iceberg.apache.org/b", settings))
                    extract.assert_called_once()
                    release.set()
                    for _ in range(100):
                        gate = _EXTRACT_GATES[1]
                        if gate.acquire(blocking=False):
                            gate.release()
                            break
                        await asyncio.sleep(0.005)
                    else:
                        self.fail("extractor slot was not released after the worker exited")
        finally:
            release.set()

    async def test_timeout_does_not_release_worker_slot_early(self):
        started, release = threading.Event(), threading.Event()

        def blocked_search(*args, **kwargs):
            started.set()
            release.wait(timeout=3)
            return []

        settings = Settings(_env_file=None, web_timeout_s=0.15, web_concurrency=1)
        _SEARCH_GATES.pop(1, None)
        fake = MagicMock()
        fake.text.side_effect = blocked_search
        try:
            with patch("app.web_fallback.DDGS", return_value=fake):
                start = time.monotonic()
                self.assertEqual(await search("question", settings), [])
                self.assertLess(time.monotonic() - start, 0.8)
                self.assertTrue(started.is_set())
                self.assertEqual(await search("question again", settings), [])
                fake.text.assert_called_once()
                release.set()
                for _ in range(100):
                    gate = _SEARCH_GATES[1]
                    if gate.acquire(blocking=False):
                        gate.release()
                        break
                    await asyncio.sleep(0.005)
                else:
                    self.fail("worker slot was not released after the worker exited")
        finally:
            release.set()
