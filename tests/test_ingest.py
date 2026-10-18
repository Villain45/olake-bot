import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ingest import build_index, crawl
from ingest.chunk import chunk_markdown, embedding_text, strip_mdx


class ChunkTests(unittest.TestCase):
    def test_cleanup_preserves_shell_and_python_examples(self):
        source = "---\ntitle: Test\n---\nimport Tabs from '@theme/Tabs';\n```bash\nexport TOKEN=value\nimport example\n```"
        cleaned = strip_mdx(source)
        self.assertNotIn("@theme", cleaned)
        self.assertIn("export TOKEN=value", cleaned)
        self.assertIn("import example", cleaned)

    def test_oversized_code_is_bounded_and_fences_are_preserved(self):
        source = "# Setup\n```yaml\n" + "  key: value and more configuration\n" * 50 + "final: LAST_SENTINEL\n```"
        chunks = chunk_markdown(source, "https://olake.io/docs/setup", "Setup", "docs", max_tokens=150)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(embedding_text(chunk).encode()) + 2, 150)
            self.assertTrue(chunk["text"].startswith("```yaml\n"))
            self.assertTrue(chunk["text"].endswith("\n```"))
            self.assertEqual(chunk["anchor"], "setup")
        self.assertIn("LAST_SENTINEL", chunks[-1]["text"])

    def test_long_single_lines_and_unicode_fit_with_metadata(self):
        chunks = chunk_markdown("# " + "Heading " * 80 + "\n" + "漢字" * 600,
                                "https://olake.io/docs", "An unusually long title " * 30,
                                "docs", max_tokens=180)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(embedding_text(chunk).encode()) + 2 <= 180 for chunk in chunks))
        self.assertEqual("".join(chunk["text"] for chunk in chunks), "漢字" * 600)

    def test_supplied_token_counter_controls_budget(self):
        count = lambda value: len(value.split()) + 2
        chunks = chunk_markdown("hello world " * 200, "u", "title", "docs",
                                max_tokens=50, token_counter=count)
        self.assertTrue(all(count(embedding_text(chunk)) <= 50 for chunk in chunks))

    def test_split_fenced_code_can_be_reassembled_without_losing_characters(self):
        code = "\n".join(f"export VAR_{number}=value_for_command_{number}" for number in range(100))
        chunks = chunk_markdown(f"```bash\n{code}\n```", "u", "title", "docs", max_tokens=160)
        parts = [chunk["text"].split("\n", 1)[1].rsplit("\n", 1)[0] for chunk in chunks]
        self.assertEqual("".join(parts), code)

    def test_nested_fence_and_heading_inside_code(self):
        chunks = chunk_markdown("````markdown\n# not a heading\n```bash\nexport A=B\n```\n````",
                                "u", "title", "docs")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["anchor"], "")
        self.assertIn("export A=B", chunks[0]["text"])


class Vec(list):
    def tolist(self):
        return list(self)


class Dense:
    def __init__(self):
        self.calls = 0

    def embed(self, texts, **kwargs):
        self.calls += 1
        return [Vec([0.1, 0.2]) for _ in texts]


class Sparse(Dense):
    def embed(self, texts, **kwargs):
        self.calls += 1
        return [SimpleNamespace(indices=Vec([1]), values=Vec([1.0])) for _ in texts]


class IndexTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.settings = SimpleNamespace(collection="legacy", embed_model="dense", sparse_model="sparse",
                                        embed_dim=2, index_batch_size=1,
                                        index_cache_path=str(Path(self.tmp.name) / "cache.sqlite3"))
        self.chunks = [{"text": "a valid documentation passage", "breadcrumb": "Documentation",
                        "product": "ingestion", "url": "https://olake.io/docs", "title": "Docs"}]

    def client(self, count=1):
        client = MagicMock()
        client.get_aliases.return_value = SimpleNamespace(aliases=[SimpleNamespace(alias_name="legacy_current", collection_name="old")])
        client.collection_exists.return_value = False
        client.count.return_value = SimpleNamespace(count=count)
        return client

    def test_publish_uses_validated_staging_and_atomic_alias_without_deleting_old_collection(self):
        client = self.client()
        staging = build_index.publish_index(client, self.chunks, Dense(), Sparse(), self.settings)
        self.assertTrue(staging.startswith("legacy_build_"))
        client.delete_collection.assert_not_called()
        client.count.assert_called_once_with(staging, exact=True)
        operations = client.update_collection_aliases.call_args.kwargs["change_aliases_operations"]
        self.assertEqual(len(operations), 2)
        self.assertEqual(operations[1].create_alias.alias_name, "legacy_current")
        self.assertEqual(operations[1].create_alias.collection_name, staging)
        self.assertTrue(client.upsert.call_args.kwargs["wait"])

    def test_count_mismatch_never_publishes(self):
        client = self.client(count=0)
        with self.assertRaises(RuntimeError):
            build_index.publish_index(client, self.chunks, Dense(), Sparse(), self.settings)
        client.update_collection_aliases.assert_not_called()
        client.delete_collection.assert_not_called()

    def test_failed_upload_can_resume_without_reembedding(self):
        dense, sparse, client = Dense(), Sparse(), self.client()
        client.upsert.side_effect = RuntimeError("network interrupted")
        with patch.object(build_index.time, "sleep"):
            with self.assertRaises(RuntimeError):
                build_index.publish_index(client, self.chunks, dense, sparse, self.settings)
        self.assertEqual(client.upsert.call_count, 3)
        client.update_collection_aliases.assert_not_called()
        retry = self.client()
        build_index.publish_index(retry, self.chunks, dense, sparse, self.settings)
        self.assertEqual(dense.calls, 1)
        self.assertEqual(sparse.calls, 1)

    def test_empty_input_never_touches_collection(self):
        client = self.client()
        with self.assertRaises(ValueError):
            build_index.publish_index(client, [], Dense(), Sparse(), self.settings)
        client.create_collection.assert_not_called()

    def test_connection_failure_happens_before_loading_models(self):
        with patch.object(build_index, "make_client") as make_client, patch.object(build_index, "TextEmbedding") as embed:
            make_client.return_value.get_collections.side_effect = RuntimeError("offline")
            with self.assertRaises(RuntimeError):
                build_index.main()
            embed.assert_not_called()
            make_client.return_value.close.assert_called_once()

    def test_concurrent_local_build_is_rejected(self):
        path = Path(self.tmp.name) / "builder.lock"
        with build_index.build_lock(path):
            with self.assertRaises(RuntimeError):
                with build_index.build_lock(path):
                    self.fail("second builder acquired the same lock")

    def test_real_in_memory_qdrant_preserves_legacy_and_can_resume_staging(self):
        from qdrant_client import QdrantClient, models
        client = QdrantClient(":memory:")
        self.addCleanup(client.close)
        client.create_collection("legacy", vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE))
        with self.assertWarns(UserWarning):
            staging = build_index.publish_index(client, self.chunks, Dense(), Sparse(), self.settings)
        self.assertTrue(client.collection_exists("legacy"))
        self.assertEqual(client.count("legacy").count, 0)
        self.assertEqual(client.count("legacy_current").count, 1)
        points, _ = client.scroll("legacy_current", limit=1, with_payload=True)
        self.assertEqual(points[0].payload["embed_model"], self.settings.embed_model)
        self.assertEqual(points[0].payload["sparse_model"], self.settings.sparse_model)
        self.assertEqual(points[0].payload["embed_dim"], self.settings.embed_dim)
        dense, sparse = Dense(), Sparse()
        self.assertEqual(build_index.publish_index(client, self.chunks, dense, sparse, self.settings), staging)
        self.assertEqual(dense.calls, 0)
        aliases = {item.alias_name: item.collection_name for item in client.get_aliases().aliases}
        self.assertEqual(aliases["legacy_current"], staging)
        client.update_collection_aliases(change_aliases_operations=[
            models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name="legacy_current")),
            models.CreateAliasOperation(create_alias=models.CreateAlias(alias_name="legacy_current", collection_name="legacy")),
        ])
        # A prior staging collection is schema-checked and reused on a retry.
        self.assertEqual(build_index.publish_index(client, self.chunks, dense, sparse, self.settings), staging)
        self.assertEqual(dense.calls, 0)
        self.assertEqual(client.count("legacy_current").count, 1)

    def test_incompatible_staging_never_upserts_or_publishes(self):
        client = self.client()
        client.collection_exists.return_value = True
        client.get_collection.return_value = SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors={}, sparse_vectors={})))
        with self.assertRaises(RuntimeError):
            build_index.publish_index(client, self.chunks, Dense(), Sparse(), self.settings)
        client.upsert.assert_not_called()
        client.update_collection_aliases.assert_not_called()

    def test_dedup_keeps_product_and_indentation_distinctions(self):
        chunks = [{"product": "ingestion", "text": "  value: true"},
                  {"product": "ingestion", "text": "  value: true"},
                  {"product": "fusion", "text": "  value: true"},
                  {"product": "ingestion", "text": "value:\n  nested: true"},
                  {"product": "ingestion", "text": "value:\nnested: true"}]
        self.assertEqual(len(build_index.deduplicate(chunks)), 4)

    def test_fusion_path_and_selected_configs(self):
        self.assertEqual(build_index.product_of("olake-docs/docs/fusion/quickstart.md"), "fusion")
        self.assertTrue(build_index.selected_config(Path("values.yaml"), "charts/values.yaml"))
        self.assertTrue(build_index.selected_config(Path("sample.json"), "examples/sample.json"))
        self.assertFalse(build_index.selected_config(Path("package.json"), "package.json"))

    def test_repo_payload_has_commit_path_classification_and_config_content(self):
        root = Path(self.tmp.name) / "repos"
        repo = root / "olake-docs"
        (repo / "fusion").mkdir(parents=True)
        (repo / "fusion" / "config.yaml").write_text("database:\n  connection_timeout: 300\n  retry_count: 10\n", encoding="utf-8")
        with patch.object(build_index, "REPO_DIR", root), patch.object(build_index.subprocess, "run", return_value=SimpleNamespace(stdout="abc123\n")):
            chunks = build_index.repo_chunks()
        self.assertEqual(chunks[0]["product"], "fusion")
        self.assertEqual(chunks[0]["source_commit"], "abc123")
        self.assertIn("/blob/abc123/", chunks[0]["url"])
        self.assertIn("connection_timeout", chunks[0]["text"])


class CrawlTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.target = Path(self.tmp.name) / "docs.jsonl"

    def test_failed_page_preserves_previous_copy(self):
        old = {"url": "https://olake.io/docs/a", "text": "old", "title": "a"}
        fresh = {"url": "https://olake.io/docs/b", "text": "new", "title": "b"}
        self.target.write_text(json.dumps(old) + "\n", encoding="utf-8")
        crawl.write_crawl([old["url"], fresh["url"]], [None, fresh], self.target)
        records = [json.loads(line) for line in self.target.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(records, [old, fresh])
        self.assertEqual(list(self.target.parent.glob("*.tmp")), [])

    def test_incomplete_and_empty_crawl_leave_previous_bytes_unchanged(self):
        old = json.dumps({"url": "old", "text": "old", "title": "old"}) + "\n"
        self.target.write_text(old, encoding="utf-8")
        for urls, records in [([], []), (["a", "b"], [None, None])]:
            with self.assertRaises(RuntimeError):
                crawl.write_crawl(urls, records, self.target)
            self.assertEqual(self.target.read_text(encoding="utf-8"), old)

    def test_sitemap_http_error_is_reported(self):
        with patch.object(crawl.httpx, "get") as get:
            get.return_value.raise_for_status.side_effect = RuntimeError("HTTP error")
            with self.assertRaises(RuntimeError):
                crawl.sitemap_urls("https://olake.io/sitemap.xml")


if __name__ == "__main__":
    unittest.main()
