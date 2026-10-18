import asyncio
import threading
import unittest
from contextvars import ContextVar
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers

from app.config import Settings
from app.models import Chunk
from app.retrieval import QUERY_TOKEN_BUDGET, Retriever, canonical_page, retrieval_query


def chunk(text, url, score=1.0):
    return Chunk(id=0, text=text, url=url, title="Title", breadcrumb="Title", source_type="docs", score=score)


class RetrievalTests(unittest.TestCase):
    def make_retriever(self):
        retriever = Retriever.__new__(Retriever)
        retriever.s = Settings(_env_file=None)
        retriever.qdrant = Mock()
        retriever._index_lock = threading.Lock()
        retriever._index_expires = 0.0
        retriever._index_name = retriever.s.collection
        return retriever

    def test_alias_resolution_and_legacy_compatibility(self):
        retriever = self.make_retriever()
        retriever.qdrant.get_aliases.return_value.aliases = []
        self.assertEqual(retriever.index_version(), "olake_docs")
        retriever.qdrant.get_aliases.return_value.aliases = [
            SimpleNamespace(alias_name="olake_docs_current", collection_name="new-snapshot")
        ]
        self.assertEqual(retriever.index_version(refresh=True), "new-snapshot")

    def test_duplicate_text_and_source_diversity(self):
        chunks = [
            chunk("same", "https://one/#a"), chunk("same", "https://two/#a"),
            chunk("second", "https://one/#b"), chunk("third", "https://one/#c"),
            chunk("other", "https://two/#b"),
        ]
        selected = Retriever._diverse(chunks, 6)
        self.assertEqual([item.text for item in selected], ["same", "second", "other"])

    def test_configuration_keeps_four_relevant_sections_from_one_page(self):
        chunks = [chunk(f"step {index}", f"https://olake.io/docs/postgres/#step-{index}", score)
                  for index, score in enumerate([0.95, 0.9, 0.8, 0.7, 0.6])]
        chunks.append(chunk("related page", "https://olake.io/docs/permissions/", 0.59))
        selected = Retriever._diverse(chunks, 6, "How do I configure PostgreSQL CDC?")
        self.assertEqual([item.text for item in selected], ["step 0", "step 1", "step 2", "step 3", "related page"])
        self.assertEqual([item.score for item in selected[:4]], [0.95, 0.9, 0.8, 0.7])

    def test_configuration_does_not_fill_context_with_weak_same_page_sections(self):
        chunks = [chunk(f"section {index}", "https://olake.io/docs/postgres/", score)
                  for index, score in enumerate([0.95, 0.8, 0.2, 0.1])]
        selected = Retriever._diverse(chunks, 6, "PostgreSQL configuration")
        self.assertEqual([item.text for item in selected], ["section 0", "section 1"])

    def test_authority_only_breaks_close_relevance_ties(self):
        blog = chunk("blog explanation", "https://olake.io/blog/postgres/", 0.52)
        docs = chunk("reference commands", "https://olake.io/docs/postgres/", 0.50)
        selected = Retriever._diverse([blog, docs], 1, "Configure PostgreSQL")
        self.assertEqual(selected, [docs])
        self.assertEqual(docs.score, 0.50)
        strongest = replace(blog, score=0.9)
        self.assertEqual(Retriever._diverse([strongest, docs], 1, "Configure PostgreSQL"), [strongest])

    def test_authority_does_not_replace_supported_evidence_with_below_threshold_docs(self):
        blog = chunk("supported statement", "https://olake.io/blog/postgres/", 0.26)
        docs = chunk("weakly related reference", "https://olake.io/docs/postgres/", 0.24)
        self.assertEqual(Retriever._diverse([blog, docs], 1, "Configure PostgreSQL", min_score=0.25), [blog])

    def test_comparison_content_keeps_relevance_when_requested(self):
        comparison = chunk("product comparison", "https://olake.io/comparisons/olake-vs-other/", 0.8)
        docs = chunk("reference", "https://olake.io/docs/overview/", 0.75)
        self.assertEqual(Retriever._diverse([comparison, docs], 1, "Compare OLake versus another product"), [comparison])

    def test_agent_instruction_files_are_never_answer_evidence(self):
        candidates = [chunk("agent instructions", "https://github.com/datazip-inc/olake/blob/main/CLAUDE.md", 1.0),
                      chunk("more instructions", "https://github.com/datazip-inc/olake/blob/main/AGENTS.md", 0.99),
                      chunk("setup instructions", "https://olake.io/docs/install/", 0.9)]
        self.assertEqual([item.text for item in Retriever._diverse(candidates, 6)], ["setup instructions"])

    def test_canonical_docs_mirrors_share_page_limit(self):
        urls = ["https://olake.io/docs/connectors/postgres/#one",
                "https://github.com/datazip-inc/olake-docs/blob/ab123/docs/connectors/postgres/index.mdx#two",
                "https://github.com/datazip-inc/olake-docs/blob/main/docs/connectors/postgres/index.md#three"]
        self.assertEqual(len({canonical_page(url) for url in urls}), 1)
        selected = Retriever._diverse([chunk(f"section {i}", url) for i, url in enumerate(urls)], 6)
        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[0].url, urls[0])

    def test_canonical_urls_do_not_merge_other_repositories_or_different_pages(self):
        self.assertNotEqual(canonical_page("https://github.com/datazip-inc/olake/blob/main/docs/setup.md"),
                            canonical_page("https://olake.io/docs/setup/"))
        self.assertNotEqual(canonical_page("https://olake.io/docs/postgres/setup/"),
                            canonical_page("https://olake.io/docs/postgres/"))

    def test_near_identical_mirrors_do_not_repeat_context(self):
        text = "PostgreSQL replication configuration requires a replication slot and publication. " * 4
        site = chunk(text, "https://olake.io/docs/connectors/postgres/", 0.85)
        mirror = replace(site, text="**PostgreSQL** " + text[len("PostgreSQL "):],
                         url="https://github.com/datazip-inc/olake-docs/blob/123/docs/connectors/postgres/index.mdx",
                         source_type="repo")
        self.assertEqual(Retriever._diverse([site, mirror], 6), [site])

    def test_case_and_formatting_only_prose_mirrors_do_not_repeat_context(self):
        text = "PostgreSQL replication requires a publication for each replicated table and a logical replication slot before starting and validating your first ingestion pipeline."
        site = chunk(text, "https://olake.io/docs/connectors/postgres/", 0.85)
        mirror = replace(site, text="**POSTGRESQL** " + text[len("PostgreSQL "):].upper(),
                         url="https://github.com/datazip-inc/olake-docs/blob/main/docs/connectors/postgres/index.mdx",
                         source_type="repo")
        self.assertEqual(Retriever._diverse([site, mirror], 6), [site])

    def test_mirror_dedup_preserves_changed_configuration_values_and_identifiers(self):
        prose = (
            "PostgreSQL replication requires a publication for each replicated table and a logical replication slot. "
            "Check these database settings before configuring the source and starting the first pipeline. "
        )
        differences = [
            ("```yaml\nrds.logical_replication: 0\n```", "```yaml\nrds.logical_replication: 1\n```"),
            ("```yaml\ndatabase: CaseSensitive\n```", "```yaml\ndatabase: casesensitive\n```"),
            ("```yaml\nroot:\n  key: value\n```", "```yaml\nroot:\nkey: value\n```"),
            ("Set `DatabaseName` for this source.", "Set `databasename` for this source."),
            ("Use --full-refresh for this pipeline.", "Use --FULL-REFRESH for this pipeline."),
            ("Set replication_slot_name for this source.", "Set Replication_Slot_Name for this source."),
            ("The supported setting value is -1.", "The supported setting value is 1."),
            ("The supported timeout is 30 seconds.", "The supported timeout is 60 seconds."),
            ("database: CaseSensitive", "database: casesensitive"),
        ]
        for original, changed in differences:
            with self.subTest(original=original, changed=changed):
                site = chunk(prose + original, "https://olake.io/docs/connectors/postgres/", 0.85)
                mirror = replace(site, text=prose + changed,
                                 url="https://github.com/datazip-inc/olake-docs/blob/main/docs/connectors/postgres/index.mdx",
                                 source_type="repo")
                self.assertEqual(len(Retriever._diverse([site, mirror], 6, "Configure PostgreSQL")), 2)

    def test_mirror_dedup_preserves_one_additional_instruction(self):
        prose = "PostgreSQL replication requires a publication for each replicated table and a logical replication slot before starting a pipeline. "
        site = chunk(prose, "https://olake.io/docs/connectors/postgres/", 0.85)
        mirror = replace(site, text=prose + "Restart PostgreSQL after changing the configuration.",
                         url="https://github.com/datazip-inc/olake-docs/blob/main/docs/connectors/postgres/index.mdx",
                         source_type="repo")
        self.assertEqual(len(Retriever._diverse([site, mirror], 6, "Configure PostgreSQL")), 2)

    def test_recovery_search_expands_candidates_without_writes_or_score_changes(self):
        retriever = self.make_retriever()
        retriever.s = Settings(_env_file=None, candidates=70, top_k=2)
        retriever._request_index = ContextVar("test_index", default="published-snapshot")
        retriever.dense, retriever.sparse, retriever.ranker = Mock(), Mock(), Mock()
        retriever.dense.query_embed.return_value = iter([np.array([1.0, 0.0, 0.0])])
        retriever.sparse.query_embed.return_value = iter([SimpleNamespace(indices=np.array([1]), values=np.array([1.0]))])
        payload = dict(url="https://olake.io/docs/connectors/postgres/", anchor="configuration", title="PostgreSQL",
                       breadcrumb="PostgreSQL > Configuration", text="A replication slot is required.", source_type="docs")
        retriever.qdrant.query_points.return_value.points = [SimpleNamespace(payload=payload)]
        retriever.ranker.rerank.return_value = [{"id": 0, "score": 0.72}]
        result = retriever._retrieve_sync("Postgres CDC setup", "ingestion", recovery=True)
        args, kwargs = retriever.qdrant.query_points.call_args
        self.assertEqual(args, ("published-snapshot",))
        self.assertEqual(kwargs["limit"], 100)
        self.assertEqual([prefetch.limit for prefetch in kwargs["prefetch"]], [100, 100])
        self.assertEqual(kwargs["prefetch"][0].filter.must[0].match.value, "ingestion")
        search_query = retriever.dense.query_embed.call_args.args[0]
        self.assertIn("PostgreSQL", search_query)
        self.assertNotEqual(search_query, "Postgres CDC setup")
        self.assertEqual(retriever.ranker.rerank.call_args.args[0].query, "Postgres CDC setup")
        self.assertEqual(result[0].score, 0.72)
        self.assertLessEqual(len(result), retriever.s.top_k)
        self.assertEqual([call[0] for call in retriever.qdrant.method_calls], ["query_points"])

    def test_recovery_prefers_new_supported_evidence_from_wider_candidates(self):
        retriever = self.make_retriever()
        retriever._request_index = ContextVar("test_index", default="published-snapshot")
        previous = chunk("The first retrieved section.", "https://olake.io/docs/postgres/", 0.9)
        retriever._request_evidence = ContextVar("test_evidence", default=(previous,))
        retriever.dense, retriever.sparse, retriever.ranker = Mock(), Mock(), Mock()
        retriever.dense.query_embed.return_value = iter([np.array([1.0, 0.0, 0.0])])
        retriever.sparse.query_embed.return_value = iter([SimpleNamespace(indices=np.array([1]), values=np.array([1.0]))])
        shared = dict(url=previous.url, anchor="", title="PostgreSQL", breadcrumb="PostgreSQL", source_type="docs")
        retriever.qdrant.query_points.return_value.points = [
            SimpleNamespace(payload={**shared, "text": previous.text}),
            SimpleNamespace(payload={**shared, "text": "New configuration prerequisites."}),
            SimpleNamespace(payload={**shared, "text": "Unrelated low scoring details."}),
        ]
        scores = {0: 0.9, 1: 0.7, 2: 0.1}
        retriever.ranker.rerank.side_effect = lambda request: sorted(
            [{**passage, "score": scores[passage["id"]]} for passage in request.passages],
            key=lambda passage: passage["score"], reverse=True,
        )
        result = retriever._retrieve_sync("PostgreSQL configuration", recovery=True)
        self.assertEqual([item.text for item in result], ["New configuration prerequisites."])
        self.assertEqual(result[0].score, 0.7)
        self.assertEqual([len(call.args[0].passages) for call in retriever.ranker.rerank.call_args_list], [2, 1])

    def test_readiness_rejects_empty_or_mismatched_index(self):
        retriever = self.make_retriever()
        retriever.qdrant.get_aliases.return_value.aliases = []
        params = SimpleNamespace(vectors={"dense": SimpleNamespace(size=384)}, sparse_vectors={"bm25": object()})
        retriever.qdrant.get_collection.return_value = SimpleNamespace(config=SimpleNamespace(params=params))
        retriever.qdrant.count.return_value.count = 3
        retriever.qdrant.scroll.return_value = ([], None)
        self.assertEqual(retriever.readiness()["indexed_chunks"], 3)
        retriever.qdrant.scroll.return_value = ([SimpleNamespace(payload={"embed_model": "different-model"})], None)
        with self.assertRaises(ValueError):
            retriever.readiness()
        retriever.qdrant.scroll.return_value = ([], None)
        retriever.qdrant.count.return_value.count = 0
        with self.assertRaises(ValueError):
            retriever.readiness()
        params.vectors["dense"].size = 100
        with self.assertRaises(ValueError):
            retriever.readiness()

    def test_external_rerank_returns_scored_copies(self):
        retriever = self.make_retriever()
        retriever.ranker = Mock()
        retriever.ranker.rerank.return_value = [{"id": 1, "score": 0.9}, {"id": 0, "score": 0.1}]
        original = [chunk("one", "https://one/", 0), chunk("two", "https://two/", 0)]
        ranked = retriever._rerank_external_sync("question", original)
        self.assertEqual([item.text for item in ranked], ["two", "one"])
        self.assertEqual(ranked[0].score, 0.9)
        self.assertEqual(original[1].score, 0)

    def test_external_rerank_merges_all_batches_with_original_source_ids(self):
        retriever = self.make_retriever()
        retriever.ranker = Mock()
        scores = {0: 0.3, 1: 0.8, 2: 0.5, 3: 0.9, 4: 0.7}
        retriever.ranker.rerank.side_effect = lambda request: sorted(
            [{**passage, "score": scores[passage["id"]]} for passage in request.passages],
            key=lambda passage: passage["score"], reverse=True,
        )
        original = [chunk(f"passage {i}", f"https://source{i}.test/", 0) for i in scores]
        ranked = retriever._rerank_external_sync("question", original)
        self.assertEqual([item.url for item in ranked], [original[i].url for i in (3, 1, 4, 2, 0)])
        self.assertEqual([len(call.args[0].passages) for call in retriever.ranker.rerank.call_args_list], [2, 2, 1])
        self.assertTrue(all(item.score == 0 for item in original))


class RetrievalQueryTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    def test_short_query_and_exact_case_sensitive_identifier_are_preserved(self):
        question = "How do I set replication_slot_name for PostgreSQL CDC?"
        self.assertEqual(retrieval_query(question, self.tokenizer), question)

    def test_long_pasted_config_preserves_question_error_and_identifier(self):
        question = "```yaml\n" + "\n".join(f"field_{index}: dummy" for index in range(300)) + (
            "\nreplication_slot_name: olake\nERROR: replication slot does not exist\n```\n"
            "Why does PostgreSQL CDC fail with replication_slot_name?"
        )
        query = retrieval_query(question, self.tokenizer)
        self.assertIn("Why does PostgreSQL CDC", query)
        self.assertIn("replication_slot_name", query)
        self.assertIn("replication slot does not exist", query)
        self.assertLessEqual(len(self.tokenizer.encode(query, add_special_tokens=False).ids), QUERY_TOKEN_BUDGET)
        self.assertIn("field_299", question)

    def test_single_line_json_does_not_hide_question_at_end(self):
        question = '{"config": [' + ", ".join('"ignored"' for _ in range(200)) + "]}\nHow do I configure PostgreSQL?"
        query = retrieval_query(question, self.tokenizer)
        self.assertTrue(query.startswith("How do I configure PostgreSQL?"))

    def test_recovery_rewrite_is_deterministic_and_bounded(self):
        question = "How do I configure Postgres CDC? " + "background " * 200
        query = retrieval_query(question, self.tokenizer, recovery=True)
        self.assertEqual(query, retrieval_query(question, self.tokenizer, recovery=True))
        self.assertIn("PostgreSQL", query)
        self.assertIn("change data capture", query)
        self.assertIn("prerequisites", query)
        self.assertLessEqual(len(self.tokenizer.encode(query, add_special_tokens=False).ids), QUERY_TOKEN_BUDGET)

    def test_real_ranker_tokenizer_keeps_passage_space_for_long_configuration(self):
        path = Path(".model_cache/ms-marco-MiniLM-L-12-v2/tokenizer.json")
        if not path.exists():
            self.skipTest("Reranker tokenizer is not cached locally")
        tokenizer = Tokenizer.from_file(str(path))
        tokenizer.no_padding()
        tokenizer.no_truncation()
        question = "```yaml\n" + "\n".join(f"long_configuration_field_{i}: value" for i in range(200)) + (
            "\n```\nHow do I fix PostgreSQL replication_slot_name errors?"
        )
        for recovery in (False, True):
            query = retrieval_query(question, tokenizer, recovery=recovery)
            self.assertIn("replication_slot_name", query)
            self.assertIn("PostgreSQL", query)
            self.assertLessEqual(len(tokenizer.encode(query, add_special_tokens=False).ids), QUERY_TOKEN_BUDGET)


class WorkerLimitTests(unittest.IsolatedAsyncioTestCase):
    async def test_recovery_history_survives_wait_for_and_isolated_concurrent_requests(self):
        retriever = Retriever.__new__(Retriever)
        retriever._sem, retriever._workers = asyncio.Semaphore(2), set()
        retriever._request_evidence = ContextVar("test_evidence", default=None)

        def work(question, product, recovery):
            if recovery:
                return list(retriever._request_evidence.get())
            return [chunk(question, "https://olake.io/docs/" + question)]

        retriever._retrieve_sync = work

        async def ask(question):
            retriever.begin_request()
            await asyncio.wait_for(retriever.retrieve(question), 1)
            await asyncio.sleep(0.01)
            return await asyncio.wait_for(retriever.retrieve(question, recovery=True), 1)

        first, second = await asyncio.gather(ask("postgres"), ask("mysql"))
        self.assertEqual([item.text for item in first], ["postgres"])
        self.assertEqual([item.text for item in second], ["mysql"])

    async def test_readiness_probes_share_one_worker_after_timeout(self):
        retriever = Retriever.__new__(Retriever)
        retriever._workers, retriever._probes = set(), {}
        started, release = threading.Event(), threading.Event()
        calls = []

        def readiness():
            calls.append(1)
            started.set()
            release.wait(2)
            return {"status": "ready"}

        retriever.readiness = readiness
        first = asyncio.create_task(retriever.check_readiness())
        try:
            await asyncio.to_thread(started.wait, 1)
            first.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await first
            second = asyncio.create_task(retriever.check_readiness())
            await asyncio.sleep(0.01)
            self.assertEqual(len(calls), 1)
            release.set()
            self.assertEqual(await second, {"status": "ready"})
        finally:
            release.set()
            await asyncio.gather(*tuple(retriever._workers), return_exceptions=True)

    async def test_cancelled_request_keeps_worker_slot_until_thread_finishes(self):
        retriever = Retriever.__new__(Retriever)
        retriever._sem = asyncio.Semaphore(1)
        retriever._workers = set()
        started, release = threading.Event(), threading.Event()

        def work():
            started.set()
            release.wait(2)

        task = asyncio.create_task(retriever._run_worker(work))
        try:
            await asyncio.to_thread(started.wait, 1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertTrue(retriever._sem.locked())
        finally:
            release.set()
            await asyncio.gather(*tuple(retriever._workers), return_exceptions=True)
            await asyncio.sleep(0)
        self.assertFalse(retriever._sem.locked())


if __name__ == "__main__":
    unittest.main()
