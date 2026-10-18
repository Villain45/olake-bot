import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from app.config import Settings
from app.models import Chunk
from app.llm import validate_llm_result
from app.rag import IDK, RAG, _clarify, _render_result


def chunk(text="Supported setting.", url="https://olake.io/docs/a", score=0.9, source_type="docs"):
    return Chunk(0, url, "Title", "Overview", text, source_type, score)


class RAGTests(unittest.IsolatedAsyncioTestCase):
    def make_rag(self, chunks, result=None, **settings):
        settings.setdefault("web_fallback_enabled", False)
        retriever = SimpleNamespace(retrieve=AsyncMock(return_value=chunks), rerank_external=AsyncMock())
        llm = SimpleNamespace(generate_json=AsyncMock(return_value=result or {
            "answerable": True, "answer": "Use the setting.", "used_ids": [1],
        }))
        return RAG(Settings(_env_file=None, **settings), retriever, llm)

    async def test_low_confidence_without_web_refuses_before_llm(self):
        rag = self.make_rag([chunk(score=0.1)], web_fallback_enabled=False)
        result = await rag.ask("How?")
        self.assertEqual((result.answer, result.sources), (IDK, []))
        rag.llm.generate_json.assert_not_awaited()

    async def test_no_search_results_refuses(self):
        rag = self.make_rag([chunk(score=0.1)], web_fallback_enabled=True)
        with patch("app.rag.web_fallback.search", new=AsyncMock(return_value=[])):
            self.assertEqual((await rag.ask("How?")).sources, [])
        rag.llm.generate_json.assert_not_awaited()

    async def test_external_evidence_is_reranked_and_low_score_refused(self):
        external = chunk(score=0.0, source_type="external")
        rag = self.make_rag([], web_fallback_enabled=True)
        rag.retriever.rerank_external.return_value = [external]
        with patch("app.rag.web_fallback.search", new=AsyncMock(return_value=[external])):
            result = await rag.ask("How?")
        self.assertEqual((result.answer, result.sources), (IDK, []))
        rag.retriever.rerank_external.assert_awaited_once_with("How?", [external])
        rag.llm.generate_json.assert_not_awaited()

    async def test_external_answer_label_is_enforced(self):
        external = chunk(score=0.8, source_type="external", url="https://iceberg.apache.org/docs/a")
        rag = self.make_rag([], web_fallback_enabled=True)
        rag.retriever.rerank_external.return_value = [external]
        with patch("app.rag.web_fallback.search", new=AsyncMock(return_value=[external])):
            result = await rag.ask("How?")
        self.assertTrue(result.answer.startswith("Based on external sources"))
        self.assertEqual(result.sources, [external.url])

    async def test_refusal_and_invalid_citations_never_gain_sources(self):
        bad_results = [
            {"answerable": False, "answer": "Maybe guessed content", "used_ids": [1]},
            {"answerable": "false", "answer": "Maybe guessed content", "used_ids": [1]},
            {"answerable": True, "answer": "Maybe guessed content", "used_ids": []},
            {"answerable": True, "answer": "Maybe guessed content", "used_ids": [99]},
            {"answerable": True, "answer": "Maybe guessed content", "used_ids": [True]},
        ]
        for result in bad_results:
            with self.subTest(result=result):
                rag = self.make_rag([chunk()], result=result)
                answer = await rag.ask("How?")
                self.assertEqual((answer.answer, answer.sources), (IDK, []))

    async def test_prompt_deduplicates_text_retains_distinct_same_url_chunks_and_question(self):
        original = chunk()
        rag = self.make_rag([
            original,
            chunk(url="https://olake.io/docs/duplicate", score=0.8),
            chunk(text="Different text same URL", score=0.7),
            chunk(text="Unique evidence", url="https://olake.io/docs/b", score=0.6),
        ])
        question = "Why does this fail?\n```yaml\nname: CaseSensitive\n```"
        result = await rag.ask(question)
        user_prompt = rag.llm.generate_json.await_args.args[1]
        self.assertIn(question, user_prompt)
        self.assertEqual(user_prompt.count("source_type="), 3)
        self.assertIn("Different text same URL", user_prompt)
        self.assertEqual(result.sources, [original.url])
        self.assertEqual(original.id, 0)

    async def test_deduplication_preserves_yaml_indentation(self):
        rag = self.make_rag([
            chunk(text="root:\n  key: value"),
            chunk(text="root:\nkey: value", score=0.8),
        ])
        await rag.ask("How?")
        self.assertEqual(rag.llm.generate_json.await_args.args[1].count("source_type="), 2)

    async def test_explicit_product_passes_to_retrieval(self):
        rag = self.make_rag([chunk()])
        await rag.ask("How?", product="fusion")
        rag.retriever.retrieve.assert_awaited_once_with("How?", product="fusion")

    async def test_weak_retrieval_recovers_locally_once(self):
        rag = self.make_rag([])
        rag.retriever.begin_request = Mock()
        rag.retriever.retrieve.side_effect = [[chunk(score=0.1)], [chunk()]]
        result = await rag.ask("PostgreSQL CDC replication", product="ingestion")
        self.assertEqual(result.sources, [chunk().url])
        rag.retriever.begin_request.assert_called_once_with()
        self.assertEqual(rag.retriever.retrieve.await_args_list[1].kwargs, {"product": "ingestion", "recovery": True})
        self.assertEqual(rag.retriever.retrieve.await_count, 2)
        rag.llm.generate_json.assert_awaited_once()

    async def test_unanswerable_result_retries_with_new_evidence(self):
        rag = self.make_rag([chunk()])
        rag.retriever.retrieve.side_effect = [[chunk()], [chunk(text="New supported detail.", url="https://olake.io/docs/b")]]
        rag.llm.generate_json.side_effect = [
            {"answerable": False, "answer": "Not enough evidence.", "used_ids": []},
            {"answerable": True, "answer": "Use the new detail. [2]", "used_ids": [2]},
        ]
        result = await rag.ask("PostgreSQL CDC replication")
        self.assertEqual(result.sources, ["https://olake.io/docs/b"])
        self.assertIn("New supported detail.", rag.llm.generate_json.await_args.args[1])
        self.assertIn("Server validation feedback", rag.llm.generate_json.await_args.args[1])
        self.assertEqual(rag.retriever.retrieve.await_count, 2)
        self.assertEqual(rag.llm.generate_json.await_count, 2)

    async def test_optional_recovery_failure_retains_original_evidence(self):
        rag = self.make_rag([chunk()])
        rag.retriever.retrieve.side_effect = [[chunk()], RuntimeError("optional search failed")]
        rag.llm.generate_json.side_effect = [
            {"answerable": True, "answer": "Unsupported [99]", "used_ids": [99]},
            {"answerable": True, "answer": "Supported. [1]", "used_ids": [1]},
        ]
        self.assertEqual((await rag.ask("How?")).sources, [chunk().url])
        self.assertIn(chunk().text, rag.llm.generate_json.await_args.args[1])

    async def test_optional_web_failure_does_not_break_repair(self):
        rag = self.make_rag([chunk()], web_fallback_enabled=True)
        rag.llm.generate_json.side_effect = [
            {"answerable": False, "answer": "Not enough evidence.", "used_ids": []},
            {"answerable": True, "answer": "Supported. [1]", "used_ids": [1]},
        ]
        with patch("app.rag.web_fallback.search", new=AsyncMock(side_effect=TimeoutError)) as search:
            self.assertEqual((await rag.ask("How?")).sources, [chunk().url])
        search.assert_awaited_once()

    async def test_recovery_calls_are_bounded(self):
        rag = self.make_rag([chunk(score=0.1)], web_fallback_enabled=True)
        with patch("app.rag.web_fallback.search", new=AsyncMock(return_value=[])) as search:
            result = await rag.ask("How?")
        self.assertEqual(result.answer, IDK)
        self.assertEqual(rag.retriever.retrieve.await_count, 2)
        search.assert_awaited_once()
        rag.llm.generate_json.assert_not_awaited()

    async def test_insufficient_time_skips_optional_recovery(self):
        rag = self.make_rag([chunk()], result={"answerable": False, "answer": "No evidence", "used_ids": []}, recovery_min_remaining_s=100)
        result = await rag.ask("How?")
        self.assertEqual(result.answer, IDK)
        rag.retriever.retrieve.assert_awaited_once()
        rag.llm.generate_json.assert_awaited_once()
        self.assertGreater(rag.llm.generate_json.await_args.kwargs["timeout_s"], 0)
        self.assertLess(rag.llm.generate_json.await_args.kwargs["timeout_s"], rag.s.request_timeout_s)

    async def test_recovery_preserves_cited_original_sections(self):
        originals = [chunk(text=f"Original section {i}.", score=0.9-i*0.01) for i in range(6)]
        novel = [chunk(text=f"New section {i}.", url="https://olake.io/docs/b", score=0.8) for i in range(6)]
        rag = self.make_rag(originals)
        rag.retriever.retrieve.side_effect = [originals, novel]
        rag.llm.generate_json.side_effect = [
            {"answerable": True, "answer": "Unsupported [99]", "used_ids": [3, 4, 5]},
            {"answerable": True, "answer": "Supported [1]", "used_ids": [1]},
        ]
        await rag.ask("How?")
        prompt = rag.llm.generate_json.await_args.args[1]
        for i in (2, 3, 4):
            self.assertIn(f"Original section {i}.", prompt)
        self.assertEqual(sum(f"New section {i}." in prompt for i in range(6)), 2)

    async def test_clarification_does_not_invoke_models_or_search(self):
        rag = self.make_rag([])
        result = await rag.ask("How do I configure a source in the OLake UI?")
        self.assertEqual(result.answer, "Which source database are you configuring?")
        self.assertEqual(result.sources, [])
        rag.retriever.retrieve.assert_not_awaited()
        rag.llm.generate_json.assert_not_awaited()
        for question in ("How can I see source logs in the OLake UI?", "How do source connectors recover from failures?", "How do I change source database credentials in OLake?", "How does catalog management work?"):
            self.assertIsNone(_clarify(question), question)

    async def test_model_clarification_is_returned_without_sources(self):
        rag = self.make_rag([chunk()], result={"answerable": False, "answer": "Need a catalog.", "used_ids": [], "clarification": "Which catalog are you using?"})
        result = await rag.ask("How?")
        self.assertEqual((result.answer, result.sources), ("Which catalog are you using?", []))
        rag.llm.generate_json.assert_awaited_once()

    def render(self, answer, evidence="Supported setting.", used_ids=None):
        result = validate_llm_result({"answerable": True, "answer": answer, "used_ids": used_ids or [1]})
        return _render_result(result, [Chunk(1, chunk().url, "Title", "Overview", evidence, "docs", 0.9)])

    def test_inline_citations_are_server_links_and_code_arrays_preserved(self):
        response, issue = self.render("Use this setting. [1]\n```json\n[99]\n```\nInline `[88]` is an array.")
        self.assertIsNone(issue)
        self.assertIn("[1](https://olake.io/docs/a)", response.answer)
        self.assertIn("```json\n[99]\n```", response.answer)
        self.assertIn("`[88]`", response.answer)
        self.assertEqual(response.sources, [chunk().url])

    def test_bad_inline_citations_and_invented_urls_are_rejected(self):
        for answer in ("Claim [99]", "Claim [1, 99]", "Claim [99](https://olake.io/docs/a)", "Claim [1, 99](https://olake.io/docs/a)", "Claim https://invented.example/path"):
            with self.subTest(answer=answer):
                response, issue = self.render(answer)
                self.assertIsNone(response)
                self.assertIsNotNone(issue)

    def test_grouped_citations_are_validated_and_resolved(self):
        evidence = [Chunk(1, chunk().url, "Title", "Overview", "First evidence.", "docs", 0.9),
                    Chunk(2, "https://olake.io/docs/b", "Title", "Overview", "Second evidence.", "docs", 0.8)]
        result = validate_llm_result({"answerable": True, "answer": "Supported by both [1, 2].", "used_ids": [1, 2]})
        response, issue = _render_result(result, evidence)
        self.assertIsNone(issue)
        self.assertIn("[1](https://olake.io/docs/a), [2](https://olake.io/docs/b)", response.answer)
        self.assertEqual(len(response.sources), 2)

    def test_known_url_followed_by_punctuation_is_accepted(self):
        response, issue = self.render("See https://olake.io/docs/a.")
        self.assertIsNone(issue)
        self.assertEqual(response.sources, [chunk().url])

    def test_invented_configuration_keys_and_flags_are_rejected(self):
        for answer in ('```yaml\ninvented_key: true\n```', 'Run `olake --invented-flag`.'):
            response, issue = self.render(answer)
            self.assertIsNone(response)
            self.assertIn("absent from the cited evidence", issue)
        response, issue = self.render('Configuration:\n```yaml\nreplication_slot: my_slot\n```\nRun `olake --config config.json`.', evidence='replication_slot: my_slot; use olake --config config.json')
        self.assertIsNone(issue)
        self.assertTrue(response.sources)
