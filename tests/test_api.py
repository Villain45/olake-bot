import asyncio
import os
import unittest
from unittest.mock import patch

import httpx

from app import main
from app.config import LLM_MODEL_PRIORITY
from app.diagnostics import record
from app.schemas import AskResponse


class FakeRetriever:
    def __init__(self, settings):
        self.version = "snapshot-1"
        self.ready = True

    def index_version(self):
        return self.version

    async def current_index(self):
        return self.version

    async def check_readiness(self):
        return self.readiness()

    def readiness(self):
        if not self.ready:
            raise RuntimeError("Qdrant unavailable")
        return {"status": "ready", "indexed_chunks": 3, "index_version": self.version}

    async def aclose(self):
        pass


class FakeLLM:
    def __init__(self, settings):
        pass

    async def aclose(self):
        pass


class FakeRAG:
    def __init__(self, settings, retriever, llm):
        self.calls = []
        self.traces = {}
        self.gate = None
        self.started = asyncio.Event()

    async def ask(self, question, product=None):
        self.calls.append((question, product))
        trace = self.traces.get(question, {})
        if "top_score" in trace:
            record(top_score=trace["top_score"])
        self.started.set()
        if self.gate is not None:
            await self.gate.wait()
        await asyncio.sleep(0.01 if question != "slow" else 0.2)
        record(**{name: value for name, value in trace.items() if name != "top_score"})
        if question == "error":
            raise RuntimeError("upstream error")
        return AskResponse(answer=question, sources=[] if question == "refuse" else ["https://olake.io/docs/"])


class APITests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.patches = [
            patch.object(main, "Retriever", FakeRetriever),
            patch.object(main, "LLM", FakeLLM),
            patch.object(main, "RAG", FakeRAG),
            patch.object(main.limiter, "enabled", False),
            patch.object(main.settings, "openrouter_api_key", "test-only"),
            patch.object(main.settings, "request_timeout_s", 1.0),
            patch.object(main.settings, "max_question_chars", 2000),
        ]
        for item in self.patches:
            item.start()
        self.lifespan = main.lifespan(main.app)
        await self.lifespan.__aenter__()
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test")

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.lifespan.__aexit__(None, None, None)
        for item in reversed(self.patches):
            item.stop()

    async def test_preserves_question_format_and_contract(self):
        question = "How do I configure this?\n```yaml\nsource:\n  name: MyTable\n```"
        response = await self.client.post("/ask", json={"question": question})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(set(response.json()), {"answer", "sources"})
        self.assertEqual(main.app.state.rag.calls, [(question, None)])

    async def test_models_lists_only_the_three_allowed_models_in_priority_order(self):
        response = await self.client.get("/models")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"models": list(LLM_MODEL_PRIORITY), "default": LLM_MODEL_PRIORITY[0]})
        self.assertFalse(main.app.state.rag.calls)

    async def test_version_reads_render_environment_on_each_request(self):
        with patch.dict(os.environ, {"RENDER_GIT_COMMIT": "commit-first", "RENDER_INSTANCE_ID": "instance-first"}):
            first = await self.client.get("/version")
            self.assertEqual(first.status_code, 200)
            self.assertEqual(first.json(), {"commit": "commit-first", "deployment": "instance-first"})
            os.environ["RENDER_GIT_COMMIT"] = "commit-second"
            os.environ["RENDER_INSTANCE_ID"] = "instance-second"
            second = await self.client.get("/version")
            self.assertEqual(second.status_code, 200)
            self.assertEqual(second.json(), {"commit": "commit-second", "deployment": "instance-second"})
        self.assertFalse(main.app.state.rag.calls)

    async def test_version_returns_null_for_missing_render_variables(self):
        with patch.dict(os.environ):
            for name in ("RENDER_GIT_COMMIT", "RENDER_INSTANCE_ID"):
                os.environ.pop(name, None)
            response = await self.client.get("/version")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"commit": None, "deployment": None})
        self.assertFalse(main.app.state.rag.calls)

    async def test_rejects_empty_or_oversized_questions(self):
        for question in [" \n\t", "x" * 2001]:
            response = await self.client.post("/ask", json={"question": question})
            self.assertEqual(response.status_code, 422)
        with patch.object(main.settings, "max_question_chars", 10):
            response = await self.client.post("/ask", json={"question": "x" * 11})
            self.assertEqual(response.status_code, 422)
        self.assertFalse(main.app.state.rag.calls)

    async def test_case_sensitive_cache_and_index_invalidation(self):
        responses = []
        for question in ["MyTable", "mytable", "MyTable"]:
            responses.append(await self.client.post("/ask", json={"question": question}))
        self.assertEqual(len(main.app.state.rag.calls), 2)
        self.assertEqual([response.headers["X-Cache"] for response in responses], ["MISS", "MISS", "HIT"])
        self.assertTrue(all(response.headers["X-Index-Version"] == "snapshot-1" for response in responses))
        main.app.state.retriever.version = "snapshot-2"
        fresh = await self.client.post("/ask", json={"question": "MyTable"})
        self.assertEqual(fresh.headers["X-Cache"], "MISS")
        self.assertEqual(fresh.headers["X-Index-Version"], "snapshot-2")
        cached = await self.client.post("/ask", json={"question": "MyTable"})
        self.assertEqual(cached.headers["X-Cache"], "HIT")
        self.assertEqual(cached.headers["X-Index-Version"], "snapshot-2")
        self.assertEqual(len(main.app.state.rag.calls), 3)

    async def test_answer_diagnostics_are_preserved_on_cache_hit(self):
        main.app.state.rag.traces["diagnostics"] = {
            "top_score": 0.81234567, "answer_model": LLM_MODEL_PRIORITY[1],
            "retrieval_recovered": True, "answer_recovered": False,
        }
        first = await self.client.post("/ask", json={"question": "diagnostics"})
        cached = await self.client.post("/ask", json={"question": "diagnostics"})
        self.assertEqual(first.headers["X-Cache"], "MISS")
        self.assertEqual(cached.headers["X-Cache"], "HIT")
        for response in (first, cached):
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["X-Index-Version"], "snapshot-1")
            self.assertEqual(response.headers["X-Retrieval-Score"], "0.812346")
            self.assertEqual(response.headers["X-Answer-Model"], LLM_MODEL_PRIORITY[1])
            self.assertEqual(response.headers["X-Recovery-Attempted"], "true")
            self.assertNotIn("X-Refusal-Reason", response.headers)
            self.assertEqual(set(response.json()), {"answer", "sources"})
        self.assertEqual(len(main.app.state.rag.calls), 1)

    async def test_coalesced_answer_and_refusal_preserve_producer_diagnostics(self):
        for question in ("shared diagnostics", "refuse"):
            with self.subTest(question=question):
                main.app.state.rag.gate = asyncio.Event()
                main.app.state.rag.started.clear()
                trace = {"top_score": 0.45, "answer_model": LLM_MODEL_PRIORITY[2], "answer_recovered": True}
                if question == "refuse":
                    trace["refusal_reason"] = "unsupported_answer"
                main.app.state.rag.traces[question] = trace
                first = asyncio.create_task(self.client.post("/ask", json={"question": question}))
                await asyncio.wait_for(main.app.state.rag.started.wait(), 1)
                others = [asyncio.create_task(self.client.post("/ask", json={"question": question})) for _ in range(3)]
                try:
                    await asyncio.sleep(0.01)
                finally:
                    main.app.state.rag.gate.set()
                responses = await asyncio.gather(first, *others)
                self.assertEqual([response.headers["X-Cache"] for response in responses], ["MISS"] + ["COALESCED"] * 3)
                for response in responses:
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.headers["X-Retrieval-Score"], "0.450000")
                    self.assertEqual(response.headers["X-Answer-Model"], LLM_MODEL_PRIORITY[2])
                    self.assertEqual(response.headers["X-Recovery-Attempted"], "true")
                    self.assertEqual(response.headers["X-Index-Version"], "snapshot-1")
                    self.assertEqual(response.headers.get("X-Refusal-Reason"), trace.get("refusal_reason"))
                self.assertEqual(sum(item[0] == question for item in main.app.state.rag.calls), 1)

    async def test_concurrent_different_questions_keep_diagnostics_isolated(self):
        main.app.state.rag.traces.update({
            "answer": {"top_score": 0.93, "answer_model": LLM_MODEL_PRIORITY[0], "retrieval_recovered": False},
            "refuse": {"top_score": 0.12, "answer_model": LLM_MODEL_PRIORITY[1],
                       "retrieval_recovered": True, "refusal_reason": "insufficient_evidence"},
        })
        answer, refusal = await asyncio.gather(*[
            self.client.post("/ask", json={"question": question}) for question in ("answer", "refuse")
        ])
        for response in (answer, refusal):
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["X-Cache"], "MISS")
        self.assertEqual(answer.headers["X-Retrieval-Score"], "0.930000")
        self.assertEqual(answer.headers["X-Answer-Model"], LLM_MODEL_PRIORITY[0])
        self.assertEqual(answer.headers["X-Recovery-Attempted"], "false")
        self.assertNotIn("X-Refusal-Reason", answer.headers)
        self.assertEqual(refusal.headers["X-Retrieval-Score"], "0.120000")
        self.assertEqual(refusal.headers["X-Answer-Model"], LLM_MODEL_PRIORITY[1])
        self.assertEqual(refusal.headers["X-Recovery-Attempted"], "true")
        self.assertEqual(refusal.headers["X-Refusal-Reason"], "insufficient_evidence")
        cached_answer, cached_refusal = await asyncio.gather(*[
            self.client.post("/ask", json={"question": question}) for question in ("answer", "refuse")
        ])
        for original, cached in ((answer, cached_answer), (refusal, cached_refusal)):
            self.assertEqual(cached.headers["X-Cache"], "HIT")
            for name in ("X-Retrieval-Score", "X-Answer-Model", "X-Recovery-Attempted", "X-Refusal-Reason"):
                self.assertEqual(cached.headers.get(name), original.headers.get(name))
        self.assertEqual(len(main.app.state.rag.calls), 2)

    async def test_identical_concurrent_questions_share_work(self):
        responses = await asyncio.gather(*[
            self.client.post("/ask", json={"question": "same"}) for _ in range(5)
        ])
        self.assertTrue(all(response.status_code == 200 for response in responses))
        self.assertEqual(len(main.app.state.rag.calls), 1)

    async def test_product_is_validated_and_passed_through(self):
        response = await self.client.post("/ask", json={"question": "compaction", "product": "fusion"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(main.app.state.rag.calls, [("compaction", "fusion")])
        invalid = await self.client.post("/ask", json={"question": "compaction", "product": "unknown"})
        self.assertEqual(invalid.status_code, 422)

    async def test_readiness_fails_without_blocking_liveness(self):
        main.app.state.retriever.ready = False
        self.assertEqual((await self.client.get("/readyz")).status_code, 503)
        self.assertEqual((await self.client.get("/healthz")).status_code, 200)
        main.app.state.retriever.ready = True
        self.assertEqual((await self.client.get("/readyz")).status_code, 200)

    async def test_deadline_and_errors_are_not_cached(self):
        with patch.object(main.settings, "request_timeout_s", 0.02):
            response = await self.client.post("/ask", json={"question": "slow"})
            self.assertEqual(response.status_code, 504)
        for _ in range(2):
            response = await self.client.post("/ask", json={"question": "error"})
            self.assertEqual(response.status_code, 503)
        self.assertEqual(sum(question == "error" for question, _ in main.app.state.rag.calls), 2)

    async def test_refusals_use_short_lived_cache(self):
        await self.client.post("/ask", json={"question": "refuse"})
        self.assertFalse(main.cache)
        self.assertEqual(len(main.refusal_cache), 1)


if __name__ == "__main__":
    unittest.main()
