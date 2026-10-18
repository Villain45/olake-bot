import asyncio
import time
import unittest
from unittest.mock import AsyncMock

import httpx
from pydantic import ValidationError

from app.config import LLM_MODEL_PRIORITY, Settings
from app.diagnostics import request_trace
from app.llm import LLM, extract_json, validate_llm_result


class ResultValidationTests(unittest.TestCase):
    def test_rejects_coerced_and_incomplete_fields(self):
        valid = {"answerable": True, "answer": "Use this setting.", "used_ids": [1]}
        for change in (
            {"answerable": "false"}, {"used_ids": [True]}, {"used_ids": ["1"]},
            {"answer": "  "}, {"used_ids": []}, {"answer": 3}, {"used_ids": [0]},
        ):
            with self.subTest(change=change), self.assertRaises(ValidationError):
                validate_llm_result({**valid, **change})
        with self.assertRaises(ValidationError):
            validate_llm_result({"answer": "Missing fields"})

    def test_accepts_typed_refusal_and_trims_answer(self):
        result = validate_llm_result({"answerable": False, "answer": " No evidence. ", "used_ids": []})
        self.assertFalse(result.answerable)
        self.assertEqual(result.answer, "No evidence.")

    def test_accepts_optional_clarification_for_ambiguous_question(self):
        result = validate_llm_result({
            "answerable": False, "answer": "More details are needed.", "used_ids": [],
            "clarification": " Which catalog are you using? ",
        })
        self.assertEqual(result.clarification, "Which catalog are you using?")
        ordinary = validate_llm_result({"answerable": False, "answer": "No evidence.", "used_ids": []})
        self.assertNotIn("clarification", ordinary.model_dump(exclude_none=True))

    def test_rejects_invalid_or_answerable_clarification(self):
        valid = {"answerable": False, "answer": "More details needed.", "used_ids": [],
                 "clarification": "Which catalog are you using?"}
        for change in (
            {"answerable": True, "used_ids": [1]}, {"used_ids": [1]},
            {"clarification": " "}, {"clarification": "Specify a catalog."},
            {"clarification": "x" * 300 + "?"}, {"clarification": 1},
        ):
            with self.subTest(change=change), self.assertRaises(ValidationError):
                validate_llm_result({**valid, **change})

    def test_tolerates_model_thinking_and_fences(self):
        self.assertEqual(extract_json('<think>hidden</think>```json\n{"x": 1}\n```'), {"x": 1})
        with self.assertRaises(ValueError):
            extract_json([{"type": "text"}])

    def test_json_extraction_preserves_code_fences_in_answer(self):
        text = '```json\n{"answerable":true,"answer":"Use:\\n```yaml\\na: 1\\n```","used_ids":[1]}\n```'
        self.assertIn("```yaml\na: 1\n```", extract_json(text)["answer"])


class LLMRetryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.llm = LLM(Settings(_env_file=None))
        self.generate = self.llm.generate_json

    async def asyncTearDown(self):
        await self.llm.aclose()

    async def test_invalid_schema_retries_before_accepting_answer(self):
        good = {"answerable": True, "answer": "An answer", "used_ids": [1]}
        self.llm._post = AsyncMock(side_effect=[{**good, "answerable": "true"}, good])
        self.assertEqual(await self.generate("system", "question"), good)
        self.assertEqual(self.llm._post.await_count, 2)
        self.assertEqual([call.kwargs["model"] for call in self.llm._post.await_args_list], list(LLM_MODEL_PRIORITY[:2]))

    async def test_schema_recovery_feedback_does_not_echo_invalid_content(self):
        good = {"answerable": True, "answer": "An answer", "used_ids": [1]}
        self.llm._post = AsyncMock(side_effect=[{**good, "answerable": "secret-invalid-content"}, good])
        with self.assertLogs("olake.llm", level="INFO") as captured:
            self.assertEqual(await self.generate("system", "question"), good)
        correction = self.llm._post.await_args_list[1].args[0]
        self.assertIn("schema validation", correction)
        self.assertIn("boolean", correction)
        self.assertNotIn("secret-invalid-content", correction)
        self.assertNotIn("secret-invalid-content", "\n".join(captured.output))
        self.assertIn("outcome=invalid_schema", "\n".join(captured.output))
        self.assertIn("outcome=success", "\n".join(captured.output))

    async def test_invalid_schema_exhausts_three_attempts(self):
        self.llm._post = AsyncMock(return_value={"answerable": "false"})
        with self.assertRaises(ValidationError):
            await self.generate("system", "question")
        self.assertEqual(self.llm._post.await_count, 3)
        self.assertEqual([call.kwargs["model"] for call in self.llm._post.await_args_list], list(LLM_MODEL_PRIORITY))

    async def test_json_mode_rejection_falls_back_and_validates(self):
        req = httpx.Request("POST", "https://example.test")
        resp = httpx.Response(400, request=req)
        error = httpx.HTTPStatusError("unsupported response format", request=req, response=resp)
        good = {"answerable": True, "answer": "An answer", "used_ids": [1]}
        self.llm._post = AsyncMock(side_effect=[error, good])
        self.assertEqual(await self.generate("system", "question"), good)
        self.assertEqual(self.llm._post.await_args_list[1].args, ("system", "question", False))
        self.assertEqual(self.llm._post.await_args_list[1].kwargs["model"], LLM_MODEL_PRIORITY[0])
        self.assertFalse(self.llm._json_mode[LLM_MODEL_PRIORITY[0]])
        self.assertTrue(self.llm._json_mode[LLM_MODEL_PRIORITY[1]])

    async def test_malformed_envelope_retries(self):
        self.llm.http.post = AsyncMock(return_value=httpx.Response(
            200, request=httpx.Request("POST", "https://example.test"), json={"choices": []}
        ))
        with self.assertRaisesRegex(ValueError, "message content"):
            await self.generate("system", "question")
        self.assertEqual(self.llm.http.post.await_count, 3)

    async def test_truncated_output_retries_even_when_json_is_valid(self):
        self.llm.http.post = AsyncMock(return_value=httpx.Response(
            200, request=httpx.Request("POST", "https://example.test"), json={"choices": [{
                "finish_reason": "length", "message": {
                    "content": '{"answerable":true,"answer":"Partial answer","used_ids":[1]}'
                },
            }]}
        ))
        with self.assertRaisesRegex(ValueError, "truncated"):
            await self.generate("system", "question")
        self.assertEqual(self.llm.http.post.await_count, 3)
        bodies = [call.kwargs["json"] for call in self.llm.http.post.await_args_list]
        self.assertEqual([body["max_tokens"] for body in bodies], [1200, 1800, 2400])
        self.assertNotIn("Response correction", bodies[0]["messages"][0]["content"])
        self.assertIn("exceeded the output limit", bodies[1]["messages"][0]["content"])

    async def test_truncation_growth_respects_absolute_cap_and_configured_baseline(self):
        for baseline, expected in ((3000, [3000, 4096, 4096]), (5000, [5000, 5000, 5000])):
            with self.subTest(baseline=baseline):
                self.llm.s.llm_max_output_tokens = baseline
                self.llm.http.post = AsyncMock(return_value=httpx.Response(
                    200, request=httpx.Request("POST", "https://example.test"),
                    json={"choices": [{"finish_reason": "length", "message": {"content": "{}"}}]},
                ))
                with self.assertRaisesRegex(ValueError, "truncated"):
                    await self.generate("system", "question")
                self.assertEqual([call.kwargs["json"]["max_tokens"]
                                  for call in self.llm.http.post.await_args_list], expected)

    async def test_malformed_json_gets_targeted_feedback_without_growing_token_allowance(self):
        import json
        good = {"answerable": True, "answer": "An answer", "used_ids": [1]}
        self.llm._post = AsyncMock(side_effect=[json.JSONDecodeError("bad", "secret-output", 0), good])
        self.assertEqual(await self.generate("system", "question"), good)
        second_call = self.llm._post.await_args_list[1]
        self.assertIn("not a valid JSON response", second_call.args[0])
        self.assertNotIn("secret-output", second_call.args[0])
        self.assertNotIn("max_output_tokens", second_call.kwargs)

    async def test_actual_request_bodies_follow_allowed_priority_after_403(self):
        sent = []

        def respond(request):
            import json
            body = json.loads(request.content)
            sent.append(body)
            if len(sent) < 3:
                return httpx.Response(403, json={"error": {"message": "Model is unavailable for this key"}})
            return httpx.Response(200, json={"model": LLM_MODEL_PRIORITY[2], "choices": [{"message": {
                "content": '{"answerable":true,"answer":"A clearly written answer.","used_ids":[1]}'
            }}]})

        await self.llm.http.aclose()
        self.llm.http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        result = await self.generate("system", "question")
        self.assertEqual(result["answer"], "A clearly written answer.")
        self.assertEqual([body["model"] for body in sent], list(LLM_MODEL_PRIORITY))
        self.assertTrue(all("models" not in body for body in sent))

    async def test_successful_primary_does_not_call_fallbacks(self):
        self.llm._post = AsyncMock(return_value={"answerable": True, "answer": "Answer", "used_ids": [1]})
        await self.generate("system", "question")
        self.assertEqual(self.llm._post.await_count, 1)
        self.assertEqual(self.llm._post.await_args.kwargs["model"], LLM_MODEL_PRIORITY[0])

    async def test_diagnostics_record_successful_model_after_fallback(self):
        good = {"answerable": True, "answer": "Answer", "used_ids": [1]}
        self.llm._post = AsyncMock(side_effect=[asyncio.TimeoutError(), good])
        trace = {}
        token = request_trace.set(trace)
        try:
            await self.generate("system", "question")
        finally:
            request_trace.reset(token)
        self.assertEqual(trace, {"answer_model": LLM_MODEL_PRIORITY[1]})

    async def test_authentication_failure_is_not_retried(self):
        request = httpx.Request("POST", "https://example.test")
        error = httpx.HTTPStatusError("Invalid API key", request=request, response=httpx.Response(401, request=request))
        self.llm._post = AsyncMock(side_effect=error)
        with self.assertRaises(httpx.HTTPStatusError):
            await self.generate("system", "question")
        self.assertEqual(self.llm._post.await_count, 1)

    async def test_model_timeout_tries_next_allowed_model(self):
        good = {"answerable": True, "answer": "Answer", "used_ids": [1]}
        self.llm._post = AsyncMock(side_effect=[asyncio.TimeoutError(), good])
        self.assertEqual(await self.generate("system", "question"), good)
        self.assertEqual([call.kwargs["model"] for call in self.llm._post.await_args_list], list(LLM_MODEL_PRIORITY[:2]))

    async def test_caller_deadline_bounds_all_model_attempts(self):
        async def slow(*args, **kwargs):
            await asyncio.sleep(1)

        self.llm._post = AsyncMock(side_effect=slow)
        started = time.monotonic()
        with self.assertRaises(asyncio.TimeoutError):
            await self.generate("system", "question", timeout_s=0.06)
        self.assertLess(time.monotonic() - started, 0.4)
        self.assertEqual(self.llm._post.await_count, 3)

    async def test_expired_caller_deadline_does_not_send_request(self):
        self.llm._post = AsyncMock()
        with self.assertRaises(asyncio.TimeoutError):
            await self.generate("system", "question", timeout_s=0)
        self.llm._post.assert_not_awaited()

    async def test_primary_clarification_does_not_call_fallback_models(self):
        result = {"answerable": False, "answer": "Please specify the catalog.", "used_ids": [],
                  "clarification": "Which catalog are you using?"}
        self.llm._post = AsyncMock(return_value=result)
        self.assertEqual(await self.generate("system", "question"), result)
        self.llm._post.assert_awaited_once()


class ModelPolicyTests(unittest.TestCase):
    def test_previous_glm_first_settings_use_gemini_first_routing(self):
        settings = Settings(
            _env_file=None,
            llm_model="z-ai/glm-4.7-flash",
            llm_fallback_models=["google/gemini-2.5-flash-lite", "deepseek/deepseek-v4.1-flash"],
        )
        self.assertEqual(settings.llm_model, "google/gemini-2.5-flash-lite")
        self.assertEqual(settings.llm_fallback_models,
                         ["z-ai/glm-4.7-flash", "deepseek/deepseek-v4.1-flash"])
        self.assertEqual(settings.llm_models, (settings.llm_model, *settings.llm_fallback_models))

    def test_legacy_models_are_replaced_without_changing_index_settings(self):
        settings = Settings(_env_file=None, llm_model="old/forbidden-model", llm_fallback_models=["another/forbidden"],
                            collection="completed_index", embed_model="existing-embedder", embed_dim=384)
        self.assertEqual(settings.llm_models, LLM_MODEL_PRIORITY)
        self.assertEqual((settings.llm_model, *settings.llm_fallback_models), LLM_MODEL_PRIORITY)
        self.assertEqual((settings.collection, settings.embed_model, settings.embed_dim),
                         ("completed_index", "existing-embedder", 384))
