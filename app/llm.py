import asyncio
import json
import logging
import re
from typing import Annotated

import httpx
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, ValidationError, field_validator, model_validator

from .config import LLM_MODEL_PRIORITY, Settings
from .diagnostics import record

log = logging.getLogger("olake.llm")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_MAX_RECOVERY_OUTPUT_TOKENS = 4096


class OutputTruncatedError(ValueError):
    """A syntactically valid partial answer is still an incomplete answer."""


class LLMResult(BaseModel):
    """The model's answer contract; never coerce strings into booleans or IDs."""

    model_config = ConfigDict(extra="forbid", strict=True)

    answerable: StrictBool
    answer: StrictStr
    used_ids: list[Annotated[StrictInt, Field(gt=0)]]
    clarification: StrictStr | None = None

    @field_validator("answer")
    @classmethod
    def nonempty_answer(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("answer must not be empty")
        return value

    @field_validator("clarification")
    @classmethod
    def concise_clarification(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value or len(value) > 300 or not value.endswith("?"):
            raise ValueError("clarification must be a question of at most 300 characters")
        return value

    @model_validator(mode="after")
    def require_citations(self) -> "LLMResult":
        if self.answerable and not self.used_ids:
            raise ValueError("answerable responses must cite supporting passage IDs")
        if self.clarification is not None and (self.answerable or self.used_ids):
            raise ValueError("clarification requires answerable=false and no passage IDs")
        return self


def validate_llm_result(value: object) -> LLMResult:
    return LLMResult.model_validate(value)


def extract_json(text: str) -> dict:
    """Tolerant JSON parsing: strips <think> blocks and ``` fences."""
    if not isinstance(text, str):
        raise ValueError("LLM content must be a string")
    text = re.sub(r"\A\s*<think>.*?</think>", "", text, flags=re.S).strip()
    # Locate the JSON object instead of removing every code fence: fences inside
    # the answer are user-facing configuration examples and must be preserved.
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in LLM output")
    return json.loads(text[start : end + 1])


def _retryable(exc: BaseException) -> bool:
    # Move to the next permitted model for model-specific failures, network errors,
    # throttling and invalid output. Authentication/billing failures stop immediately.
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (400, 403, 404, 408, 422, 429) or exc.response.status_code >= 500
    return isinstance(exc, (httpx.TransportError, asyncio.TimeoutError, ValueError, KeyError))


def _failure_code(exc: BaseException) -> str:
    """Log only controlled categories, never response bodies or user input."""
    if isinstance(exc, httpx.HTTPStatusError):
        return f"http_{exc.response.status_code}"
    if isinstance(exc, OutputTruncatedError):
        return "truncated"
    if isinstance(exc, ValidationError):
        return "invalid_schema"
    if isinstance(exc, json.JSONDecodeError):
        return "invalid_json"
    if isinstance(exc, asyncio.TimeoutError):
        return "timeout"
    if isinstance(exc, httpx.TransportError):
        return "transport_error"
    return "invalid_response"


def _recovery_feedback(exc: BaseException) -> str:
    """Give the next permitted model useful correction without echoing bad output."""
    if isinstance(exc, OutputTruncatedError):
        return (
            "The previous attempt exceeded the output limit. Keep the answer concise, "
            "finish any code examples, and return the complete JSON object."
        )
    if isinstance(exc, ValidationError):
        return (
            "The previous attempt failed JSON schema validation. Use a boolean answerable, "
            "a nonempty string answer, and an array of positive integer used_ids. "
            "An answerable response must cite passages. If clarification is needed, set "
            "answerable=false, used_ids=[], and clarification to a question under 300 characters."
        )
    if isinstance(exc, (ValueError, KeyError)):
        return (
            "The previous attempt was not a valid JSON response. Return one complete JSON "
            "object only, with answerable, answer, and used_ids; escape quotes and newlines "
            "inside JSON strings."
        )
    return ""


class LLM:
    """OpenRouter (OpenAI-compatible chat completions). Swap this class if the
    organizers give a different endpoint."""

    def __init__(self, s: Settings):
        self.s = s
        self.sem = asyncio.Semaphore(s.llm_concurrency)
        self.http = httpx.AsyncClient(
            timeout=s.llm_timeout_s,
            headers={
                "Authorization": f"Bearer {s.openrouter_api_key}",
                "HTTP-Referer": s.app_url,  # optional, shows up in OpenRouter dashboard
                "X-Title": "Ask OLake",
            },
        )
        self._json_mode = {model: s.llm_json_mode for model in LLM_MODEL_PRIORITY}

    async def aclose(self) -> None:
        await self.http.aclose()

    async def _post(
        self, system: str, user: str, json_mode: bool, *, model: str,
        max_output_tokens: int | None = None,
    ) -> dict:
        s = self.s
        if model not in LLM_MODEL_PRIORITY:
            raise ValueError("Model is not permitted by this deployment")
        if s.llm_use_system_instruction:
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        else:
            messages = [{"role": "user", "content": f"{system}\n\n{user}"}]

        body: dict = {
            "model": model,
            "messages": messages,
            "temperature": s.llm_temperature,
            "max_tokens": max_output_tokens if max_output_tokens is not None else s.llm_max_output_tokens,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        resp = await self.http.post(OPENROUTER_URL, json=body)
        resp.raise_for_status()
        data = resp.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (IndexError, KeyError, TypeError) as exc:
            raise ValueError("LLM response has no message content") from exc
        if data["choices"][0].get("finish_reason") == "length":
            raise OutputTruncatedError("LLM response was truncated at the output token limit")
        if data.get("model"):
            log.info("served by %s", data["model"])
        usage = data.get("usage")
        if isinstance(usage, dict):
            log.info("LLM usage prompt_tokens=%s completion_tokens=%s", *(
                value if type(value) is int else None
                for value in (usage.get("prompt_tokens"), usage.get("completion_tokens"))
            ))
        return extract_json(content)

    async def generate_json(self, system: str, user: str, *, timeout_s: float | None = None) -> dict:
        async with self.sem:
            # Reserve time for all three models within the API's overall deadline.
            budget = max(0.001, self.s.request_timeout_s - self.s.web_timeout_s - 5)
            if timeout_s is not None:
                budget = min(budget, timeout_s)
            if budget <= 0:
                raise asyncio.TimeoutError("No answer generation time remains")
            deadline = asyncio.get_running_loop().time() + budget
            baseline_tokens = self.s.llm_max_output_tokens
            output_tokens = baseline_tokens
            recovery_cap = max(baseline_tokens, min(2 * baseline_tokens, _MAX_RECOVERY_OUTPUT_TOKENS))
            feedback = ""
            for position, model in enumerate(LLM_MODEL_PRIORITY):
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise asyncio.TimeoutError("Answer generation deadline exceeded")
                timeout = min(self.s.llm_timeout_s, max(0.001, remaining / (len(LLM_MODEL_PRIORITY) - position)))
                started = asyncio.get_running_loop().time()
                attempt_system = f"{system}\n\nResponse correction: {feedback}" if feedback else system
                output_options = {"max_output_tokens": output_tokens} if output_tokens != baseline_tokens else {}
                try:
                    async with asyncio.timeout(timeout):
                        try:
                            result = await self._post(attempt_system, user, self._json_mode[model], model=model, **output_options)
                        except httpx.HTTPStatusError as exc:
                            # Unsupported JSON mode is remembered per model, not globally.
                            if self._json_mode[model] and exc.response.status_code == 400:
                                log.warning("JSON mode rejected by %s; retrying without it", model)
                                self._json_mode[model] = False
                                result = await self._post(attempt_system, user, False, model=model, **output_options)
                            else:
                                raise
                        validated = validate_llm_result(result).model_dump(exclude_none=True)
                        record(answer_model=model)
                        log.info("LLM attempt=%d model=%s outcome=success elapsed_s=%.3f", position + 1, model,
                                 asyncio.get_running_loop().time() - started)
                        return validated
                except Exception as exc:
                    outcome = _failure_code(exc)
                    log.warning("LLM attempt=%d model=%s outcome=%s elapsed_s=%.3f", position + 1, model, outcome,
                                asyncio.get_running_loop().time() - started)
                    if not _retryable(exc) or position == len(LLM_MODEL_PRIORITY) - 1:
                        raise
                    feedback = _recovery_feedback(exc) or feedback
                    if isinstance(exc, OutputTruncatedError):
                        output_tokens = min(recovery_cap, output_tokens + max(1, output_tokens // 2))
                    log.warning("Trying next permitted model %s", LLM_MODEL_PRIORITY[position + 1])
            raise RuntimeError("No permitted model produced an answer")
