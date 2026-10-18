import asyncio
import logging
import re
import time
from dataclasses import replace
from urllib.parse import quote

from pydantic import ValidationError

from . import web_fallback
from .config import Settings
from .diagnostics import record
from .llm import LLM, validate_llm_result
from .models import Chunk
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .retrieval import Retriever
from .schemas import AskResponse

log = logging.getLogger("olake.rag")

IDK = (
    "I couldn't find this in the OLake docs or repos, so I'd rather not guess. "
    "Try rephrasing, or ask in the OLake community Slack."
)


def _dedupe(urls: list[str]) -> list[str]:
    return list(dict.fromkeys(urls))


def _dedupe_chunks(chunks: list[Chunk]) -> list[Chunk]:
    seen_text: set[str] = set()
    unique = []
    for chunk in chunks:
        text_key = chunk.text.strip()
        if not text_key or not chunk.url or text_key in seen_text:
            continue
        seen_text.add(text_key)
        unique.append(chunk)
    return unique


def _clarify(question: str) -> str | None:
    """Ask only for genuinely missing setup choices, not broad overview questions."""
    text = question.lower()
    procedural = bool(re.search(r"\b(configur\w*|setup|set up|connect|deploy|install)\b", text))
    if not procedural:
        return None
    method_missing = not re.search(r"\b(ui|cli)\b", text)
    if "catalog" in text and not re.search(r"\b(glue|rest|jdbc|hive|nessie|polaris|snowflake|s3 tables)\b", text):
        if not re.search(r"\b(which|what|supported|available|types|compare|difference|options)\b", text):
            return ("Which Iceberg catalog are you configuring, and are you using the OLake UI or CLI?"
                    if method_missing else "Which Iceberg catalog are you configuring?")
    source_setup = re.search(r"\b(?:configur\w*|setup|set up|connect|install)\s+(?:(?:a|an|the|my|new)\s+)*(?:source|database)\b", text)
    if source_setup and not re.search(r"\b(logs?|status|monitor\w*|credentials?|passwords?)\b", text) and not re.search(
        r"\b(postgres(?:ql)?|mysql|mariadb|mongo(?:db)?|kafka|oracle|db2|sql server|mssql|s3|clickhouse)\b", text
    ):
        if not re.search(r"\b(which|what|supported|available|types|compare|difference|options)\b", text):
            return ("Which source database are you configuring, and are you using the OLake UI or CLI?"
                    if method_missing else "Which source database are you configuring?")
    if len(text.split()) <= 12 and re.search(r"\b(set (?:it|this) up|configure (?:it|this)|deploy (?:it|this))\b", text):
        return "Are you setting up OLake ingestion or OLake Fusion, and where are you deploying it?"
    return None


_CODE = re.compile(r"(`{3,}|~{3,})[^\n]*\n.*?\n\1[ \t]*(?:\n|$)|`[^`\n]+`", re.S)
_CITATION = re.compile(r"(?<!!)\[(\d+(?:\s*,\s*\d+)*)\](?!\()")


def _map_prose(text: str, transform) -> str:
    parts, offset = [], 0
    for match in _CODE.finditer(text):
        parts.extend((transform(text[offset:match.start()]), match.group()))
        offset = match.end()
    parts.append(transform(text[offset:]))
    return "".join(parts)


def _grounding_issue(answer: str, used: list[Chunk]) -> str | None:
    """Catch invented configuration keys/flags; this is not a factual-answer judge."""
    evidence = "\n".join(f"{chunk.title}\n{chunk.breadcrumb}\n{chunk.text}" for chunk in used)
    code = "\n".join(match.group() for match in _CODE.finditer(answer))
    keys = set(re.findall(r"(?m)^\s*[\"']?([A-Za-z_][\w.-]*)[\"']?\s*:", code))
    flags = set(re.findall(r"(?<![\w-])--[A-Za-z][A-Za-z0-9_-]*", answer))
    missing = [term for term in sorted(keys | flags)
               if not re.search(r"(?<![\w.-])" + re.escape(term) + r"(?![\w.-])", evidence)]
    if missing:
        return "Configuration keys or flags are absent from the cited evidence: " + ", ".join(missing[:8])
    return None


def _render_result(result, chunks: list[Chunk]) -> tuple[AskResponse | None, str | None]:
    if result.clarification:
        return AskResponse(answer=result.clarification, sources=[]), None
    if not result.answerable:
        return None, "The supplied passages do not contain enough evidence to answer"
    by_id = {chunk.id: chunk for chunk in chunks}
    if any(identifier not in by_id for identifier in result.used_ids):
        return None, "Use only the passage IDs supplied in the current context"
    used = [by_id[identifier] for identifier in dict.fromkeys(result.used_ids)]
    if not used:
        return None, "An answer needs at least one supporting passage ID"
    markers = []
    _map_prose(result.answer, lambda prose: markers.extend(int(value.strip()) for group in _CITATION.findall(prose) for value in group.split(",")) or prose)
    prelinked = []
    _map_prose(result.answer, lambda prose: prelinked.extend(re.findall(r"\[\d+(?:\s*,\s*\d+)*\]\(", prose)) or prose)
    if prelinked:
        return None, "Use bare [passage_id] citations; the server supplies their URL links"
    if any(identifier not in result.used_ids for identifier in markers):
        return None, "Inline citation numbers must match the supporting used_ids"
    issue = _grounding_issue(result.answer, used)
    if issue:
        return None, issue
    allowed_urls = {chunk.url for chunk in used}
    prose_urls = []
    _map_prose(result.answer, lambda prose: prose_urls.extend(re.findall(r"https?://[^\s<>\])]+", prose)) or prose)
    prose_urls = [url.rstrip(".,!?:;") for url in prose_urls]
    if any(url not in allowed_urls and not any(url in chunk.text for chunk in used) for url in prose_urls):
        return None, "Do not invent URL links; cite the supplied evidence with [passage_id]"

    def link_citations(prose):
        def links(match):
            return ", ".join(f"[{int(value)}]({quote(by_id[int(value)].url, safe=':/?#[]@!$&*+,;=%-')})" for value in match.group(1).split(","))
        return _CITATION.sub(links, prose)

    answer = _map_prose(result.answer, link_citations)
    if not markers:
        links = [f"[{chunk.id}]({quote(chunk.url, safe=':/?#[]@!$&*+,;=%-')})" for chunk in used]
        answer += "\n\nSources: " + " ".join(dict.fromkeys(links))
    if any(chunk.source_type == "external" for chunk in used):
        answer = "Based on external sources outside the OLake docs and repos:\n\n" + answer
    return AskResponse(answer=answer, sources=_dedupe([chunk.url for chunk in used])), None


class RAG:
    def __init__(self, s: Settings, retriever: Retriever, llm: LLM):
        self.s, self.retriever, self.llm = s, retriever, llm

    async def ask(self, question: str, product: str | None = None) -> AskResponse:
        s = self.s
        started = time.monotonic()
        deadline = started + s.request_timeout_s - min(1.0, s.request_timeout_s * 0.05)
        recovered = searched = False
        begin_request = getattr(self.retriever, "begin_request", None)
        if begin_request:
            begin_request()
        clarification = _clarify(question)
        if clarification:
            record(refusal_reason="clarification")
            return AskResponse(answer=clarification, sources=[])

        def remaining():
            return max(0.0, deadline - time.monotonic())

        def supported(values):
            return _dedupe_chunks([chunk for chunk in values if chunk.score >= s.min_rerank_score])[:s.top_k]

        async def retrieve(recovery=False):
            options = {"product": product} if product else {}
            if recovery:
                options["recovery"] = True
            try:
                values = await asyncio.wait_for(self.retriever.retrieve(question, **options), timeout=remaining())
            except Exception as exc:
                if not recovery:
                    raise
                log.warning("optional retrieval recovery failed: %s", type(exc).__name__)
                return []
            top = max((chunk.score for chunk in values), default=0.0)
            record(top_score=max(top, top_scores[0]), threshold=s.min_rerank_score)
            top_scores[0] = max(top_scores[0], top)
            log.info("rag retrieval recovery=%s retrieved=%d accepted=%d top_score=%.4f threshold=%.4f",
                     recovery, len(values), len(supported(values)), top, s.min_rerank_score)
            return supported(values)

        async def external():
            nonlocal searched
            if searched or not s.web_fallback_enabled or remaining() < s.recovery_min_remaining_s:
                return []
            searched = True
            try:
                values = await asyncio.wait_for(web_fallback.search(question, s), timeout=remaining())
                if values:
                    values = await asyncio.wait_for(self.retriever.rerank_external(question, values), timeout=remaining())
            except Exception as exc:
                log.warning("optional web recovery failed: %s", type(exc).__name__)
                return []
            return supported(values)

        top_scores = [0.0]
        chunks = await retrieve()
        if not chunks and s.retrieval_recovery_enabled and remaining() >= s.recovery_min_remaining_s:
            recovered = True
            record(retrieval_recovered=True)
            chunks = await retrieve(recovery=True)
        if not chunks:
            chunks = await external()
        if not chunks:
            record(refusal_reason="insufficient_evidence")
            return AskResponse(answer=IDK, sources=[])

        issue = None
        attempts = 2 if s.answer_recovery_enabled else 1
        for attempt in range(attempts):
            result = None
            context = [replace(chunk, id=number) for number, chunk in enumerate(chunks, start=1)]
            prompt = build_user_prompt(question, context)
            if issue:
                prompt += "\n\nServer validation feedback: " + issue + ". Answer only from the current passages; ask a concise clarification if a required setup choice is missing."
            if remaining() <= 0:
                break
            async with asyncio.timeout(remaining()):
                out = await self.llm.generate_json(SYSTEM_PROMPT, prompt, timeout_s=remaining())
            try:
                result = validate_llm_result(out)
                response, issue = _render_result(result, context)
            except (ValidationError, TypeError, ValueError):
                response, issue = None, "Return the required typed JSON and valid supporting passage IDs"
            if response:
                if not response.sources:
                    record(refusal_reason="clarification")
                return response
            log.info("rag answer rejected attempt=%d reason=%s", attempt + 1, issue)
            if attempt + 1 == attempts or remaining() < s.recovery_min_remaining_s:
                break
            record(answer_recovered=True)
            extra = []
            if not recovered and s.retrieval_recovery_enabled:
                recovered = True
                record(retrieval_recovered=True)
                extra = await retrieve(recovery=True)
            if result is None or not result.answerable:
                extra += await external()
            # Keep supporting sections together; reserve at most two slots for new evidence.
            old_text = {chunk.text.strip() for chunk in chunks}
            novel = _dedupe_chunks([chunk for chunk in extra if chunk.text.strip() not in old_text])
            originals = chunks
            if result is not None and result.answerable:
                cited = {identifier - 1 for identifier in result.used_ids if 0 < identifier <= len(chunks)}
                originals = [value for i, value in enumerate(chunks) if i in cited] + [value for i, value in enumerate(chunks) if i not in cited]
            new_slots = min(2, len(novel), max(1, s.top_k - 1))
            retain = s.top_k - new_slots
            chunks = _dedupe_chunks(originals[:retain] + novel[:new_slots] + originals[retain:])[:s.top_k]
        record(refusal_reason="unsupported_answer")
        return AskResponse(answer=IDK, sources=[])
