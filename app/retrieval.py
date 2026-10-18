import asyncio
import hashlib
import logging
import re
import threading
import time
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from urllib.parse import unquote, urlsplit

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import models as m
from tokenizers import Tokenizer

from .config import Settings
from .inference import CpuRanker as Ranker, rerank_batched
from .models import Chunk
from .qdrant import make_client

log = logging.getLogger("olake.retrieval")

QUERY_TOKEN_BUDGET = 96
_INSTRUCTIONS = {"agents.md", "claude.md"}
_CONFIG_QUESTION = re.compile(
    r"\b(configur\w*|setup|set up|deploy\w*|install\w*|prerequisites?|permissions?|examples?|steps?)\b|\bhow (?:do i|to)\b",
    re.I,
)
_ERROR = re.compile(
    r"\b(error|exception|fatal|permission denied|does not exist|cannot|could not|invalid|failed|sqlstate)\b", re.I,
)


@contextmanager
def _stage(name: str):
    started = time.perf_counter()
    log.info("retrieval stage=%s started", name)
    try:
        yield
    finally:
        log.info("retrieval stage=%s seconds=%.3f", name, time.perf_counter() - started)


def _token_count(text: str, tokenizer: Tokenizer | None) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False).ids) if tokenizer else len(text.encode("utf-8"))


def _truncate_query(text: str, budget: int, tokenizer: Tokenizer | None) -> str:
    if not tokenizer:
        return text.encode("utf-8")[:budget].decode("utf-8", errors="ignore").strip()
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if len(encoded.ids) <= budget:
        return text.strip()
    cut = encoded.offsets[budget - 1][1]
    result = text[:cut].strip()
    # Re-tokenizing a cut word may produce more pieces than the original prefix.
    while result and _token_count(result, tokenizer) > budget:
        result = result[:-1].rstrip()
    return result


def retrieval_query(question: str, tokenizer: Tokenizer | None = None, *, recovery: bool = False) -> str:
    """Keep intent and diagnostic identifiers within the reranker's query budget.

    The original question is never modified; this string is only for search/ranking.
    The supplied tokenizer must have padding and truncation disabled.
    """
    compact = " ".join(question.split())
    if _token_count(compact, tokenizer) <= QUERY_TOKEN_BUDGET:
        query = compact
    else:
        prose, errors, keys = [], [], []
        fenced = False
        for line in question.splitlines():
            line = line.strip()
            if line.startswith(("```", "~~~")):
                fenced = not fenced
                continue
            if not line:
                continue
            if _ERROR.search(line):
                errors.append(line)
            key = re.match(r'^\s*["\']?([\w.-]+)["\']?\s*[:=]', line)
            if key and key.group(1).lower() not in {"password", "secret", "token", "api_key", "username", "host"}:
                keys.append(key.group(1))
            if not fenced and not key and line.strip(" ,{}[]") and not line.startswith(("{", "[")):
                prose.append(line)
        keys.extend(re.findall(r'["\']([\w.-]+)["\']\s*:', question))
        # Error lines and exact field names remain useful when a large config precedes the question.
        identifiers = re.findall(r"\b\w+(?:[_.-]\w+)+\b", " ".join(prose + errors))
        key_text = " ".join(dict.fromkeys(identifiers + keys))
        parts = [
            _truncate_query(" ".join(prose) or compact, 48, tokenizer),
            _truncate_query(" ".join(dict.fromkeys(errors)), 28, tokenizer),
            _truncate_query(key_text, 20, tokenizer),
        ]
        query = _truncate_query(" ".join(part for part in parts if part), QUERY_TOKEN_BUDGET, tokenizer)
    if recovery:
        query = re.sub(r"^(?:please\s+)?(?:can you\s+|could you\s+)?(?:tell me\s+)?", "", query, flags=re.I)
        aliases = ((r"\bpostgres\b", "PostgreSQL"), (r"\bcdc\b", "change data capture"), (r"\bk8s\b", "Kubernetes"))
        expansions = [expanded for pattern, expanded in aliases if re.search(pattern, query, re.I) and expanded.lower() not in query.lower()]
        if _CONFIG_QUESTION.search(question):
            expansions.append("configuration prerequisites permissions examples")
        elif _ERROR.search(question):
            expansions.append("troubleshooting")
        suffix = " ".join(expansions) or "OLake documentation"
        # Reserve space for the rewrite so a long query does not swallow it.
        query = _truncate_query(query, 68, tokenizer) + " " + suffix
    return _truncate_query(query, QUERY_TOKEN_BUDGET, tokenizer)


def canonical_page(url: str) -> str:
    """Unify known OLake docs mirrors without rewriting returned citation URLs."""
    parsed = urlsplit(url)
    host, path = (parsed.hostname or "").lower(), unquote(parsed.path).rstrip("/")
    if host in {"olake.io", "www.olake.io"}:
        return "olake.io" + path
    if host == "github.com":
        match = re.match(r"^/datazip-inc/olake-docs/blob/[^/]+/docs/(.+)\.mdx?$", path, re.I)
        if match:
            route = re.sub(r"(?:^|/)index$", "", match.group(1)).rstrip("/")
            return "olake.io/docs" + (f"/{route}" if route else "")
    return host + path


def _is_instruction(url: str) -> bool:
    return unquote(urlsplit(url).path).rsplit("/", 1)[-1].lower() in _INSTRUCTIONS


def _authority(chunk: Chunk, question: str) -> float:
    path = urlsplit(chunk.url).path.lower()
    comparing = bool(re.search(r"\b(compare|comparison|versus|vs|alternatives?)\b", question, re.I))
    if any(part in path for part in ("/blog/", "/blogs/", "/customer-stories/")):
        return 0.0 if comparing else -0.025
    if any(part in path for part in ("/comparison/", "/comparisons/", "/compare/")):
        return 0.0 if comparing else -0.025
    if canonical_page(chunk.url).startswith("olake.io/docs"):
        return 0.03 if chunk.source_type == "docs" else 0.015
    if path.endswith((".yaml", ".yml", ".json")):
        return 0.02
    return 0.0


def _mirror_duplicate(chunk: Chunk, selected: list[Chunk]) -> bool:
    """Deduplicate prose mirrors without hiding different technical instructions."""
    words = re.findall(r"\w+", chunk.text.lower())
    if len(words) < 20:
        return False

    def protected_values(text: str) -> tuple:
        text = text.replace("\r\n", "\n")
        return (
            # Code and configuration identifiers are case sensitive. Preserve
            # their punctuation and indentation, including changed YAML values.
            re.findall(r"```[^\n]*\n.*?```|~~~[^\n]*\n.*?~~~|`[^`\n]+`", text, re.S),
            re.findall(r"(?<![\w-])--[A-Za-z][A-Za-z0-9_-]*", text),
            re.findall(r"\b\w+(?:[_.-]\w+)+\b", text),
            re.findall(r"[-+]?\d+(?:\.\d+)*", text),
            re.findall(r"(?m)^\s*[\"']?[A-Za-z_][\w.-]*[\"']?\s*[:=].+$", text),
            re.findall(r"\b[A-Za-z_][\w.-]*[\"']?\s*[:=]\s*[\"']?[\w.+/-]+[\"']?", text),
        )

    technical = protected_values(chunk.text)
    for previous in selected:
        if canonical_page(previous.url) != canonical_page(chunk.url) or urlsplit(previous.url).hostname == urlsplit(chunk.url).hostname:
            continue
        other = re.findall(r"\w+", previous.text.lower())
        # A high overlap can still hide one crucial changed flag or value.
        if words == other and technical == protected_values(previous.text):
            return True
    return False


class Retriever:
    """Hybrid (dense + BM25 sparse, RRF-fused in Qdrant) then FlashRank rerank. CPU only."""

    def __init__(self, s: Settings):
        self.s = s
        self.qdrant = make_client(s)
        self.dense = TextEmbedding(s.embed_model, cache_dir=s.model_cache_dir, threads=s.inference_threads)
        self.sparse = SparseTextEmbedding(s.sparse_model, cache_dir=s.model_cache_dir, threads=s.inference_threads)
        self.ranker = Ranker(model_name=s.rerank_model, cache_dir=s.model_cache_dir, threads=s.inference_threads)
        log.info("CPU inference threads=%d rerank_batch_size=%d retrieval_concurrency=%d",
                 s.inference_threads, s.rerank_batch_size, s.retrieval_concurrency)
        tokenizer = getattr(self.ranker, "tokenizer", None)
        self._query_tokenizer = Tokenizer.from_str(tokenizer.to_str()) if tokenizer is not None else None
        if self._query_tokenizer is not None:
            self._query_tokenizer.no_truncation()
            self._query_tokenizer.no_padding()
        self._sem = asyncio.Semaphore(s.retrieval_concurrency)
        self._workers: set[asyncio.Task] = set()
        self._index_lock = threading.Lock()
        self._index_name = s.collection
        self._index_expires = 0.0
        self._probes: dict[str, asyncio.Task] = {}
        self._request_index = ContextVar("request_index", default=None)
        self._request_evidence = ContextVar("request_evidence", default=None)

    def begin_request(self) -> None:
        # Child tasks spawned by wait_for inherit this list, but other requests do not.
        self._request_evidence.set([])

    async def _probe(self, name, function):
        task = self._probes.get(name)
        if task is None or task.done():
            task = asyncio.create_task(asyncio.to_thread(function))
            self._probes[name] = task
            self._workers.add(task)

            def finished(worker):
                self._workers.discard(worker)
                if not worker.cancelled():
                    worker.exception()

            task.add_done_callback(finished)
        await asyncio.wait({task})
        return task.result()

    async def current_index(self) -> str:
        name = self._index_name if time.monotonic() < self._index_expires else await self._probe("index", self.index_version)
        # New producer tasks and to_thread inherit this request's snapshot.
        self._request_index.set(name)
        return name

    async def check_readiness(self) -> dict:
        return await self._probe("readiness", self.readiness)

    def index_version(self, *, refresh: bool = False) -> str:
        """Use published snapshots, falling back to indexes built by older versions."""
        with self._index_lock:
            if not refresh and time.monotonic() < self._index_expires:
                return self._index_name
            alias = f"{self.s.collection}_current"
            aliases = self.qdrant.get_aliases().aliases
            self._index_name = next(
                (entry.collection_name for entry in aliases if entry.alias_name == alias),
                self.s.collection,
            )
            self._index_expires = time.monotonic() + self.s.index_refresh_s
            return self._index_name

    @staticmethod
    def _diverse(chunks: list[Chunk], limit: int, question: str = "", *, min_score: float = 0.25) -> list[Chunk]:
        selected, seen, pages = [], set(), Counter()
        page_best = {}
        for chunk in chunks:
            page = canonical_page(chunk.url)
            page_best[page] = max(page_best.get(page, 0), chunk.score)
        instructional = bool(_CONFIG_QUESTION.search(question))
        # Authority only breaks close relevance ties; never alter confidence scores.
        ordered = sorted(chunks, key=lambda chunk: (chunk.score >= min_score, chunk.score + _authority(chunk, question)), reverse=True)
        for chunk in ordered:
            digest = hashlib.sha256(chunk.text.strip().encode("utf-8")).digest()
            page = canonical_page(chunk.url)
            cap = 4 if instructional and chunk.score >= max(min_score, page_best[page] * 0.65) else 2
            if not chunk.text.strip() or _is_instruction(chunk.url) or digest in seen or pages[page] >= cap or _mirror_duplicate(chunk, selected):
                continue
            selected.append(chunk)
            seen.add(digest)
            pages[page] += 1
            if len(selected) == limit:
                break
        return selected

    async def _run_worker(self, function, *args):
        queued = time.perf_counter()
        await self._sem.acquire()
        log.info("retrieval stage=queue seconds=%.3f", time.perf_counter() - queued)
        task = asyncio.create_task(asyncio.to_thread(function, *args))
        self._workers.add(task)

        def finished(worker):
            self._workers.discard(worker)
            self._sem.release()
            if not worker.cancelled():
                worker.exception()  # Consume failures even if the request timed out.

        task.add_done_callback(finished)
        # Cancellation cannot stop a CPU/network thread. Keep its slot until it finishes.
        await asyncio.wait({task})
        return task.result()

    def _retrieve_sync(self, question: str, product: str | None = None, recovery: bool = False) -> list[Chunk]:
        s = self.s
        started = time.perf_counter()
        tokenizer = getattr(self, "_query_tokenizer", None)
        query = retrieval_query(question, tokenizer, recovery=recovery)
        ranking_query = retrieval_query(question, tokenizer)
        candidates = min(100, s.candidates * 2) if recovery else s.candidates
        with _stage("embedding_dense"):
            dense_vec = next(iter(self.dense.query_embed(query))).tolist()
        with _stage("embedding_sparse"):
            sp = next(iter(self.sparse.query_embed(query)))
        sparse_vec = m.SparseVector(indices=sp.indices.tolist(), values=sp.values.tolist())

        flt = None
        if product:
            flt = m.Filter(must=[m.FieldCondition(key="product", match=m.MatchValue(value=product))])

        with _stage("qdrant"):
            res = self.qdrant.query_points(
                self._request_index.get() or self.index_version(),
                prefetch=[
                    m.Prefetch(query=dense_vec, using="dense", limit=candidates, filter=flt),
                    m.Prefetch(query=sparse_vec, using="bm25", limit=candidates, filter=flt),
                ],
                query=m.FusionQuery(fusion=m.Fusion.RRF),
                limit=candidates,
                with_payload=True,
            ).points
        res = [point for point in res if not _is_instruction(point.payload.get("url", ""))]
        if not res:
            return []

        passages = [
            {"id": i, "text": f"{p.payload['breadcrumb']}\n{p.payload['text']}"}
            for i, p in enumerate(res)
        ]
        with _stage("rerank_local"):
            ranked = rerank_batched(self.ranker, ranking_query, passages, self.s.rerank_batch_size)

        chunks: list[Chunk] = []
        for r in ranked:
            p = res[r["id"]].payload
            url = p["url"] + (f"#{p['anchor']}" if p.get("anchor") else "")
            chunks.append(
                Chunk(
                    id=0,  # assigned in the pipeline
                    url=url,
                    title=p["title"],
                    breadcrumb=p["breadcrumb"],
                    text=p["text"],
                    source_type=p["source_type"],
                    score=float(r["score"]),
                )
            )
        history = getattr(self, "_request_evidence", None)
        previous = (history.get() or []) if recovery and history is not None else []
        novel = [chunk for chunk in chunks if chunk.score >= s.min_rerank_score
                 and not any(chunk.text.strip() == old.text.strip() for old in previous)
                 and not _mirror_duplicate(chunk, previous)] if previous else []
        result = self._diverse(novel or chunks, s.top_k, question, min_score=s.min_rerank_score)
        log.info("retrieval seconds=%.3f candidates=%d selected=%d recovery=%s", time.perf_counter() - started, len(res), len(result), recovery)
        return result

    def _rerank_external_sync(self, question: str, chunks: list[Chunk]) -> list[Chunk]:
        if not chunks:
            return []
        passages = [
            {"id": i, "text": f"{chunk.title}\n{chunk.text}"}
            for i, chunk in enumerate(chunks)
        ]
        query = retrieval_query(question, getattr(self, "_query_tokenizer", None))
        with _stage("rerank_external"):
            ranked = rerank_batched(self.ranker, query, passages, self.s.rerank_batch_size)
        return self._diverse(
            [replace(chunks[item["id"]], score=float(item["score"])) for item in ranked],
            self.s.top_k,
            question,
            min_score=self.s.min_rerank_score,
        )

    async def retrieve(self, question: str, product: str | None = None, *, recovery: bool = False) -> list[Chunk]:
        if self._request_evidence.get() is None:
            self.begin_request()
        result = await self._run_worker(self._retrieve_sync, question, product, recovery)
        if not recovery:
            self._request_evidence.get()[:] = result
        return result

    async def rerank_external(self, question: str, chunks: list[Chunk]) -> list[Chunk]:
        return await self._run_worker(self._rerank_external_sync, question, chunks)

    def count(self) -> int:
        return self.qdrant.count(self.index_version(), exact=True).count

    def readiness(self) -> dict:
        name = self.index_version(refresh=True)
        info = self.qdrant.get_collection(name)
        vectors = info.config.params.vectors
        dense = vectors.get("dense") if isinstance(vectors, dict) else None
        sparse = info.config.params.sparse_vectors or {}
        if dense is None or dense.size != self.s.embed_dim or "bm25" not in sparse:
            raise ValueError("Index vectors do not match the configured embedding models")
        points = self.qdrant.count(name, exact=True).count
        if points < 1:
            raise ValueError("Index is empty")
        records, _ = self.qdrant.scroll(
            name, limit=1, with_payload=["embed_model", "sparse_model", "embed_dim"], with_vectors=False,
        )
        if records:
            payload = records[0].payload or {}
            for field in ("embed_model", "sparse_model", "embed_dim"):
                if field in payload and payload[field] != getattr(self.s, field):
                    raise ValueError(f"Index {field} does not match the configured model")
        return {"status": "ready", "indexed_chunks": points, "index_version": name}

    async def aclose(self) -> None:
        if self._workers:
            await asyncio.gather(*tuple(self._workers), return_exceptions=True)
        await asyncio.to_thread(self.qdrant.close)
