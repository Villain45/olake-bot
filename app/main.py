import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from cachetools import TTLCache
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from .config import get_settings
from .diagnostics import request_trace
from .llm import LLM
from .rag import RAG
from .retrieval import Retriever
from .schemas import AskRequest, AskResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("olake.api")

settings = get_settings()
# TODO: behind a proxy, use X-Forwarded-For so the limiter sees real client IPs
limiter = Limiter(key_func=get_remote_address)
cache: TTLCache = TTLCache(maxsize=1024, ttl=settings.cache_ttl_s)
refusal_cache: TTLCache = TTLCache(maxsize=1024, ttl=settings.refusal_cache_ttl_s)
trace_cache: TTLCache = TTLCache(maxsize=2048, ttl=max(settings.cache_ttl_s, settings.refusal_cache_ttl_s))
inflight: dict[tuple, asyncio.Task] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    retriever = Retriever(settings)
    llm = LLM(settings)
    app.state.retriever = retriever
    app.state.rag = RAG(settings, retriever, llm)
    cache.clear()
    refusal_cache.clear()
    trace_cache.clear()
    inflight.clear()
    try:
        yield
    finally:
        pending = tuple(inflight.values())
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        await llm.aclose()
        await retriever.aclose()


app = FastAPI(title="Ask OLake", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
    expose_headers=["X-Cache", "X-Index-Version", "X-Retrieval-Score", "X-Recovery-Attempted", "X-Answer-Model", "X-Refusal-Reason"],
)


@app.get("/version")
async def version():
    """Read Render's commit and running instance identity for each request."""
    return {
        "commit": os.getenv("RENDER_GIT_COMMIT"),
        "deployment": os.getenv("RENDER_INSTANCE_ID"),
    }


@app.get("/healthz")
async def healthz():
    """Liveness is independent of remote services; /readyz checks readiness."""
    return {"status": "ok", "llm_model": settings.llm_model}


@app.get("/models")
async def models():
    """Only expose the models permitted by this deployment's OpenRouter key."""
    return {"models": list(settings.llm_models), "default": settings.llm_models[0]}


@app.get("/readyz")
async def readyz():
    try:
        if not settings.openrouter_api_key.strip():
            raise ValueError("OPENROUTER_API_KEY is not configured")
        result = await asyncio.wait_for(
            app.state.retriever.check_readiness(),
            timeout=settings.readiness_timeout_s,
        )
        return {**result, "llm_model": settings.llm_model}
    except Exception as exc:  # noqa: BLE001
        log.warning("readiness failed: %s", type(exc).__name__)
        raise HTTPException(status_code=503, detail="Assistant is not ready: check index and configuration.") from None


async def _produce_answer(key: tuple, question: str, product: str | None, timeout: float) -> AskResponse:
    started = time.perf_counter()
    work = app.state.rag.ask(question, product=product) if product else app.state.rag.ask(question)
    response = await asyncio.wait_for(work, timeout=timeout)
    (cache if response.sources else refusal_cache)[key] = response
    trace_cache[key] = dict(request_trace.get() or {})
    log.info("ask seconds=%.3f sources=%d index=%s", time.perf_counter() - started, len(response.sources), key[0])
    return response


def _response_headers(response: Response, key: tuple, cache_status: str) -> None:
    trace = trace_cache.get(key, {})
    response.headers["X-Cache"] = cache_status
    response.headers["X-Index-Version"] = key[0]
    response.headers["X-Recovery-Attempted"] = str(bool(trace.get("retrieval_recovered") or trace.get("answer_recovered"))).lower()
    if "top_score" in trace:
        response.headers["X-Retrieval-Score"] = f"{trace['top_score']:.6f}"
    if trace.get("answer_model"):
        response.headers["X-Answer-Model"] = trace["answer_model"]
    if trace.get("refusal_reason"):
        response.headers["X-Refusal-Reason"] = trace["refusal_reason"]


@app.post("/ask", response_model=AskResponse)
@limiter.limit(settings.rate_limit)
async def ask(request: Request, body: AskRequest, response: Response) -> AskResponse:
    q = body.question
    if len(q) > settings.max_question_chars:
        raise HTTPException(status_code=422, detail=f"Question exceeds {settings.max_question_chars} characters.")
    deadline = time.monotonic() + settings.request_timeout_s
    trace_token = request_trace.set({})
    try:
        async with asyncio.timeout(settings.request_timeout_s):
            version = await app.state.retriever.current_index()
            key = (version, settings.llm_models, body.product, q)
            if key in cache:
                _response_headers(response, key, "HIT")
                return cache[key]
            if key in refusal_cache:
                _response_headers(response, key, "HIT")
                return refusal_cache[key]
            task = inflight.get(key)
            cache_status = "COALESCED" if task is not None else "MISS"
            if task is None:
                task = asyncio.create_task(_produce_answer(key, q, body.product, max(0.001, deadline - time.monotonic())))
                inflight[key] = task

                def finished(worker):
                    if inflight.get(key) is worker:
                        inflight.pop(key, None)
                    if not worker.cancelled():
                        worker.exception()

                task.add_done_callback(finished)
            # asyncio.wait leaves shared work alive when an individual waiter
            # disconnects, without an abandoned shield future logging failures.
            await asyncio.wait({task})
            result = task.result()
            _response_headers(response, key, cache_status)
            return result
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Assistant timed out, please retry.") from None
    except Exception as exc:  # noqa: BLE001
        # Validation errors may contain model output echoing a pasted configuration.
        log.error("ask failed: %s", type(exc).__name__)
        raise HTTPException(status_code=503, detail="Assistant temporarily unavailable, please retry.") from None
    finally:
        request_trace.reset(trace_token)


# Serve the built embeddable widget (see widget/README.md) at /widget/ask-olake.js
_dist = Path(__file__).resolve().parent.parent / "widget" / "dist"
if _dist.exists():
    app.mount("/widget", StaticFiles(directory=_dist), name="widget")
