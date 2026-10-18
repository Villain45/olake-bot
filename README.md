---
title: Ask OLake
sdk: docker
app_port: 7860
---

# Ask OLake

RAG chat assistant over OLake documentation and repositories.
Contract: `POST /ask {"question": "..."} -> {"answer": "...", "sources": ["<url>", ...]}`.

## Architecture

    question -> hybrid search (Qdrant: dense + BM25, RRF) -> FlashRank rerank (top 6)
             -> low confidence? -> one wider local search -> trusted-domain web search + rerank
             -> validated LLM JSON -> passage-linked answer or one bounded repair
             -> {answer, sources}

CPU models are baked into the Docker image. Qdrant stores dense and BM25 vectors;
FlashRank reranks retrieved passages. Positive answers require valid passage citations,
and insufficient evidence produces a refusal. Requests have caching, rate limiting and
bounded concurrency. Pasted code and configuration retain indentation and letter case.

Retrieval uses a compact query of at most 96 reranker tokens when a question contains a
large configuration dump; the answer model still receives the original question. Setup
questions can use up to four relevant sections from one page. Official documentation is
preferred for close relevance ties, known mirrors are deduplicated, and repository agent
instruction files are excluded from answer evidence. These are query-time changes and
use the existing index without rebuilding it.

At most one wider retrieval and one answer repair are attempted within the request
deadline. Optional recovery failures retain usable evidence; a late web fetch timeout
keeps passages already fetched. Missing setup choices can produce a clarification
question. Inline passage citations are resolved to source links by the server, with
checks for unknown IDs, invented URLs and unsupported configuration keys or flags.
These checks catch specific failures; they do not prove every factual claim is correct.

`RETRIEVAL_RECOVERY_ENABLED` and `ANSWER_RECOVERY_ENABLED` default to `true`;
`RECOVERY_MIN_REMAINING_S` defaults to `8`. `MIN_RERANK_SCORE` remains `0.25` pending
reviewed calibration. Restart the API to load runtime changes; no index command is needed.

## Setup

    uv sync --frozen

Copy `.env.example` to `.env` and fill in the keys (`Copy-Item .env.example .env` in
PowerShell, or `cp .env.example .env` on Linux/macOS). Keep `.env` local.

The template contains the three connection values needed for setup. The app loads
`.env`; optional settings and defaults are defined in `app/config.py`. To override a
setting, add its uppercase name to `.env`, for example `TOP_K=8`.
List values use JSON arrays. LLM routing is restricted to the three models below;
legacy `LLM_MODEL` and `LLM_FALLBACK_MODELS` overrides are normalized to this order.
Process environment variables take precedence over `.env`.

Run ingestion when ready to fetch and publish an index:

    uv run python -m ingest.crawl docs
    uv run python -m ingest.crawl repos
    uv run python -m ingest.build_index

Start the API after an index exists:

    uv run uvicorn app.main:app --reload --port 8080

Set `QDRANT_URL` to your Qdrant Cloud HTTPS endpoint and `QDRANT_API_KEY` to its
database API key. The API and indexer share these settings. Supply both as
environment variables when deploying. `COLLECTION` defaults to `olake_docs`.
`OPENROUTER_API_KEY` is needed for answers. The supplied key permits exactly these
models, tried in this priority order:

1. `google/gemini-2.5-flash-lite` (primary)
2. `z-ai/glm-4.7-flash`
3. `deepseek/deepseek-v4.1-flash`

Gemini is tried first to avoid waiting for a slow GLM attempt before reaching it.
The application enforces this order even if `.env` still contains the previous
GLM-first settings. Restart a running API to load the updated order.
`GET /models` returns `{"models": ["...", "...", "..."], "default": "google/gemini-2.5-flash-lite"}`.
The API moves to the next allowed model for model-specific errors (including 403),
timeouts and invalid output. JSON-mode compatibility is handled per model. A grammar
instruction improves the prose while preserving commands, identifiers and facts.
Invalid output receives controlled correction feedback on the next attempt, and
truncated output gets a bounded increase in its output-token allowance.
Changing the answer model does not change embeddings, Qdrant collections or the built
index; no reindexing is needed.

To inspect the completed index and try answers, start the API above and visit
`http://localhost:8080/docs`. Check `GET /models` and `GET /readyz`, then try `POST /ask`
with `{"question": "What is OLake Fusion and what does it do?"}`. Readiness reports the
active index version and chunk count. It performs read-only queries, not an LLM call.
Inspect the returned answer and source links before running the evaluation commands.

## Rebuilding safely

The API reads the `{COLLECTION}_current` alias when present, and falls back to the
original `COLLECTION` for existing installations. A rebuild writes a separate collection,
checks the completed index, and switches the alias only after success. Existing
collections remain available if a build fails. Build output reports progress and the
published collection; local embedding caches allow repeated work to be reused.

The indexer prints and immediately flushes timestamped stage updates, elapsed-time
heartbeats during long operations, and batch progress with completed chunks, percentage,
embedding cache reuse, speed and ETA. Warnings and retry failures appear on stderr;
fatal errors include a full traceback and return a nonzero exit code. These console
updates apply to newly started builds; an already running process keeps its loaded code.

If an older index build is already running while this code is updated, let it finish.
That process uses the implementation it loaded when it started. The next build uses the
new chunking and safe publication flow. Avoid overlapping index builds. Improvements to
chunking, metadata, duplicate handling and indexed file types take effect after rebuilding.

The raw documentation and repository snapshots live under `data/`. Refresh them with
the crawl commands before rebuilding when newer source content is needed. A deployment
does not automatically crawl or rebuild on startup.

## Scheduled refresh

The `Refresh OLake index` workflow in `.github/workflows/refresh-index.yml` fetches
documentation, updates the five source repositories, then runs the existing indexer.
Its `ingest.refresh_index` wrapper performs retention after successful publication.
It is scheduled daily at **03:17 UTC / 08:47 IST**. GitHub may delay scheduled runs;
the workflow must be on the repository's default branch.

Automatic refresh is **disabled by default** so an existing local index build can
finish. This workflow does not start on a push and does not modify the indexer.
GitHub workflow concurrency prevents its own builds from overlapping, but cannot
detect a build running on your laptop or another service. Finish any such build
before enabling automation or starting a manual refresh.

In GitHub, open **Settings > Secrets and variables > Actions** and configure:

- Repository secrets `QDRANT_URL` and `QDRANT_API_KEY`, using the Cloud connection
  values from your local `.env`. The indexer does not need an OpenRouter key.
- Repository variable `QDRANT_COLLECTION`, matching your API's `COLLECTION`
  (default `olake_docs`).
- After the current build finishes and this workflow is pushed to the default branch,
  set repository variable `ENABLE_SCHEDULED_REFRESH` to `true` to enable daily runs.
  Set it to `false` to pause future automatic runs; it does not stop an active run.

To run once after local indexing finishes, open **Actions > Refresh OLake index >
Run workflow**, select the default branch, and confirm that no other indexer is
running. Manual runs work while the automatic schedule is disabled. View crawl and
batch progress in that run's logs. A failed crawl prevents publication, and the
indexer's existing validation and alias switch preserve the current index on failure.

Models, source snapshots and the local embedding database are cached between GitHub
runs. Each build saves updated cache state, including after ordinary step failures;
cache eviction or a forced termination can require recomputing embeddings. The first
GitHub run has its own empty cache and does not reuse your laptop's cache. Keep one
scheduled publisher per Qdrant collection.

After a successful build, the wrapper points `{COLLECTION}_previous` at the formerly
current index, keeping one rollback alongside `{COLLECTION}_current`. It deletes only
the displaced previous index, after checking that no alias still references it. Failed
builds do not run retention; an unchanged published index keeps its existing rollback.
Deletion failures leave the new index available and are retried from a small retention
ledger saved alongside the embedding cache. A cleanup failure marks the workflow failed
without undoing publication. Do not remove that ledger if a cleanup retry is needed.

Retention does not discover candidates by listing collection names or timestamps.
Untracked historical collections, incomplete staging builds, other collections and
targets of other aliases remain untouched. An existing legacy `COLLECTION` can become
the rollback when it was serving traffic before a successful refresh; it is eligible
for deletion only after a later refresh replaces it in the rollback slot.

Direct `python -m ingest.build_index` runs, including a build already in progress,
remain unchanged and do not run cleanup. To build with retention manually after all
other builders finish, use `uv run python -m ingest.refresh_index`. For a manual
rollback, atomically swap the targets of the `_current` and `_previous` aliases in
Qdrant while refresh jobs are paused; both collections remain available.

## Checks and evaluation

Run regression checks without calling Qdrant, OpenRouter or web search:

    uv run python -m unittest discover -s tests -v

The 48-case evaluation covers ingestion, configuration, Fusion, refusals and injection
attempts. It runs only when explicitly invoked against an already running API, and can
consume OpenRouter credits:

    uv run python eval/run_eval.py http://localhost:8080 --limit 5 --output eval/results.json
    uv run python eval/run_eval.py http://localhost:8080 --output eval/results.json

The runner exits nonzero for HTTP/response errors or failed measured quality checks.
Unknown expectations remain unmeasured. Source checks recognize the same repository
file at a commit or on `main`, and known OLake documentation mirrors, while retaining
exact page matching.

It records HTTP errors, p50/p95 latency, expected-source matches, keyword matches and a
refusal-wording proxy for answerability. Expectations come from the local documentation
snapshot; `null` expectations need manual review. These checks are signals for review,
not a factual-correctness or citation-support score. Errors are reported separately from
successful-response proxy rates. Cost stays unknown unless the API explicitly reports it;
the standard `{answer, sources}` response does not expose cost. Latency is reported
overall, by category, and separately for cache hits and generation requests when response
headers are available. `--interval` defaults to 3.1 seconds between requests to respect the default
20/minute rate limit. Use a lower value only when your API limit permits it.

`/ask` includes diagnostic headers without changing the JSON contract: `X-Cache`
(`MISS`, `HIT` or `COALESCED`), `X-Index-Version`, `X-Retrieval-Score` when available,
`X-Recovery-Attempted`, `X-Answer-Model` when used, and `X-Refusal-Reason` for refusals
or clarifications. The score is the highest observed reranker score, not a factual
confidence probability. Cached answers preserve the diagnostics from their generation.

To export scores and perform an offline threshold sweep:

    uv run python eval/run_eval.py http://localhost:8080 --output eval/results.json --score-output eval/scores.jsonl
    uv run python eval/run_eval.py --calibrate eval/scores.jsonl --output eval/calibration.json

Exported labels are provisional case-answerability proxies. Review whether the actual
retrieved evidence supports each answer before using these labels to tune the gate.
Calibration accepts reviewed JSON/JSONL records with `score` and boolean `answerable`,
reports precision/recall/F1, and recommends a threshold only when both classes exist.
It makes no API calls and changes no settings or index data. Validate a proposed
threshold on held-out cases before changing the default.

Review failing cases before changing `TOP_K`, `CANDIDATES` or `MIN_RERANK_SCORE`.
No production latency, answer-quality or cost benchmark is claimed until the live suite
has been run and its answers reviewed.

## Deploy

    docker build -t olake-ask .
    docker run -p 8080:7860 --env-file .env olake-ask

The container defaults to port 7860; set `PORT` if your host requires another port.
Dependencies are installed from `uv.lock`, and models are downloaded during image build.
The app runs as a non-root user with write access to `/models`. Model downloads require
network access at build time. The API image includes `app/`; there is no widget source in
this repository. A separately built widget can be mounted at `/app/widget/dist` to use
the optional `/widget` static route.

`GET /healthz` is a liveness check. Use `GET /readyz` for routing readiness: it returns
503 when the configured index is unavailable, empty or incompatible. Readiness does not
make a billable LLM request. Keep the API close to Qdrant and configure a persistent
service if cold-start latency matters. Schedule crawling and rebuilding separately if
automatic freshness is required.
