---
title: Ask OLake
sdk: docker
app_port: 7860
---

<p align="center">
  <img src="docs/assets/ask-olake-banner.svg" alt="Ask OLake — from documentation to answers" width="100%">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&amp;logo=python&amp;logoColor=white" alt="Python 3.11 or newer">
  <img src="https://img.shields.io/badge/FastAPI-009688?style=flat-square&amp;logo=fastapi&amp;logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Qdrant_Cloud-DC244C?style=flat-square" alt="Qdrant Cloud">
  <img src="https://img.shields.io/badge/OpenRouter-6366F1?style=flat-square" alt="OpenRouter">
</p>

<p align="center">
  <strong>Ask about OLake. Get an answer you can trace.</strong><br>
  A documentation assistant for ingestion, CDC, and Iceberg table maintenance.
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#try-it">Try it</a> ·
  <a href="#api-at-a-glance">API</a> ·
  <a href="docs/OPERATIONS.md#deploy">Deploy</a> ·
  <a href="docs/OPERATIONS.md">Operations guide</a>
</p>

---

| 🔎 Find the right context | 🔗 Follow the sources | 🔄 Keep docs fresh |
| --- | --- | --- |
| Hybrid search across OLake docs and repositories. | Answers link back to supporting passages. | Optional GitHub Actions refresh every three hours. |

## Quick start

**You'll need:** Python 3.11+, [uv](https://docs.astral.sh/uv/), Git, Qdrant Cloud, and an OpenRouter key.

For a new setup, copy [.env.example](.env.example) to `.env` and fill in
`OPENROUTER_API_KEY`, `QDRANT_URL`, and `QDRANT_API_KEY`. Keep an existing configured `.env`.

With an index available in your configured Qdrant collection, run:

```sh
uv sync --frozen
uv run uvicorn app.main:app --reload --port 8080
```

Open **[localhost:8080/docs](http://localhost:8080/docs)** to try the API.

<details>
<summary><strong>First setup? Build an index</strong></summary>

Install dependencies with `uv sync --frozen`, then run these commands before starting
the API. Skip this step if you already have a usable index.

```sh
uv run python -m ingest.crawl docs
uv run python -m ingest.crawl repos
uv run python -m ingest.build_index
```

Let any existing build finish before starting another. Your first local run also
downloads the CPU models.

</details>

## Try it

Ask a question from PowerShell:

```powershell
Invoke-RestMethod -Uri "http://localhost:8080/ask" `
  -Method Post -ContentType "application/json" `
  -Body '{"question":"How do I configure a Postgres source for CDC?"}'
```

The response contains **`answer`** and **`sources`**. If a setup choice is missing,
the assistant can ask a clarification; if the evidence is insufficient, it declines to guess.

## API at a glance

| Route | What you get |
| --- | --- |
| `POST /ask` | An answer with source links. |
| `GET /readyz` | Index and configuration readiness. |
| `GET /models` | Available models in fallback order. |
| `GET /version` | Render commit and running instance identifiers. |

<p align="center">
  <sub>OLake documentation → relevant evidence → answers with sources.</sub>
</p>
