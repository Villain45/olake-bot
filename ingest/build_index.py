"""Chunk, cache embeddings, and publish a validated index through a Qdrant alias.

Run ``python -m ingest.build_index``. Interrupted builds reuse embeddings and the
staging collection. Existing collections are never deleted by this command.
"""
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from contextlib import contextmanager
from pathlib import Path

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import models as m

from app.config import get_settings
from app.qdrant import make_client
from ingest.chunk import chunk_markdown, embedding_text

RAW = Path("data/raw/docs.jsonl")
REPO_DIR = Path("data/repos")
SKIP_DIRS = {"node_modules", ".git", "vendor", "dist", "build", ".venv"}


def report(message: str, level: str = "INFO") -> None:
    """Flush each message so progress remains visible even with redirected output."""
    level = level.upper()
    stream = sys.stderr if level in {"WARN", "WARNING", "ERROR"} else sys.stdout
    print(f"{datetime.now().strftime('%H:%M:%S')} [{level}] {message}", file=stream, flush=True)


@contextmanager
def stage(label: str, heartbeat_s: float = 10.0):
    """Keep long model/network operations visible without hiding exceptions."""
    started = time.perf_counter()
    stopped = threading.Event()
    report(f"{label}: started")

    def heartbeat():
        while not stopped.wait(heartbeat_s):
            report(f"{label}: still running, elapsed {time.perf_counter() - started:.1f}s")

    worker = threading.Thread(target=heartbeat, daemon=True)
    worker.start()
    try:
        yield
    except BaseException as exc:
        stopped.set()
        report(f"{label}: failed after {time.perf_counter() - started:.1f}s: {exc}", "ERROR")
        raise
    else:
        stopped.set()
        report(f"{label}: completed in {time.perf_counter() - started:.1f}s")
    finally:
        stopped.set()
        worker.join()


def product_of(path: str) -> str:
    return "fusion" if "fusion" in path.lower() else "ingestion"


def docs_chunks(**chunk_options) -> list[dict]:
    out = []
    if not RAW.exists():
        report(f"Documentation input not found: {RAW}; checking repository sources instead", "WARNING")
        return out
    lines = [line for line in RAW.read_text(encoding="utf-8").splitlines() if line.strip()]
    report(f"Documentation pages: 0/{len(lines)}")
    for number, line in enumerate(lines, start=1):
        record = json.loads(line)
        for chunk in chunk_markdown(record["text"], record["url"], record["title"], "docs",
                                    product_of(record["url"]), **chunk_options):
            chunk["fetched_at"] = record.get("fetched_at")
            out.append(chunk)
        if number % 10 == 0 or number == len(lines):
            report(f"Documentation pages: {number}/{len(lines)}; {len(out)} chunks")
    return out


def selected_config(path: Path, rel: str) -> bool:
    if path.suffix.lower() not in {".yaml", ".yml", ".json"}:
        return False
    lowered = rel.lower()
    return any(word in lowered for word in ("example", "sample", "config", "values", "schema", "spec", "connector"))


def repo_chunks(**chunk_options) -> list[dict]:
    out = []
    repos = sorted(repo for repo in REPO_DIR.glob("*") if repo.is_dir())
    report(f"Repository sources: {len(repos)} repositories")
    for number, repo in enumerate(repos, start=1):
        report(f"Repository {number}/{len(repos)}: {repo.name}; scanning sources")
        processed, before = 0, len(out)
        commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
        for file in sorted(repo.rglob("*")):
            if not file.is_file() or SKIP_DIRS & set(file.relative_to(repo).parts):
                continue
            rel = file.relative_to(repo).as_posix()
            is_config = selected_config(file, rel)
            if file.suffix.lower() not in {".md", ".mdx"} and not is_config:
                continue
            if file.stat().st_size > 1_000_000:
                report(f"Skipping oversized source {repo.name}/{rel}", "WARNING")
                continue
            text = file.read_text(encoding="utf-8", errors="replace")
            if is_config:
                language = "json" if file.suffix.lower() == ".json" else "yaml"
                text = f"```{language}\n{text}\n```"
            url = f"https://github.com/datazip-inc/{repo.name}/blob/{commit}/{rel}"
            for chunk in chunk_markdown(text, url, f"{repo.name}/{rel}", "repo",
                                        product_of(f"{repo.name}/{rel}"), **chunk_options):
                chunk["source_commit"] = commit
                out.append(chunk)
            processed += 1
            if processed % 25 == 0:
                report(f"Repository {repo.name}: {processed} files processed; {len(out) - before} chunks")
        report(f"Repository {number}/{len(repos)} complete: {repo.name}; {processed} files; {len(out) - before} chunks")
    return out


def deduplicate(chunks: list[dict]) -> list[dict]:
    """Prefer documentation for identical product-specific passages."""
    seen, out = set(), []
    for chunk in chunks:
        key = (chunk["product"], chunk["text"].replace("\r\n", "\n").strip())
        if key not in seen:
            seen.add(key)
            out.append(chunk)
    return out


def token_counter(dense):
    """Clone the loaded tokenizer so counting never truncates model inputs."""
    tokenizer = getattr(getattr(dense, "model", None), "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("Embedding model has no tokenizer; cannot verify chunk token limits")
    from tokenizers import Tokenizer
    counter = Tokenizer.from_str(tokenizer.to_str())
    counter.no_truncation()
    counter.no_padding()
    return lambda text: len(counter.encode(text).ids)


def _retry(operation, attempts: int = 3, *, description: str = "Qdrant operation"):
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as exc:
            if attempt + 1 == attempts:
                report(f"{description}: attempt {attempt + 1}/{attempts} failed; no retries left: {exc}", "ERROR")
                raise
            delay = 2 ** attempt
            report(f"{description}: attempt {attempt + 1}/{attempts} failed: {exc}; retrying in {delay}s", "WARNING")
            time.sleep(delay)


class EmbeddingCache:
    def __init__(self, path, models):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self.models = models

    def key(self, text):
        return hashlib.sha256(f"{self.models}\n{text}".encode()).hexdigest()

    def vectors(self, texts, dense, sparse, batch_size):
        values, missing = {}, []
        for text in texts:
            key = self.key(text)
            row = self.db.execute("SELECT value FROM embeddings WHERE key = ?", (key,)).fetchone()
            if row:
                values[key] = json.loads(row[0])
            elif key not in missing:
                missing.append(key)
        by_key = {self.key(text): text for text in texts}
        report(f"Embedding cache: {len(texts) - len(missing)}/{len(texts)} reused; {len(missing)} new embeddings needed")
        if missing:
            inputs = [by_key[key] for key in missing]
            with stage(f"Dense embeddings ({len(inputs)} texts)"):
                dense_vecs = list(dense.embed(inputs, batch_size=batch_size))
                if len(dense_vecs) != len(inputs):
                    raise RuntimeError(f"Dense embedding model returned {len(dense_vecs)}/{len(inputs)} vectors")
            with stage(f"Sparse embeddings ({len(inputs)} texts)"):
                sparse_vecs = list(sparse.embed(inputs, batch_size=batch_size))
                if len(sparse_vecs) != len(inputs):
                    raise RuntimeError(f"Sparse embedding model returned {len(sparse_vecs)}/{len(inputs)} vectors")
            for key, dvec, svec in zip(missing, dense_vecs, sparse_vecs):
                value = {"dense": dvec.tolist(), "bm25": {
                    "indices": svec.indices.tolist(), "values": svec.values.tolist()}}
                self.db.execute("INSERT OR REPLACE INTO embeddings VALUES (?, ?)", (key, json.dumps(value)))
                values[key] = value
            self.db.commit()
        return [values[self.key(text)] for text in texts], len(missing)

    def close(self):
        self.db.close()


@contextmanager
def build_lock(path):
    """Reject simultaneous builders sharing this local checkpoint directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if not handle.tell():
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("Another index build is using this cache; wait for it to finish") from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def publish_index(client, chunks, dense, sparse, settings) -> str:
    report(f"Acquiring build lock; embedding cache: {settings.index_cache_path}")
    with build_lock(f"{settings.index_cache_path}.build.lock"):
        return _publish_index(client, chunks, dense, sparse, settings)


def _publish_index(client, chunks, dense, sparse, settings) -> str:
    if not chunks:
        raise ValueError("No source chunks; refusing to publish an empty index")
    model_key = f"v2:{settings.embed_model}:{settings.sparse_model}:{settings.embed_dim}"
    digest = hashlib.sha256(json.dumps([model_key, chunks], sort_keys=True).encode()).hexdigest()[:20]
    staging = f"{settings.collection}_build_{digest}"
    alias = f"{settings.collection}_current"
    report(f"Preparing {len(chunks)} chunks; staging collection: {staging}; publication alias: {alias}")
    with stage("Checking published collection aliases"):
        aliases = {item.alias_name: item.collection_name for item in client.get_aliases().aliases}
    if aliases.get(alias) == staging:
        with stage("Validating the already-published index"):
            if client.count(staging, exact=True).count != len(chunks):
                raise RuntimeError("Published collection count is invalid; refusing to update it in place")
        report(f"100.0% complete: index {staging} is already current ({len(chunks)} chunks)")
        return staging
    with stage("Checking staging collection"):
        exists = client.collection_exists(staging)
    if not exists:
        with stage(f"Creating staging collection {staging}"):
            client.create_collection(
                staging,
                vectors_config={"dense": m.VectorParams(size=settings.embed_dim, distance=m.Distance.COSINE)},
                sparse_vectors_config={"bm25": m.SparseVectorParams(modifier=m.Modifier.IDF)},
            )
    else:
        with stage("Validating existing staging collection schema"):
            params = client.get_collection(staging).config.params
            vector = params.vectors.get("dense") if isinstance(params.vectors, dict) else None
            sparse_params = params.sparse_vectors.get("bm25") if params.sparse_vectors else None
            if (vector is None or vector.size != settings.embed_dim or vector.distance != m.Distance.COSINE
                    or sparse_params is None or sparse_params.modifier != m.Modifier.IDF):
                raise RuntimeError("Existing staging collection has an incompatible vector schema")
    with stage("Preparing product payload index"):
        client.create_payload_index(staging, "product", m.PayloadSchemaType.KEYWORD, wait=True)
    size = max(1, settings.index_batch_size)
    cache = EmbeddingCache(settings.index_cache_path, model_key)
    indexed_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    total_batches = (len(chunks) + size - 1) // size
    try:
        for start in range(0, len(chunks), size):
            batch = chunks[start:start + size]
            number = start // size + 1
            report(f"Batch {number}/{total_batches}: starting chunks {start + 1}-{start + len(batch)}/{len(chunks)}")
            vectors, computed = cache.vectors([embedding_text(chunk) for chunk in batch], dense, sparse, size)
            points = [m.PointStruct(id=start + offset, vector=vector,
                                    payload={**chunk, "indexed_at": indexed_at, "index_version": digest,
                                             "embed_model": settings.embed_model,
                                             "sparse_model": settings.sparse_model,
                                             "embed_dim": settings.embed_dim})
                      for offset, (chunk, vector) in enumerate(zip(batch, vectors))]
            with stage(f"Upload batch {number}/{total_batches}"):
                _retry(lambda: client.upsert(staging, points=points, wait=True),
                       description=f"Upload batch {number}/{total_batches}")
            completed = start + len(batch)
            elapsed = max(time.perf_counter() - started, 0.001)
            rate = completed / elapsed
            report(f"Batch {number}/{total_batches} complete: {completed}/{len(chunks)} chunks "
                   f"({completed / len(chunks):.1%}); {computed} new, {len(batch) - computed} cached; "
                   f"elapsed {elapsed:.1f}s; {rate:.1f} chunks/s; ETA {(len(chunks) - completed) / rate:.1f}s")
        with stage("Validating uploaded chunk count"):
            actual = client.count(staging, exact=True).count
        if actual != len(chunks):
            raise RuntimeError(f"Staging count {actual} does not match expected {len(chunks)}; old index remains current")
        operations = []
        if alias in aliases:
            operations.append(m.DeleteAliasOperation(delete_alias=m.DeleteAlias(alias_name=alias)))
        operations.append(m.CreateAliasOperation(create_alias=m.CreateAlias(collection_name=staging, alias_name=alias)))
        # Qdrant applies all alias operations atomically. Never drop a legacy collection.
        with stage(f"Publishing alias {alias}"):
            client.update_collection_aliases(change_aliases_operations=operations)
    finally:
        cache.close()
    report(f"Published {len(chunks)} chunks as {alias} -> {staging}")
    return staging


def main() -> None:
    started = time.perf_counter()
    report("Index build starting")
    with stage("Loading configuration and creating Qdrant client"):
        settings = get_settings()
        client = make_client(settings, timeout=60)
    try:
        # Fail before expensive CPU embedding work when credentials or network are wrong.
        with stage("Checking Qdrant connectivity"):
            client.get_collections()
        with stage(f"Loading dense model {settings.embed_model}"):
            dense = TextEmbedding(settings.embed_model, cache_dir=settings.model_cache_dir)
        with stage(f"Loading sparse model {settings.sparse_model}"):
            sparse = SparseTextEmbedding(settings.sparse_model, cache_dir=settings.model_cache_dir)
        options = {"token_counter": token_counter(dense), "max_tokens": settings.index_max_tokens}
        with stage("Chunking documentation"):
            docs = docs_chunks(**options)
        with stage("Chunking repository sources"):
            repos = repo_chunks(**options)
        with stage("Deduplicating chunks"):
            chunks = deduplicate(docs + repos)
        report(f"Sources ready: {len(docs)} documentation chunks + {len(repos)} repository chunks; "
               f"{len(chunks)} unique chunks; {len(docs) + len(repos) - len(chunks)} duplicates removed")
        publish_index(client, chunks, dense, sparse, settings)
    finally:
        original_error = sys.exc_info()[1]
        try:
            with stage("Closing Qdrant client"):
                client.close()
        except Exception:
            if original_error is None:
                raise
            # A cleanup failure must not replace the error that stopped the build.
            traceback.print_exc(file=sys.stderr)
    report(f"Index build finished successfully in {time.perf_counter() - started:.1f}s")


def cli() -> int:
    try:
        main()
    except KeyboardInterrupt:
        report("Index build interrupted; saved embeddings can be reused on the next run", "ERROR")
        return 130
    except Exception as exc:
        report(f"Index build failed: {exc}", "ERROR")
        if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
            report(str(exc.stderr).strip(), "ERROR")
        traceback.print_exc(file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
