"""Exercise the installed Qdrant client locally with deterministic model doubles."""

import contextlib
import hashlib
import io
import re
import tempfile
import unittest
import warnings
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from qdrant_client import QdrantClient, models

from app.config import Settings
from app.rag import RAG
from app.retrieval import Retriever
from ingest.build_index import publish_index


class FakeDense:
    def __init__(self, *args, **kwargs):
        self.calls = 0

    @staticmethod
    def vector(text):
        lowered = text.lower()
        if "postgres" in lowered:
            return np.array([1.0, 0.0, 0.0])
        if "fusion" in lowered:
            return np.array([0.0, 1.0, 0.0])
        return np.array([0.0, 0.0, 1.0])

    def embed(self, texts, **kwargs):
        self.calls += 1
        return [self.vector(text) for text in texts]

    def query_embed(self, text):
        return iter([self.vector(text)])


class FakeSparse:
    def __init__(self, *args, **kwargs):
        self.calls = 0

    @staticmethod
    def vector(text):
        tokens = Counter(re.findall(r"[a-z]+", text.lower()))
        pairs = sorted((int(hashlib.sha256(token.encode()).hexdigest()[:7], 16), count) for token, count in tokens.items())
        return SimpleNamespace(
            indices=np.array([index for index, _ in pairs], dtype=np.int64),
            values=np.array([float(count) for _, count in pairs]),
        )

    def embed(self, texts, **kwargs):
        self.calls += 1
        return [self.vector(text) for text in texts]

    def query_embed(self, text):
        return iter([self.vector(text)])


class FakeRanker:
    def __init__(self, *args, **kwargs):
        pass

    def rerank(self, request):
        words = set(re.findall(r"[a-z]+", request.query.lower()))
        passages = [
            {**passage, "score": 0.95 if words.intersection(re.findall(r"[a-z]+", passage["text"].lower())) else 0.01}
            for passage in request.passages
        ]
        return sorted(passages, key=lambda passage: passage["score"], reverse=True)


class FakeLLM:
    async def generate_json(self, system_prompt, user_prompt, *, timeout_s=None):
        return {"answerable": True, "answer": "PostgreSQL CDC uses logical replication and a replication slot.", "used_ids": [1]}


def sample_chunks():
    return [
        {
            "url": "https://olake.io/docs/connectors/postgres/",
            "anchor": "configuration",
            "title": "PostgreSQL",
            "breadcrumb": "PostgreSQL > CDC",
            "text": "PostgreSQL CDC requires logical replication and a replication slot.",
            "source_type": "docs",
            "product": "ingestion",
        },
        {
            "url": "https://olake.io/docs/fusion/compaction/types-of-compaction/",
            "anchor": "lite-compaction",
            "title": "Fusion compaction",
            "breadcrumb": "Fusion > Compaction",
            "text": "OLake Fusion compaction merges small files into larger Iceberg data files.",
            "source_type": "docs",
            "product": "fusion",
        },
    ]


class LocalQdrantIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_publication_hybrid_retrieval_and_cited_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(
                _env_file=None,
                qdrant_url="",
                qdrant_api_key="",
                collection="integration_docs",
                embed_dim=3,
                candidates=4,
                top_k=2,
                index_batch_size=1,
                index_cache_path=str(Path(directory) / "embeddings.sqlite3"),
                web_fallback_enabled=False,
            )
            client = QdrantClient(location=":memory:")
            self.addCleanup(client.close)
            client.create_collection(
                settings.collection,
                vectors_config={"dense": models.VectorParams(size=3, distance=models.Distance.COSINE)},
                sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
            )
            client.upsert(settings.collection, [models.PointStruct(
                id=99,
                vector={"dense": [0.0, 0.0, 1.0], "bm25": models.SparseVector(indices=[99], values=[1.0])},
                payload={"text": "Original physical collection remains available."},
            )])
            dense, sparse = FakeDense(), FakeSparse()
            chunks = sample_chunks()
            with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
                warnings.filterwarnings("ignore", message="Payload indexes have no effect.*")
                published = publish_index(client, chunks, dense, sparse, settings)
            self.assertEqual(1, client.count(settings.collection, exact=True).count)
            self.assertEqual(2, client.count(f"{settings.collection}_current", exact=True).count)
            self.assertTrue(Path(settings.index_cache_path).is_file())

            # Identical reruns do not embed or mutate the published collection.
            calls = dense.calls, sparse.calls
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(published, publish_index(client, chunks, dense, sparse, settings))
            self.assertEqual(calls, (dense.calls, sparse.calls))

            with (
                patch("app.retrieval.make_client", return_value=client),
                patch("app.retrieval.TextEmbedding", FakeDense),
                patch("app.retrieval.SparseTextEmbedding", FakeSparse),
                patch("app.retrieval.Ranker", FakeRanker),
            ):
                retriever = Retriever(settings)
            try:
                self.assertEqual(published, retriever.index_version(refresh=True))
                self.assertEqual(2, retriever.count())
                self.assertEqual({"status": "ready", "indexed_chunks": 2, "index_version": published}, retriever.readiness())
                passages = await retriever.retrieve("PostgreSQL CDC replication", product="ingestion")
                self.assertEqual(1, len(passages))
                self.assertEqual(chunks[0]["url"] + "#configuration", passages[0].url)
                self.assertGreater(passages[0].score, settings.min_rerank_score)
                fusion = await retriever.retrieve("Fusion compaction", product="fusion")
                self.assertEqual(1, len(fusion))
                self.assertIn("fusion", fusion[0].url)
                answer = await RAG(settings, retriever, FakeLLM()).ask("PostgreSQL CDC replication", product="ingestion")
                self.assertIn("replication slot", answer.answer)
                self.assertEqual([chunks[0]["url"] + "#configuration"], answer.sources)

                # A changed snapshot switches the real alias and reuses the unchanged embedding.
                updated = sample_chunks()
                updated[0]["text"] += " Checkpoints persist across interrupted syncs."
                with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
                    warnings.filterwarnings("ignore", message="Payload indexes have no effect.*")
                    replacement = publish_index(client, updated, dense, sparse, settings)
                self.assertNotEqual(published, replacement)
                self.assertEqual((calls[0] + 1, calls[1] + 1), (dense.calls, sparse.calls))
                self.assertEqual(1, client.count(settings.collection, exact=True).count)
                self.assertEqual(2, client.count(published, exact=True).count)
                self.assertEqual(replacement, retriever.readiness()["index_version"])
            finally:
                await retriever.aclose()


if __name__ == "__main__":
    unittest.main()
