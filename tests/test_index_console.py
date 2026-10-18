"""Console visibility checks using deterministic model and client doubles."""

import contextlib
import io
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

from ingest import build_index


class FlushRecorder(io.StringIO):
    def __init__(self):
        super().__init__()
        self.flush_count = 0

    def flush(self):
        self.flush_count += 1
        super().flush()


class ConsoleTests(unittest.TestCase):
    def test_reports_flush_immediately_and_route_errors_to_stderr(self):
        stdout, stderr = FlushRecorder(), FlushRecorder()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            build_index.report("Indexing started")
            build_index.report("Retrying upload", level="WARNING")
            build_index.report("Upload failed", level="ERROR")
        self.assertIn("Indexing started", stdout.getvalue())
        self.assertRegex(stdout.getvalue(), r"\d{2}:\d{2}:\d{2} \[INFO\]")
        self.assertNotIn("Upload failed", stdout.getvalue())
        self.assertIn("Retrying upload", stderr.getvalue())
        self.assertIn("Upload failed", stderr.getvalue())
        self.assertGreaterEqual(stdout.flush_count, 1)
        self.assertGreaterEqual(stderr.flush_count, 2)

    def test_stage_reports_a_heartbeat_while_the_operation_is_running(self):
        heartbeat_seen = threading.Event()
        messages = []

        def record(message, **kwargs):
            messages.append(str(message))
            if len(messages) >= 2:
                heartbeat_seen.set()

        with patch.object(build_index, "report", side_effect=record):
            with build_index.stage("Slow model", heartbeat_s=0.01):
                self.assertTrue(heartbeat_seen.wait(timeout=1), "No progress heartbeat during the operation")
        self.assertGreaterEqual(len(messages), 3)
        self.assertTrue(all("Slow model" in message for message in messages))
        self.assertIn("elapsed", messages[1])
        self.assertIn("completed", messages[-1])

    def test_failed_stage_reports_error_and_reraises_original_exception(self):
        error = RuntimeError("embedding model failed")
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            with self.assertRaises(RuntimeError) as raised:
                with build_index.stage("Dense embeddings"):
                    raise error
        self.assertIs(raised.exception, error)
        self.assertIn("Dense embeddings", stderr.getvalue())
        self.assertIn("embedding model failed", stderr.getvalue())

    def test_retry_logs_each_failed_attempt_and_preserves_final_exception(self):
        error = RuntimeError("Qdrant connection lost")
        operation = MagicMock(side_effect=error)
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), patch.object(build_index.time, "sleep") as sleep:
            with self.assertRaises(RuntimeError) as raised:
                build_index._retry(operation, description="Upload batch 2/4")
        self.assertIs(raised.exception, error)
        self.assertEqual(operation.call_count, 3)
        self.assertEqual(sleep.call_args_list, [call(1), call(2)])
        output = stderr.getvalue()
        self.assertEqual(output.count("Qdrant connection lost"), 3)
        self.assertIn("Upload batch 2/4", output)
        self.assertIn("1/3", output)
        self.assertIn("2/3", output)
        self.assertIn("3/3", output)

    def test_successful_retry_returns_value_and_reports_prior_failure(self):
        operation = MagicMock(side_effect=[RuntimeError("temporary outage"), "uploaded"])
        with contextlib.redirect_stderr(io.StringIO()) as stderr, patch.object(build_index.time, "sleep"):
            self.assertEqual(build_index._retry(operation, description="Upload batch"), "uploaded")
        self.assertEqual(operation.call_count, 2)
        self.assertIn("temporary outage", stderr.getvalue())

    def test_cli_error_prints_full_traceback_and_returns_failure_exit_code(self):
        with patch.object(build_index, "main", side_effect=RuntimeError("Cannot reach Qdrant")):
            with contextlib.redirect_stderr(io.StringIO()) as stderr:
                self.assertEqual(build_index.cli(), 1)
        self.assertIn("Traceback (most recent call last)", stderr.getvalue())
        self.assertIn("RuntimeError: Cannot reach Qdrant", stderr.getvalue())

    def test_cli_returns_success_or_interrupt_exit_code(self):
        with patch.object(build_index, "main"):
            self.assertEqual(build_index.cli(), 0)
        with patch.object(build_index, "main", side_effect=KeyboardInterrupt):
            with contextlib.redirect_stderr(io.StringIO()) as stderr:
                self.assertEqual(build_index.cli(), 130)
        self.assertTrue(stderr.getvalue().strip())

    def test_cleanup_failure_is_reported_without_masking_the_build_error(self):
        original_error = RuntimeError("Qdrant authentication failed")
        client = MagicMock()
        client.get_collections.side_effect = original_error
        client.close.side_effect = RuntimeError("Qdrant client cleanup failed")
        with (
            patch.object(build_index, "get_settings", return_value=SimpleNamespace()),
            patch.object(build_index, "make_client", return_value=client),
            patch.object(build_index, "TextEmbedding") as dense,
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()) as stderr,
        ):
            with self.assertRaises(RuntimeError) as raised:
                build_index.main()
        self.assertIs(raised.exception, original_error)
        self.assertIn("Qdrant authentication failed", stderr.getvalue())
        self.assertIn("Qdrant client cleanup failed", stderr.getvalue())
        self.assertIn("Traceback (most recent call last)", stderr.getvalue())
        client.close.assert_called_once()
        dense.assert_not_called()


class Vec(list):
    def tolist(self):
        return list(self)


class Dense:
    def __init__(self):
        self.calls = 0

    def embed(self, texts, **kwargs):
        self.calls += 1
        return [Vec([0.1, 0.2]) for _ in texts]


class Sparse(Dense):
    def embed(self, texts, **kwargs):
        self.calls += 1
        return [SimpleNamespace(indices=Vec([1]), values=Vec([1.0])) for _ in texts]


class BatchConsoleTests(unittest.TestCase):
    def test_upload_progress_covers_batches_and_cached_embeddings(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = SimpleNamespace(collection="console_test", embed_model="dense", sparse_model="sparse",
                                       embed_dim=2, index_batch_size=1,
                                       index_cache_path=str(Path(directory) / "cache.sqlite3"))
            chunks = [{"text": text, "breadcrumb": "Documentation", "product": "ingestion",
                       "url": "https://olake.io/docs", "title": "Docs"}
                      for text in ("First passage", "Second passage")]
            client = MagicMock()
            client.get_aliases.return_value = SimpleNamespace(aliases=[])
            client.collection_exists.return_value = False
            client.count.return_value = SimpleNamespace(count=2)
            dense, sparse = Dense(), Sparse()
            with contextlib.redirect_stdout(io.StringIO()) as fresh:
                build_index.publish_index(client, chunks, dense, sparse, settings)
            output = fresh.getvalue()
            self.assertIn("1/2", output)
            self.assertIn("2/2", output)
            self.assertIn("100.0%", output)
            self.assertIn("ETA", output)
            self.assertIn("cached", output.lower())
            self.assertIn("new", output.lower())
            self.assertIn("1 new, 0 cached", output)
            self.assertIn("Published", output)
            calls = dense.calls, sparse.calls
            with contextlib.redirect_stdout(io.StringIO()) as cached:
                build_index.publish_index(client, chunks, dense, sparse, settings)
            self.assertEqual((dense.calls, sparse.calls), calls)
            self.assertIn("0 new, 1 cached", cached.getvalue())
            self.assertIn("2/2", cached.getvalue())


if __name__ == "__main__":
    unittest.main()
