"""Retention checks use an isolated in-memory Qdrant; never the Cloud cluster."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from qdrant_client import QdrantClient, models

from ingest import refresh_index


class RetentionTests(unittest.TestCase):
    base = "retention_docs"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.client = QdrantClient(":memory:")
        self.addCleanup(self.client.close)
        self.path = Path(self.tmp.name) / "retention.json"
        self.ledger = refresh_index.RetentionLedger(self.path, "test-cloud/retention_docs")
        self.a, self.b, self.c, self.d = [
            f"{self.base}_build_{letter * 20}" for letter in "abcd"
        ]

    def create(self, name, *, nonempty=True):
        self.client.create_collection(
            name,
            vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
        )
        if nonempty:
            self.client.upsert(
                name, [models.PointStruct(id=0, vector=[1.0, 0.0])], wait=True
            )

    def aliases(self):
        return {
            item.alias_name: item.collection_name
            for item in self.client.get_aliases().aliases
        }

    def point(self, suffix, collection):
        alias = f"{self.base}_{suffix}"
        operations = []
        if alias in self.aliases():
            operations.append(
                models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias))
            )
        operations.append(
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(alias_name=alias, collection_name=collection)
            )
        )
        self.client.update_collection_aliases(change_aliases_operations=operations)

    def capture(self):
        return refresh_index.capture_index(self.client, self.base)

    def finish(self, before):
        return refresh_index.finish_refresh(self.client, self.base, before, self.ledger)

    def test_three_publications_keep_current_and_one_previous(self):
        self.create(self.a)
        self.point("current", self.a)
        before = self.capture()
        self.create(self.b)
        self.point("current", self.b)
        self.assertEqual([], self.finish(before))
        self.assertEqual(self.a, self.aliases()[f"{self.base}_previous"])

        before = self.capture()
        self.create(self.c)
        self.point("current", self.c)
        self.assertEqual([self.a], self.finish(before))
        self.assertFalse(self.client.collection_exists(self.a))
        self.assertTrue(self.client.collection_exists(self.b))
        self.assertTrue(self.client.collection_exists(self.c))
        self.assertEqual(self.c, self.aliases()[f"{self.base}_current"])
        self.assertEqual(self.b, self.aliases()[f"{self.base}_previous"])

    def test_first_build_has_no_rollback_or_deletion(self):
        before = self.capture()
        self.assertEqual(refresh_index.IndexSnapshot(None, None), before)
        self.create(self.a)
        self.point("current", self.a)
        self.assertEqual([], self.finish(before))
        self.assertNotIn(f"{self.base}_previous", self.aliases())

    def test_legacy_collection_becomes_rollback_before_aging_out(self):
        self.create(self.base)
        before = self.capture()
        self.assertEqual(self.base, before.current)
        self.create(self.a)
        self.point("current", self.a)
        self.assertEqual([], self.finish(before))
        self.assertEqual(self.base, self.aliases()[f"{self.base}_previous"])
        self.assertTrue(self.client.collection_exists(self.base))

        before = self.capture()
        self.create(self.b)
        self.point("current", self.b)
        self.assertEqual([self.base], self.finish(before))
        self.assertFalse(self.client.collection_exists(self.base))
        self.assertEqual(self.a, self.aliases()[f"{self.base}_previous"])

    def test_identical_rerun_preserves_rollback(self):
        for name in (self.a, self.b):
            self.create(name)
        self.point("previous", self.a)
        self.point("current", self.b)
        before = self.capture()
        with patch.object(self.client, "update_collection_aliases", wraps=self.client.update_collection_aliases) as update:
            self.assertEqual([], self.finish(before))
        update.assert_not_called()
        self.assertEqual(self.a, self.aliases()[f"{self.base}_previous"])
        self.assertTrue(self.client.collection_exists(self.a))

    def test_untracked_staging_and_unrelated_collections_are_preserved(self):
        failed = f"{self.base}_build_{'e' * 20}"
        unrelated = "another_application"
        malformed = f"{self.base}_build_wrong"
        for name in (self.a, self.b, self.c, failed, unrelated, malformed):
            self.create(name)
        self.point("previous", self.a)
        self.point("current", self.b)
        before = self.capture()
        self.point("current", self.c)
        self.assertEqual([self.a], self.finish(before))
        for name in (failed, unrelated, malformed):
            self.assertTrue(self.client.collection_exists(name), name)

    def test_any_other_alias_protects_a_retired_collection(self):
        for name in (self.a, self.b, self.c):
            self.create(name)
        self.point("previous", self.a)
        self.point("current", self.b)
        self.client.update_collection_aliases(change_aliases_operations=[
            models.CreateAliasOperation(create_alias=models.CreateAlias(
                alias_name="external_reader", collection_name=self.a
            ))
        ])
        before = self.capture()
        self.point("current", self.c)
        self.assertEqual([], self.finish(before))
        self.assertTrue(self.client.collection_exists(self.a))
        self.assertEqual(self.a, self.aliases()["external_reader"])

    def test_failed_delete_is_persisted_and_retried_on_unchanged_rerun(self):
        for name in (self.a, self.b, self.c):
            self.create(name)
        self.point("previous", self.a)
        self.point("current", self.b)
        before = self.capture()
        self.point("current", self.c)
        with patch.object(self.client, "delete_collection", side_effect=RuntimeError("temporary network failure")):
            with self.assertRaisesRegex(RuntimeError, "temporary network failure"):
                self.finish(before)
        self.assertEqual(self.b, self.aliases()[f"{self.base}_previous"])
        self.assertEqual(self.c, self.aliases()[f"{self.base}_current"])
        reloaded = refresh_index.RetentionLedger(self.path, "test-cloud/retention_docs")
        self.assertIn(self.a, reloaded.pending)
        before = self.capture()
        self.assertEqual([self.a], refresh_index.finish_refresh(
            self.client, self.base, before, reloaded
        ))
        self.assertNotIn(self.a, reloaded.pending)
        self.assertEqual(self.b, self.aliases()[f"{self.base}_previous"])

    def test_missing_pending_collection_is_discarded(self):
        self.create(self.c)
        self.point("current", self.c)
        self.ledger.add(self.a)
        self.assertEqual([], self.finish(self.capture()))
        self.assertNotIn(self.a, self.ledger.pending)

    def test_pending_current_and_rollback_are_never_deleted(self):
        for name in (self.b, self.c):
            self.create(name)
            self.ledger.add(name)
        self.point("previous", self.b)
        self.point("current", self.c)
        self.assertEqual([], self.finish(self.capture()))
        self.assertTrue(self.client.collection_exists(self.b))
        self.assertTrue(self.client.collection_exists(self.c))

    def test_foreign_and_malformed_pending_names_are_never_deleted(self):
        self.create(self.c)
        self.point("current", self.c)
        for name in ("another_application", f"{self.base}_build_wrong"):
            self.create(name)
            self.ledger.add(name)
        with patch.object(self.client, "delete_collection", wraps=self.client.delete_collection) as delete:
            with self.assertRaises(RuntimeError):
                self.finish(self.capture())
        delete.assert_not_called()
        self.assertTrue(self.client.collection_exists("another_application"))
        self.assertTrue(self.client.collection_exists(f"{self.base}_build_wrong"))

    def test_mixed_owned_and_unsafe_candidates_refuse_before_any_deletion(self):
        foreign = "zzz_another_application"
        for name in (self.a, self.c, foreign):
            self.create(name)
        self.point("current", self.c)
        self.ledger.add(self.a)
        self.ledger.add(foreign)
        with patch.object(self.client, "delete_collection", wraps=self.client.delete_collection) as delete:
            with self.assertRaises(RuntimeError):
                self.finish(self.capture())
        delete.assert_not_called()
        self.assertTrue(self.client.collection_exists(self.a))
        self.assertTrue(self.client.collection_exists(foreign))

    def test_empty_rollback_after_capture_prevents_pending_deletion(self):
        for name in (self.a, self.b, self.c):
            self.create(name)
        self.point("previous", self.b)
        self.point("current", self.c)
        before = self.capture()
        self.ledger.add(self.a)
        self.client.delete(self.b, models.PointIdsList(points=[0]), wait=True)
        with patch.object(self.client, "delete_collection", wraps=self.client.delete_collection) as delete:
            with self.assertRaises(RuntimeError):
                self.finish(before)
        delete.assert_not_called()
        self.assertTrue(self.client.collection_exists(self.a))

    def test_alias_added_during_existence_check_prevents_deletion(self):
        for name in (self.a, self.b, self.c):
            self.create(name)
        self.point("previous", self.b)
        self.point("current", self.c)
        before = self.capture()
        self.ledger.add(self.a)
        exists = self.client.collection_exists

        def add_reader_alias(name):
            result = exists(name)
            if name == self.a:
                self.client.update_collection_aliases(change_aliases_operations=[
                    models.CreateAliasOperation(create_alias=models.CreateAlias(
                        alias_name="new_reader", collection_name=self.a
                    ))
                ])
            return result

        with patch.object(self.client, "collection_exists", side_effect=add_reader_alias):
            with patch.object(self.client, "delete_collection", wraps=self.client.delete_collection) as delete:
                self.assertEqual([], self.finish(before))
        delete.assert_not_called()
        self.assertTrue(self.client.collection_exists(self.a))
        self.assertEqual(self.a, self.aliases()["new_reader"])

    def test_previous_alias_changed_during_build_prevents_cleanup(self):
        for name in (self.a, self.b, self.c, self.d):
            self.create(name)
        self.point("previous", self.a)
        self.point("current", self.b)
        before = self.capture()
        self.point("current", self.c)
        self.point("previous", self.d)
        with patch.object(self.client, "delete_collection", wraps=self.client.delete_collection) as delete:
            with self.assertRaises(RuntimeError):
                self.finish(before)
        delete.assert_not_called()
        self.assertEqual(self.d, self.aliases()[f"{self.base}_previous"])

    def test_current_alias_changed_during_rotation_prevents_cleanup(self):
        for name in (self.a, self.b, self.c, self.d):
            self.create(name)
        self.point("previous", self.a)
        self.point("current", self.b)
        before = self.capture()
        self.point("current", self.c)
        update = self.client.update_collection_aliases

        def race(*args, **kwargs):
            result = update(*args, **kwargs)
            operations = kwargs["change_aliases_operations"]
            if any(getattr(op, "create_alias", None) and op.create_alias.alias_name == f"{self.base}_previous" for op in operations):
                update(change_aliases_operations=[
                    models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=f"{self.base}_current")),
                    models.CreateAliasOperation(create_alias=models.CreateAlias(alias_name=f"{self.base}_current", collection_name=self.d)),
                ])
            return result

        with patch.object(self.client, "update_collection_aliases", side_effect=race):
            with patch.object(self.client, "delete_collection", wraps=self.client.delete_collection) as delete:
                with self.assertRaises(RuntimeError):
                    self.finish(before)
        delete.assert_not_called()
        self.assertTrue(self.client.collection_exists(self.a))
        self.assertEqual(self.d, self.aliases()[f"{self.base}_current"])

    def test_capture_rejects_foreign_and_empty_current_collections(self):
        for name in ("another_application", self.a):
            with self.subTest(name=name):
                self.create(name, nonempty=False)
                self.point("current", name)
                with self.assertRaises((ValueError, RuntimeError)):
                    self.capture()

    def test_capture_rejects_invalid_rollback_target(self):
        self.create(self.c)
        self.create("another_application")
        self.point("current", self.c)
        self.point("previous", "another_application")
        with self.assertRaises((ValueError, RuntimeError)):
            self.capture()

    def test_manual_rollback_is_preserved_on_identical_rebuild(self):
        for name in (self.b, self.c):
            self.create(name)
        self.point("current", self.c)
        self.point("previous", self.b)
        self.client.update_collection_aliases(change_aliases_operations=[
            models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=f"{self.base}_current")),
            models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=f"{self.base}_previous")),
            models.CreateAliasOperation(create_alias=models.CreateAlias(alias_name=f"{self.base}_current", collection_name=self.b)),
            models.CreateAliasOperation(create_alias=models.CreateAlias(alias_name=f"{self.base}_previous", collection_name=self.c)),
        ])
        self.assertEqual([], self.finish(self.capture()))
        self.assertEqual(self.b, self.aliases()[f"{self.base}_current"])
        self.assertEqual(self.c, self.aliases()[f"{self.base}_previous"])

    def test_empty_new_publication_does_not_delete_or_rotate(self):
        self.create(self.a)
        self.point("current", self.a)
        before = self.capture()
        self.create(self.b, nonempty=False)
        self.point("current", self.b)
        with patch.object(self.client, "delete_collection", wraps=self.client.delete_collection) as delete:
            with self.assertRaises((ValueError, RuntimeError)):
                self.finish(before)
        delete.assert_not_called()
        self.assertNotIn(f"{self.base}_previous", self.aliases())

    def test_ledger_persists_and_does_not_reuse_another_cloud_scope(self):
        self.ledger.add(self.a)
        self.assertEqual({self.a}, refresh_index.RetentionLedger(
            self.path, "test-cloud/retention_docs"
        ).pending)
        self.assertEqual(set(), refresh_index.RetentionLedger(
            self.path, "different-cloud/retention_docs"
        ).pending)
        self.ledger.discard(self.a)
        self.assertEqual(set(), refresh_index.RetentionLedger(
            self.path, "test-cloud/retention_docs"
        ).pending)

    def test_malformed_ledger_refuses_cleanup(self):
        self.path.write_text("{not valid JSON", encoding="utf-8")
        with self.assertRaises((ValueError, RuntimeError, json.JSONDecodeError)):
            refresh_index.RetentionLedger(self.path, "test-cloud/retention_docs")

    def test_builder_failure_never_runs_retention(self):
        settings = SimpleNamespace(
            collection=self.base,
            qdrant_url="https://isolated-test.invalid",
            qdrant_api_key="test-key",
            index_cache_path=str(Path(self.tmp.name) / "embeddings.sqlite3"),
        )
        failure = subprocess.CalledProcessError(9, ["python", "-m", "ingest.build_index"])
        with patch.object(refresh_index, "get_settings", return_value=settings):
            with patch.object(refresh_index, "make_client", return_value=self.client):
                with patch.object(refresh_index.subprocess, "run", side_effect=failure) as run:
                    with patch.object(refresh_index, "finish_refresh") as finish:
                        self.assertNotEqual(0, refresh_index.cli())
        run.assert_called_once()
        self.assertTrue(run.call_args.kwargs.get("check"))
        finish.assert_not_called()


if __name__ == "__main__":
    unittest.main()
