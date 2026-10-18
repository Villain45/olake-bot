import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from flashrank import Ranker
from flashrank.Config import model_file_map

from app.inference import CpuRanker, rerank_batched
from app.config import Settings


MODEL_NAME = "ms-marco-MiniLM-L-12-v2"


class InferenceSettingsTests(unittest.TestCase):
    def test_defaults_fit_the_shared_cpu(self):
        settings = Settings(_env_file=None)
        self.assertEqual(settings.inference_threads, 1)
        self.assertEqual(settings.rerank_batch_size, 2)

    def test_nonpositive_inference_settings_are_rejected(self):
        for name in ("inference_threads", "rerank_batch_size"):
            for value in (0, -1):
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValueError):
                        Settings(_env_file=None, **{name: value})


class CpuRankerTests(unittest.TestCase):
    def test_initialization_constructs_only_the_configured_cpu_session(self):
        for threads in (1, 3):
            with self.subTest(threads=threads), \
                    patch.object(Ranker, "__init__", side_effect=AssertionError("default session")), \
                    patch.object(Ranker, "_prepare_model_dir") as prepare, \
                    patch.object(Ranker, "_get_tokenizer") as tokenizer, \
                    patch("app.inference.ort.SessionOptions") as session_options, \
                    patch("app.inference.ort.InferenceSession") as session:
                ranker = CpuRanker(model_name=MODEL_NAME, cache_dir="cache", threads=threads)

                options = session_options.return_value
                self.assertEqual(options.intra_op_num_threads, threads)
                self.assertEqual(options.inter_op_num_threads, threads)
                from app.inference import ort
                self.assertEqual(options.execution_mode, ort.ExecutionMode.ORT_SEQUENTIAL)
                self.assertEqual(
                    options.add_session_config_entry.call_args_list,
                    [unittest.mock.call("session.intra_op.allow_spinning", "0"),
                     unittest.mock.call("session.inter_op.allow_spinning", "0")],
                )
                session.assert_called_once_with(
                    str(Path("cache") / MODEL_NAME / model_file_map[MODEL_NAME]),
                    sess_options=options,
                    providers=["CPUExecutionProvider"],
                )
                prepare.assert_called_once_with(MODEL_NAME)
                tokenizer.assert_called_once_with(512)
                self.assertIs(ranker.tokenizer, tokenizer.return_value)
                self.assertIs(ranker.session, session.return_value)
                self.assertIsNone(ranker.llm_model)
                self.assertIs(CpuRanker.rerank, Ranker.rerank)

    def test_default_threads_and_explicit_token_budget(self):
        with patch.object(Ranker, "_prepare_model_dir"), \
                patch.object(Ranker, "_get_tokenizer") as tokenizer, \
                patch("app.inference.ort.SessionOptions") as session_options, \
                patch("app.inference.ort.InferenceSession"):
            CpuRanker(model_name=MODEL_NAME, cache_dir="cache", max_length=256)
        self.assertEqual(session_options.return_value.intra_op_num_threads, 1)
        self.assertEqual(session_options.return_value.inter_op_num_threads, 1)
        tokenizer.assert_called_once_with(256)

    def test_invalid_configuration_is_rejected_before_loading(self):
        for name in ("threads", "max_length"):
            for value in (0, -1, 1.5, "1", True, None):
                with self.subTest(name=name, value=value), \
                        patch.object(Ranker, "_prepare_model_dir") as prepare, \
                        patch("app.inference.ort.InferenceSession") as session:
                    with self.assertRaisesRegex(ValueError, name):
                        CpuRanker(model_name=MODEL_NAME, cache_dir="cache", **{name: value})
                    prepare.assert_not_called()
                    session.assert_not_called()

    def test_listwise_initialization_delegates_without_an_onnx_session(self):
        with patch.object(Ranker, "__init__", return_value=None) as initialize, \
                patch("app.inference.ort.InferenceSession") as session:
            CpuRanker(model_name="rank_zephyr_7b_v1_full", cache_dir="cache", max_length=1024)
        initialize.assert_called_once_with(
            model_name="rank_zephyr_7b_v1_full", cache_dir="cache", max_length=1024,
        )
        session.assert_not_called()


class MutatingRanker:
    """Match FlashRank's mutation while deliberately reversing score ties."""

    def __init__(self):
        self.calls = []

    def rerank(self, request):
        self.calls.append((request.query, [dict(passage) for passage in request.passages]))
        for passage in request.passages:
            passage["score"] = float(passage["text"])
        request.passages.reverse()
        request.passages.sort(key=lambda passage: passage["score"], reverse=True)
        return request.passages


class BatchedRerankTests(unittest.TestCase):
    def test_empty_input_does_not_call_ranker(self):
        ranker = Mock()
        self.assertEqual(rerank_batched(ranker, "query", [], 2), [])
        ranker.rerank.assert_not_called()

    def test_batches_merge_globally_with_original_ids_and_stable_ties(self):
        ranker = MutatingRanker()
        passages = [
            {"id": "a", "text": "0.5", "meta": {"source": "docs"}},
            {"id": 47, "text": "0.5"},
            {"id": "last-batch-winner", "text": "0.9"},
            {"id": "a", "text": "0.5"},
            {"id": -1, "text": "0.1"},
        ]
        before = [dict(passage) for passage in passages]

        results = rerank_batched(ranker, "original query", passages, 2)

        self.assertEqual([len(batch) for _, batch in ranker.calls], [2, 2, 1])
        self.assertEqual([query for query, _ in ranker.calls], ["original query"] * 3)
        self.assertEqual([result["id"] for result in results], ["last-batch-winner", "a", 47, "a", -1])
        self.assertEqual([result["score"] for result in results], [0.9, 0.5, 0.5, 0.5, 0.1])
        self.assertEqual(results[1]["meta"], {"source": "docs"})
        self.assertEqual(passages, before)
        self.assertTrue(all(result is not passage for result in results for passage in passages))
        self.assertTrue(all(set(result) == set(before[index]) | {"score"}
                            for result, index in zip(results, [2, 0, 1, 3, 4])))

    def test_one_passage_and_batch_larger_than_input(self):
        ranker = MutatingRanker()
        self.assertEqual(rerank_batched(ranker, "q", [{"id": "single", "text": "0.8"}], 8),
                         [{"id": "single", "text": "0.8", "score": 0.8}])
        self.assertEqual(len(ranker.calls), 1)

    def test_plain_mock_returning_new_dicts_uses_original_ids_for_ties(self):
        ranker = Mock()
        ranker.rerank.side_effect = [
            [{"id": "second", "score": 0.5}, {"id": "first", "score": 0.5}],
            [{"id": "third", "score": 0.9}],
        ]
        results = rerank_batched(ranker, "q", [
            {"id": "first", "text": "one"},
            {"id": "second", "text": "two"},
            {"id": "third", "text": "three"},
        ], 2)
        self.assertEqual([result["id"] for result in results], ["third", "first", "second"])
        self.assertEqual(ranker.rerank.call_count, 2)

    def test_listwise_model_receives_one_complete_copied_request(self):
        ranker = Ranker.__new__(Ranker)
        ranker.llm_model = object()
        ranker.rerank = Mock(side_effect=lambda request: list(reversed(request.passages)))
        passages = [{"id": index, "text": str(index)} for index in range(3)]
        results = rerank_batched(ranker, "q", passages, 1)
        ranker.rerank.assert_called_once()
        self.assertEqual([result["id"] for result in results], [2, 1, 0])
        self.assertTrue(all(result is not passage for result in results for passage in passages))

    def test_invalid_batch_size_is_rejected_even_for_empty_input(self):
        for value in (0, -1, 1.5, "1", True, None):
            with self.subTest(value=value):
                ranker = Mock()
                with self.assertRaisesRegex(ValueError, "batch_size"):
                    rerank_batched(ranker, "q", [], value)
                ranker.rerank.assert_not_called()


if __name__ == "__main__":
    unittest.main()
