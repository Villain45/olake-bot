import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from eval.run_eval import (
    calibrate_threshold, load_calibration_records, load_cases, main, percentile,
    reported_cost, run_cases, score_result, source_matches, summarize,
)


class EvaluationTests(unittest.TestCase):
    def test_corpus_has_48_unique_cases_with_review_expectations(self):
        cases = load_cases(Path(__file__).resolve().parents[1] / "eval" / "questions.jsonl")
        self.assertEqual(48, len(cases))
        self.assertEqual(len(cases), len({case["id"] for case in cases}))
        self.assertTrue(any(case["answerable"] is None for case in cases))
        self.assertGreaterEqual(sum(case["answerable"] is False for case in cases), 5)
        self.assertTrue(all(case.get("expected_sources") and case.get("expected_keywords") for case in cases if case["answerable"] is True))

    def test_expected_page_matches_fragment_but_not_other_hosts_or_pages(self):
        page = "https://olake.io/docs/connectors/postgres/"
        self.assertTrue(source_matches(page + "#configuration", page))
        self.assertFalse(source_matches("https://olake.io.evil.example/docs/connectors/postgres/", page))
        self.assertFalse(source_matches("https://olake.io/docs/connectors/postgres/setup/rds/", page))
        self.assertFalse(source_matches("https://[invalid/", page))

    def test_commit_citations_match_same_github_file_only(self):
        expected = "https://github.com/datazip-inc/olake-fusion/blob/main/README.md"
        commit = "https://github.com/datazip-inc/olake-fusion/blob/" + "a" * 40
        self.assertTrue(source_matches(commit + "/README.md#setup", expected))
        for actual in (
            commit + "/other/README.md", commit.replace("olake-fusion", "olake") + "/README.md",
            commit.replace("datazip-inc", "someone-else") + "/README.md",
            expected.replace("github.com", "github.com.evil.example"),
            expected.replace("github.com/", "github.com:444/"),
            expected.replace("https://", "https://user@"),
            commit + "/../README.md",
        ):
            with self.subTest(actual=actual):
                self.assertFalse(source_matches(actual, expected))

    def test_known_docs_repo_pages_match_canonical_pages(self):
        base = "https://github.com/datazip-inc/olake-docs/blob/" + "b" * 40 + "/"
        self.assertTrue(source_matches(base + "docs/connectors/postgres/index.mdx#configuration", "https://olake.io/docs/connectors/postgres/"))
        self.assertTrue(source_matches(base + "docs/install/kubernetes.md", "https://olake.io/docs/install/kubernetes/"))
        self.assertTrue(source_matches(base + "docs/index.mdx", "https://olake.io/docs/"))
        self.assertFalse(source_matches(base + "blog/postgres.mdx", "https://olake.io/docs/postgres/"))
        self.assertFalse(source_matches(base + "docs/connectors/postgres/setup/index.mdx", "https://olake.io/docs/connectors/postgres/"))
        self.assertFalse(source_matches(base.replace("datazip-inc", "attacker") + "docs/connectors/postgres/index.mdx", "https://olake.io/docs/connectors/postgres/"))

    def test_refusal_and_sources_are_independent(self):
        case = {"answerable": False}
        result = score_result(case, {"answer": "I couldn't find this in the OLake docs.", "sources": ["https://olake.io/docs/"]})
        self.assertTrue(result["answerability_proxy_match"])
        self.assertTrue(result["refusal_detected_proxy"])
        self.assertIsNone(result["expected_source_match"])

    def test_positive_answer_requires_all_selected_proxies_to_pass(self):
        case = {"answerable": True, "expected_sources": ["https://olake.io/docs/"], "expected_keywords": ["PostgreSQL", "MySQL"]}
        result = score_result(case, {"answer": "Supports PostgreSQL and MySQL.", "sources": ["https://olake.io/docs/#supported-sources"]})
        self.assertTrue(result["proxy_checks_passed"])
        result = score_result(case, {"answer": "Supports PostgreSQL.", "sources": []})
        self.assertFalse(result["expected_keyword_match"])
        self.assertFalse(result["expected_source_match"])
        self.assertFalse(result["proxy_checks_passed"])

    def test_unknown_expectation_does_not_become_a_pass(self):
        result = score_result({"answerable": None}, {"answer": "Needs review.", "sources": []})
        self.assertIsNone(result["proxy_checks_passed"])
        self.assertIsNone(result["answerability_proxy_match"])

    def test_clarification_is_not_counted_as_answering(self):
        response = {"answer": "Which Iceberg catalog and platform are you using?", "sources": []}
        result = score_result({"answerable": True}, response)
        self.assertTrue(result["clarification_detected_proxy"])
        self.assertFalse(result["answerability_proxy_match"])
        self.assertTrue(score_result({"answerable": False}, response)["answerability_proxy_match"])

    def test_invalid_response_types_are_errors(self):
        invalid = [None, [], {"answer": 1, "sources": []}, {"answer": "yes", "sources": "url"}, {"answer": "yes", "sources": [3]}, {"answer": " ", "sources": []}]
        for data in invalid:
            with self.subTest(data=data):
                self.assertFalse(score_result({}, data)["valid_response"])

    def test_malformed_expectation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            path.write_text(json.dumps({"question": "Q", "answerable": "false"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_cases(path)

    def test_runner_continues_after_http_json_and_network_errors(self):
        calls = []

        def handler(request):
            calls.append(json.loads(request.content))
            index = len(calls)
            if index == 1:
                return httpx.Response(200, json={"answer": "PostgreSQL", "sources": ["https://olake.io/docs/"]})
            if index == 2:
                return httpx.Response(429, json={"detail": "limit"})
            if index == 3:
                return httpx.Response(200, text="not json")
            raise httpx.ConnectError("offline", request=request)

        cases = [{"question": f"Q{index}", "answerable": True} for index in range(4)]
        with httpx.Client(transport=httpx.MockTransport(handler)) as client, contextlib.redirect_stdout(io.StringIO()):
            results = run_cases(cases, client, "http://test/", interval=0)
        self.assertEqual(4, len(results))
        self.assertEqual([{"question": case["question"]} for case in cases], calls)
        summary = summarize(results)
        self.assertEqual(3, summary["errors"])
        self.assertEqual({"200": 2, "429": 1, "network_error": 1}, summary["http_statuses"])
        self.assertEqual(1, summary["answerability_proxy_match"]["measured"])
        self.assertIsNone(summary["reported_cost_usd"]["total"])

    def test_latency_and_reported_cost_do_not_invent_measurements(self):
        self.assertIsNone(percentile([], 0.95))
        self.assertEqual(2.5, percentile([1, 2, 3, 4], 0.5))
        self.assertEqual(0.0, reported_cost({"usage": {"cost": 0}}))
        self.assertEqual(0.01, reported_cost({"cost_usd": 0.01}))
        for data in ({}, {"cost_usd": True}, {"cost_usd": float("nan")}, {"cost_usd": -1}):
            with self.subTest(data=data):
                self.assertIsNone(reported_cost(data))
        self.assertEqual(0, summarize([])["queries"])

    def test_report_separates_cache_latencies_categories_and_unknowns(self):
        results = [
            {"latency_s": 0.1, "category": "connectors", "cache_state": "HIT", "index_version": "v1", "proxy_checks_passed": True, "sources": ["https://olake.io/docs/connectors/postgres/#setup"]},
            {"latency_s": 2, "category": "connectors", "cache_state": "MISS", "index_version": "v1", "proxy_checks_passed": False, "sources": ["https://olake.io/docs/connectors/postgres/"]},
            {"latency_s": 0.2, "category": "manual_review", "proxy_checks_passed": None, "sources": []},
        ]
        summary = summarize(results)
        self.assertEqual(1, summary["quality_failures"])
        self.assertEqual(1, summary["unmeasured_queries"])
        self.assertEqual(2, summary["proxy_checks_passed"]["measured"])
        self.assertEqual(0.1, summary["latency_by_cache_state_s"]["HIT"]["mean"])
        self.assertEqual(2, summary["latency_by_cache_state_s"]["MISS"]["mean"])
        self.assertEqual(1, summary["by_category"]["connectors"]["quality_failures"])
        self.assertEqual(1, summary["source_variety"]["unique_pages"])
        self.assertEqual(["v1"], summary["index_versions"])

    def test_runner_records_cache_and_index_headers(self):
        def handler(request):
            return httpx.Response(200, headers={
                "X-Cache": "COALESCED", "X-Index-Version": "test-index", "X-Retrieval-Score": "0.61",
                "X-Answer-Model": "allowed-model", "X-Recovery-Attempted": "true",
            }, json={"answer": "OLake", "sources": []})
        with httpx.Client(transport=httpx.MockTransport(handler)) as client, contextlib.redirect_stdout(io.StringIO()):
            results = run_cases([{"question": "Q", "category": "custom"}], client, "http://test", interval=0)
        self.assertEqual("COALESCED", results[0]["cache_state"])
        self.assertEqual("test-index", results[0]["index_version"])
        self.assertEqual("custom", results[0]["category"])
        self.assertEqual(0.61, results[0]["retrieval_score"])
        self.assertEqual("allowed-model", results[0]["answer_model"])
        self.assertTrue(results[0]["recovery_attempted"])

    def test_cli_fails_for_measured_quality_failures_and_errors(self):
        for row, expected_exit in (
            ({"proxy_checks_passed": False}, 1),
            ({"error": "HTTP 503"}, 1),
            ({"proxy_checks_passed": True}, 0),
            ({"proxy_checks_passed": None}, 0),
        ):
            with self.subTest(row=row), patch("eval.run_eval.load_cases", return_value=[]), patch("eval.run_eval.run_cases", return_value=[{"latency_s": 0.1, **row}]), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(expected_exit, main(["--interval", "0"]))

    def test_offline_calibration_selects_best_threshold_without_api_calls(self):
        labels = [
            {"score": 0.8, "answerable": True}, {"score": 0.6, "answerable": True},
            {"score": 0.4, "answerable": False}, {"score": 0.1, "answerable": False},
        ]
        report = calibrate_threshold(labels)
        self.assertEqual(0.6, report["summary"]["recommended_threshold"])
        self.assertEqual(1, next(row for row in report["thresholds"] if row["threshold"] == 0.6)["f1"])
        self.assertIsNone(calibrate_threshold(labels[:2])["summary"]["recommended_threshold"])
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "scores.jsonl", Path(directory) / "report.json"
            source.write_text("\n".join(json.dumps(row) for row in labels), encoding="utf-8")
            self.assertEqual(labels, load_calibration_records(source))
            with patch("eval.run_eval.httpx.Client") as client, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(0, main(["--calibrate", str(source), "--output", str(output)]))
            client.assert_not_called()
            self.assertEqual("offline_calibration", json.loads(output.read_text(encoding="utf-8"))["mode"])

    def test_score_export_marks_provisional_labels_and_skips_unknowns(self):
        rows = [
            {"id": "positive", "question": "Q", "category": "connectors", "latency_s": 0.1, "retrieval_score": 0.8, "expected_answerable": True},
            {"id": "negative", "question": "Q2", "category": "refusal", "latency_s": 0.1, "retrieval_score": 0.2, "expected_answerable": False},
            {"id": "unknown", "question": "Q3", "category": "manual_review", "latency_s": 0.1, "retrieval_score": 0.5, "expected_answerable": None},
            {"id": "error", "question": "Q4", "category": "connectors", "latency_s": 0.1, "retrieval_score": 0.8, "expected_answerable": True, "error": "bad response"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "scores.jsonl"
            with patch("eval.run_eval.load_cases", return_value=[]), patch("eval.run_eval.run_cases", return_value=rows), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(1, main(["--score-output", str(output)]))
            labels = load_calibration_records(output)
        self.assertEqual(2, len(labels))
        self.assertTrue(all(row["needs_evidence_review"] for row in labels))
        self.assertEqual(2, calibrate_threshold(labels)["summary"]["proxy_labeled_records"])

    def test_calibration_rejects_invalid_and_unlabeled_scores(self):
        for record in ({"score": 0.4, "answerable": None}, {"score": True, "answerable": True}, {"score": 1.1, "answerable": False}, {"score": -0.1, "answerable": False}, {"score": float("nan"), "answerable": False}):
            with self.subTest(record=record), tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / "scores.json"
                source.write_text(json.dumps([record]), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_calibration_records(source)


if __name__ == "__main__":
    unittest.main()
