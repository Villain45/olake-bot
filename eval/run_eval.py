"""Explicitly run API evaluation; importing this module never sends requests.

Source and keyword checks are reproducible proxies, not a grounded-answer judge.
The API has no answerability flag, so refusal detection is also only a proxy.
"""

import argparse
import json
import math
import re
import statistics
import time
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlsplit

import httpx


REFUSAL = re.compile(
    r"could(?:n['’]t| not) find|(?:cannot|can['’]t|unable to) "
    r"(?:find|answer|help|provide|assist)|don['’]t (?:know|have (?:enough|information))|"
    r"insufficient (?:information|context|evidence)",
    re.IGNORECASE,
)
CLARIFICATION = re.compile(
    r"\b(?:which|what|could you|can you|please (?:specify|clarify|share|provide)|"
    r"(?:need|tell me|let me know).{0,40}(?:which|what|version|platform|catalog))\b",
    re.IGNORECASE,
)
GITHUB_REF = re.compile(r"(?:main|master|[a-fA-F0-9]{7,64})\Z")


def load_cases(path: Path) -> list[dict]:
    cases = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get("question"), str) or not row["question"].strip():
            raise ValueError(f"{path}:{line_number}: expected a nonempty question")
        expected = row.get("answerable")
        if expected is not None and type(expected) is not bool:
            raise ValueError(f"{path}:{line_number}: answerable must be boolean or null")
        for key in ("expected_sources", "expected_keywords"):
            value = row.get(key)
            if value is not None and (
                not isinstance(value, list)
                or not value
                or any(not isinstance(item, str) or not item.strip() for item in value)
            ):
                raise ValueError(f"{path}:{line_number}: {key} must be a nonempty string list or null")
        if row.get("category") is not None and (
            not isinstance(row["category"], str) or not row["category"].strip()
        ):
            raise ValueError(f"{path}:{line_number}: category must be a nonempty string or null")
        cases.append(row)
    return cases


def canonical_source(source: str) -> tuple[str, str] | None:
    """Identify one exact page; only known public repo routes get aliases."""
    try:
        url = urlsplit(source)
        if (
            url.scheme not in {"https", "http"}
            or not url.hostname
            or url.username is not None
            or url.password is not None
            or url.port not in (None, 80 if url.scheme == "http" else 443)
        ):
            return None
    except ValueError:
        return None
    # Encoded separators can identify a different route on the source server.
    if re.search(r"%(?:2f|5c)", url.path, re.IGNORECASE):
        return None
    path = unquote(url.path).rstrip("/")
    if any(part in {".", ".."} for part in path.split("/")) or "\\" in path:
        return None
    host = url.hostname
    parts = path.strip("/").split("/")
    if host == "github.com" and len(parts) >= 5 and parts[2] == "blob" and GITHUB_REF.fullmatch(parts[3]):
        owner, repo = parts[:2]
        file_parts = parts[4:]
        if owner.casefold() == "datazip-inc" and repo.casefold() == "olake-docs" and file_parts[0] == "docs":
            doc_parts = file_parts[1:]
            if doc_parts and doc_parts[-1].endswith((".md", ".mdx")):
                doc_parts[-1] = re.sub(r"\.mdx?$", "", doc_parts[-1])
                if doc_parts[-1] == "index":
                    doc_parts.pop()
                return "olake.io", "/docs" + ("/" + "/".join(doc_parts) if doc_parts else "")
        # Branch/commit differences do not change the identity of the cited file.
        return host, "/" + "/".join([owner.casefold(), repo.casefold(), "blob", *file_parts])
    return host, path


def source_matches(source: str, expected: str) -> bool:
    """Match exact pages, including known GitHub commit and docs equivalents."""
    actual, wanted = canonical_source(source), canonical_source(expected)
    return actual is not None and wanted is not None and actual == wanted


def score_result(case: dict, data: object) -> dict:
    if (
        not isinstance(data, dict)
        or not isinstance(data.get("answer"), str)
        or not data["answer"].strip()
        or not isinstance(data.get("sources"), list)
        or any(not isinstance(url, str) for url in data["sources"])
    ):
        return {"valid_response": False, "error": "Response does not match {answer, sources}"}

    answer, sources = data["answer"], data["sources"]
    refused = bool(REFUSAL.search(answer))
    clarification = not sources and answer.rstrip().endswith("?") and bool(CLARIFICATION.search(answer))
    expected_sources = case.get("expected_sources")
    expected_keywords = case.get("expected_keywords")
    checks = {
        "answerability_proxy_match": None if case.get("answerable") is None else (not (refused or clarification)) == case["answerable"],
        "expected_source_match": None if expected_sources is None else any(
            source_matches(source, expected) for source in sources for expected in expected_sources
        ),
        "expected_keyword_match": None if expected_keywords is None else all(
            keyword.casefold() in answer.casefold() for keyword in expected_keywords
        ),
    }
    checked = [value for value in checks.values() if value is not None]
    return {
        "valid_response": True,
        "answer": answer,
        "sources": sources,
        "refusal_detected_proxy": refused,
        "clarification_detected_proxy": clarification,
        "source_count": len(sources),
        "unique_source_pages": len({page for source in sources if (page := canonical_source(source)) is not None}),
        **checks,
        "proxy_checks_passed": all(checked) if checked else None,
    }


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def latency_summary(values: list[float]) -> dict:
    return {
        "count": len(values),
        "mean": statistics.mean(values) if values else None,
        "p50": percentile(values, 0.5),
        "p95": percentile(values, 0.95),
    }


def summarize(results: list[dict]) -> dict:
    def rate(key: str, rows: list[dict] = results) -> dict:
        measured = [row[key] for row in rows if row.get(key) is not None]
        return {"passed": sum(measured), "measured": len(measured), "rate": sum(measured) / len(measured) if measured else None}

    latencies = [row["latency_s"] for row in results]
    costs = [row["reported_cost_usd"] for row in results if row.get("reported_cost_usd") is not None]
    categories = sorted({row.get("category", "uncategorized") for row in results})
    pages = {
        page for row in results for source in row.get("sources", [])
        if (page := canonical_source(source)) is not None
    }
    source_types = Counter(
        "documentation" if host == "olake.io" and (path == "/docs" or path.startswith("/docs/"))
        else "repository" if host == "github.com" else "external"
        for host, path in pages
    )
    return {
        "queries": len(results),
        "errors": sum("error" in row for row in results),
        "quality_failures": sum(row.get("proxy_checks_passed") is False for row in results),
        "unmeasured_queries": sum("error" not in row and row.get("proxy_checks_passed") is None for row in results),
        "http_statuses": dict(Counter(str(row.get("status_code", "network_error")) for row in results)),
        "latency_s": latency_summary(latencies),
        "latency_by_cache_state_s": {
            state: latency_summary([row["latency_s"] for row in results if row.get("cache_state", "UNKNOWN") == state])
            for state in sorted({row.get("cache_state", "UNKNOWN") for row in results})
        },
        "index_versions": sorted({row["index_version"] for row in results if row.get("index_version")}),
        "expected_source_match": rate("expected_source_match"),
        "expected_keyword_match": rate("expected_keyword_match"),
        "answerability_proxy_match": rate("answerability_proxy_match"),
        "proxy_checks_passed": rate("proxy_checks_passed"),
        "by_category": {
            category: {
                "queries": len(rows := [row for row in results if row.get("category", "uncategorized") == category]),
                "errors": sum("error" in row for row in rows),
                "quality_failures": sum(row.get("proxy_checks_passed") is False for row in rows),
                **{key: rate(key, rows) for key in (
                    "expected_source_match", "expected_keyword_match", "answerability_proxy_match", "proxy_checks_passed"
                )},
            }
            for category in categories
        },
        "source_variety": {"unique_pages": len(pages), "unique_pages_by_type": dict(source_types)},
        "reported_cost_usd": {"total": sum(costs) if costs else None, "measured_queries": len(costs)},
        "limitations": [
            "Source matches and keyword presence do not establish factual correctness or citation support.",
            "Answerability is inferred from refusal and clarification wording; manually review this proxy and null expectations.",
            "Unmeasured expectations are not counted as passes. Zero exit status alone does not establish quality.",
            "Latency includes failed requests and is grouped by reported cache headers; cost is unknown unless the API reports it.",
        ],
    }


def reported_cost(data: object) -> float | None:
    if not isinstance(data, dict):
        return None
    usage = data.get("usage")
    cost = data.get("cost_usd", usage.get("cost") if isinstance(usage, dict) else None)
    if type(cost) in (int, float) and math.isfinite(cost) and cost >= 0:
        return float(cost)
    return None


def case_category(case: dict) -> str:
    if case.get("category"):
        return case["category"]
    if case.get("answerable") is False:
        return "refusal"
    if case.get("answerable") is None:
        return "manual_review"
    for source in case.get("expected_sources") or []:
        page = canonical_source(source)
        if page and page[0] == "olake.io":
            path = page[1]
            for prefix, category in (
                ("/docs/fusion/", "fusion"), ("/docs/connectors/", "connectors"),
                ("/docs/writers/", "destinations"), ("/docs/install/", "deployment"),
            ):
                if path.startswith(prefix):
                    return category
    return "general"


def retrieval_score_header(response: httpx.Response) -> float | None:
    try:
        score = float(response.headers.get("x-retrieval-score", ""))
    except ValueError:
        return None
    return score if math.isfinite(score) and 0 <= score <= 1 else None


def run_cases(cases: list[dict], client: httpx.Client, base_url: str, interval: float = 3.1) -> list[dict]:
    results = []
    for index, case in enumerate(cases):
        if index and interval:
            time.sleep(interval)
        started = time.perf_counter()
        row = {
            "id": case.get("id", index + 1), "question": case["question"],
            "category": case_category(case), "expected_answerable": case.get("answerable"),
        }
        try:
            response = client.post(f"{base_url.rstrip('/')}/ask", json={"question": case["question"]})
            row["status_code"] = response.status_code
            cache_state = response.headers.get("x-cache", "UNKNOWN").upper()
            row["cache_state"] = cache_state if cache_state in {"HIT", "MISS", "COALESCED"} else "UNKNOWN"
            row["index_version"] = response.headers.get("x-index-version")
            row["retrieval_score"] = retrieval_score_header(response)
            row["answer_model"] = response.headers.get("x-answer-model")
            row["refusal_reason"] = response.headers.get("x-refusal-reason")
            recovery = response.headers.get("x-recovery-attempted", "").casefold()
            row["recovery_attempted"] = {"true": True, "false": False}.get(recovery)
            if response.is_error:
                row["error"] = f"HTTP {response.status_code}"
            else:
                try:
                    data = response.json()
                    row.update(score_result(case, data))
                    row["reported_cost_usd"] = reported_cost(data)
                except (ValueError, TypeError):
                    row["error"] = "Response was not valid JSON"
        except httpx.HTTPError as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        row["latency_s"] = time.perf_counter() - started
        results.append(row)
        print(f"[{row.get('status_code', 'ERROR')}, {row['latency_s']:.2f}s] {case['question']}")
        print(f"  {row.get('error', str(row.get('answer', ''))[:300])}")
        if "error" not in row:
            print(f"  Sources: {row.get('sources', [])}; proxy checks passed: {row.get('proxy_checks_passed')}")
    return results


def load_calibration_records(path: Path) -> list[dict]:
    """Read manually labeled retrieval scores; never query or change the index."""
    content = path.read_text(encoding="utf-8")
    try:
        rows = json.loads(content)
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in content.splitlines() if line.strip()]
    if isinstance(rows, dict):
        rows = rows.get("records", [rows])
    if not isinstance(rows, list) or not rows:
        raise ValueError("Calibration input must be a nonempty JSON array or JSONL of {score, answerable} records")
    for number, row in enumerate(rows, 1):
        if (
            not isinstance(row, dict)
            or type(row.get("score")) not in (int, float)
            or not math.isfinite(row["score"])
            or not 0 <= row["score"] <= 1
            or type(row.get("answerable")) is not bool
        ):
            raise ValueError(f"{path}: record {number}: score must be finite in [0, 1] and answerable must be boolean")
    return rows


def calibrate_threshold(records: list[dict]) -> dict:
    """Sweep the same score >= threshold gate used at runtime, using labels only."""
    if not records:
        raise ValueError("Calibration requires labeled score records")
    thresholds = sorted({0.0, *(float(row["score"]) for row in records), math.nextafter(1.0, math.inf)})
    measurements = []
    for threshold in thresholds:
        tp = sum(row["score"] >= threshold and row["answerable"] for row in records)
        fp = sum(row["score"] >= threshold and not row["answerable"] for row in records)
        fn = sum(row["score"] < threshold and row["answerable"] for row in records)
        tn = len(records) - tp - fp - fn
        measurements.append({
            "threshold": threshold,
            "true_positives": tp, "false_positives": fp, "true_negatives": tn, "false_negatives": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "accuracy": (tp + tn) / len(records),
        })
    positives = sum(row["answerable"] for row in records)
    recommended = None
    if 0 < positives < len(records):
        best = max(measurements, key=lambda row: (row["f1"] or 0, row["precision"] or 0, row["recall"] or 0, row["threshold"]))
        recommended = best["threshold"]
    return {
        "mode": "offline_calibration",
        "summary": {
            "records": len(records), "answerable": positives, "unanswerable": len(records) - positives,
            "proxy_labeled_records": sum(row.get("label_source") == "case_answerability_proxy" for row in records),
            "recommended_threshold": recommended,
            "selection_rule": "Maximum F1; ties prefer precision, recall, then the higher threshold.",
            "limitations": [
                "Labels must describe whether the retrieved evidence supports an answer, not whether an answer exists somewhere.",
                "This evaluates only the score gate, not generated answers or complete retrieval recovery.",
                "Use held-out representative cases before changing the configured threshold; no settings or index data are changed.",
                "Exported case-answerability labels are provisional proxies and require review against the retrieved evidence.",
                "No recommendation is made unless both answerable and unanswerable labels are present.",
            ],
        },
        "thresholds": measurements,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url", nargs="?", default="http://localhost:8080")
    parser.add_argument("--questions", type=Path, default=Path(__file__).with_name("questions.jsonl"))
    parser.add_argument("--output", type=Path, help="Save full results and summary as JSON")
    parser.add_argument("--limit", type=int, help="Only run the first N cases")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--interval", type=float, default=3.1, help="Delay between requests (default respects 20/minute)")
    parser.add_argument("--calibrate", type=Path, metavar="PATH", help="Offline threshold sweep over JSON/JSONL {score, answerable} labels; sends no API requests")
    parser.add_argument("--score-output", type=Path, metavar="PATH", help="Export labeled score JSONL from evaluation; case labels are proxies requiring evidence review")
    args = parser.parse_args(argv)
    if args.interval < 0 or args.timeout <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error("interval must be nonnegative; timeout and limit must be positive")
    if args.calibrate and args.score_output:
        parser.error("--score-output applies to API evaluation, not offline --calibrate")
    if args.calibrate:
        try:
            report = calibrate_threshold(load_calibration_records(args.calibrate))
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
    else:
        cases = load_cases(args.questions)
        if args.limit is not None:
            cases = cases[:args.limit]
        with httpx.Client(timeout=args.timeout) as client:
            results = run_cases(cases, client, args.base_url, args.interval)
        report = {"summary": summarize(results), "results": results}
        if args.score_output:
            labels = [
                {
                    "id": row["id"], "question": row["question"], "category": row["category"],
                    "score": row["retrieval_score"], "answerable": row["expected_answerable"],
                    "label_source": "case_answerability_proxy", "needs_evidence_review": True,
                }
                for row in results
                if "error" not in row and row.get("retrieval_score") is not None
                and type(row.get("expected_answerable")) is bool
            ]
            args.score_output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in labels), encoding="utf-8")
            report["summary"]["exported_score_records"] = len(labels)
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 1 if report["summary"].get("errors") or report["summary"].get("quality_failures") else 0


if __name__ == "__main__":
    raise SystemExit(main())
