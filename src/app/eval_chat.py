"""E2E CLI for evaluating chat-teacher behavior in Phase 1."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from app.chat.service import ChatService
from app.config import AppConfig
from app.eval_io import load_question_items
from app.logging import configure_logging, get_logger

logger = get_logger(__name__)

NO_CONTEXT_PREFIX = "Nie wiem na podstawie dostarczonych materiałów"
FORBIDDEN_MARKERS = [
    "z tego co pamiętam",
    "z tego co pamietam",
    "wydaje mi się",
    "wydaje mi sie",
    "chyba",
    "prawdopodobnie",
]
KEY_DEPENDENCIES = [
    "chromadb",
    "pymupdf",
    "ollama",
    "pydantic",
    "python-dotenv",
    "llama-index-core",
    "ragas",
    "pytest",
    "ruff",
]


@dataclass(slots=True)
class EvalItem:
    question_id: str
    category: str
    question_text: str


@dataclass(slots=True)
class EvalSettings:
    seed: int = 1234
    timeout_seconds: int = 90
    temperature: float = 0.0
    top_p: float = 1.0
    max_questions: int | None = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E2E chat evaluation and save artifacts")
    parser.add_argument(
        "--questions",
        default="eval/questions_pl.jsonl",
        help="Path to JSONL with questions",
    )
    parser.add_argument(
        "--thread-id",
        default=None,
        help="Optional existing thread id; if missing new one is created",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for deterministic eval setup",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=90,
        help="Per-request timeout used for model HTTP calls",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Decoding temperature override for eval",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Decoding top_p override for eval",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Optional limit for quick runs",
    )
    args = parser.parse_args()

    configure_logging()
    cfg = AppConfig.from_env()
    cfg.ensure_dirs()

    settings = EvalSettings(
        seed=args.seed,
        timeout_seconds=args.timeout_seconds,
        temperature=args.temperature,
        top_p=args.top_p,
        max_questions=args.max_questions,
    )
    _apply_seed(settings.seed)

    questions_path = Path(args.questions)
    items = _load_questions(questions_path)
    if settings.max_questions is not None:
        items = items[: settings.max_questions]
    if not items:
        raise RuntimeError(f"No valid questions loaded from {questions_path}")

    service = ChatService(cfg)
    thread_id = args.thread_id or service.create_thread(title="e2e-eval")

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = results_dir / f"e2e_chat_{timestamp}.jsonl"
    csv_path = results_dir / f"e2e_chat_{timestamp}.csv"
    report_path = results_dir / f"e2e_report_{timestamp}.md"
    versions_path = results_dir / "versions.json"

    rows = _run_eval_items(
        service=service,
        items=items,
        thread_id=thread_id,
        settings=settings,
    )
    _write_jsonl(rows, jsonl_path)
    _write_csv(rows, csv_path)
    report = _build_report(
        rows,
        thread_id=thread_id,
        questions_path=questions_path,
        settings=settings,
    )
    report_path.write_text(report, encoding="utf-8")
    _write_versions(
        cfg,
        versions_path,
        settings=settings,
        questions_path=questions_path,
        question_count=len(items),
    )

    print(f"Thread ID: {thread_id}")
    print(f"JSONL: {jsonl_path}")
    print(f"CSV: {csv_path}")
    print(f"REPORT: {report_path}")
    print(f"VERSIONS: {versions_path}")


def _apply_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np  # type: ignore
    except Exception:
        return
    np.random.seed(seed)


def _run_eval_items(
    *,
    service: ChatService,
    items: list[EvalItem],
    thread_id: str,
    settings: EvalSettings,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        rows.append(
            _run_single_question(
                service=service,
                item=item,
                thread_id=thread_id,
                settings=settings,
                index=index,
                total=len(items),
            )
        )
    return rows


def _run_single_question(
    *,
    service: ChatService,
    item: EvalItem,
    thread_id: str,
    settings: EvalSettings,
    index: int,
    total: int,
) -> dict[str, Any]:
    logger.info("E2E question %s/%s: %s", index, total, item.question_id)
    started = time.perf_counter()
    try:
        reply = service.respond(
            thread_id=thread_id,
            user_text=item.question_text,
            temperature=settings.temperature,
            top_p=settings.top_p,
            seed=settings.seed,
            request_timeout_seconds=float(settings.timeout_seconds),
            disable_summarization=True,
        )
    except TimeoutError as exc:
        latency_seconds = time.perf_counter() - started
        logger.warning("E2E timeout %s after %.2fs: %s", item.question_id, latency_seconds, exc)
        return _timeout_row(item=item, latency_seconds=latency_seconds, reason=str(exc))
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        latency_seconds = time.perf_counter() - started
        logger.exception("E2E error for %s", item.question_id)
        return _error_row(item=item, latency_seconds=latency_seconds, reason=str(exc))

    latency_seconds = time.perf_counter() - started
    logger.info(
        "E2E completed %s in %.2fs (citations=%s)",
        item.question_id,
        latency_seconds,
        len(reply.sources),
    )

    retrieval_summary = dict(reply.config_fingerprint.get("retrieval_summary", {}))
    citations = [
        {
            "source_file": src.source_file,
            "page": src.page,
            "chunk_id": src.chunk_id,
            "score": src.score,
        }
        for src in reply.sources[:3]
    ]

    forbidden_hits = _find_forbidden_markers(reply.content)
    has_context = bool(retrieval_summary.get("has_context", False))

    return {
        "question_id": item.question_id,
        "category": item.category,
        "question_text": item.question_text,
        "status": "ok",
        "error": "",
        "reply_text": reply.content,
        "latency_seconds": round(latency_seconds, 4),
        "has_citations": bool(reply.sources),
        "citations_count": len(reply.sources),
        "retrieval_summary": {
            "retrieved_count": retrieval_summary.get("retrieved_count"),
            "best_score": retrieval_summary.get("best_score"),
            "has_context": has_context,
            "lexical_fallback_used": retrieval_summary.get("lexical_fallback_used"),
            "query_evidence": retrieval_summary.get("query_evidence"),
            "phrase_norm": retrieval_summary.get("phrase_norm"),
            "main_keyword": retrieval_summary.get("main_keyword"),
        },
        "top_citations": citations,
        "forbidden_markers": forbidden_hits,
        "no_context_prefix_present": NO_CONTEXT_PREFIX.lower() in reply.content.lower(),
    }


def _timeout_row(*, item: EvalItem, latency_seconds: float, reason: str) -> dict[str, Any]:
    return {
        "question_id": item.question_id,
        "category": item.category,
        "question_text": item.question_text,
        "status": "timeout",
        "error": reason,
        "reply_text": "",
        "latency_seconds": round(latency_seconds, 4),
        "has_citations": False,
        "citations_count": 0,
        "retrieval_summary": _empty_retrieval_summary(),
        "top_citations": [],
        "forbidden_markers": [],
        "no_context_prefix_present": False,
    }


def _error_row(*, item: EvalItem, latency_seconds: float, reason: str) -> dict[str, Any]:
    return {
        "question_id": item.question_id,
        "category": item.category,
        "question_text": item.question_text,
        "status": "error",
        "error": reason,
        "reply_text": "",
        "latency_seconds": round(latency_seconds, 4),
        "has_citations": False,
        "citations_count": 0,
        "retrieval_summary": _empty_retrieval_summary(),
        "top_citations": [],
        "forbidden_markers": [],
        "no_context_prefix_present": False,
    }


def _empty_retrieval_summary() -> dict[str, Any]:
    return {
        "retrieved_count": None,
        "best_score": None,
        "has_context": None,
        "lexical_fallback_used": None,
        "query_evidence": None,
        "phrase_norm": None,
        "main_keyword": None,
    }


def _load_questions(path: Path) -> list[EvalItem]:
    return load_question_items(path, EvalItem)


def _find_forbidden_markers(reply_text: str) -> list[str]:
    normalized = reply_text.lower()
    return [marker for marker in FORBIDDEN_MARKERS if marker in normalized]


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "question_id",
        "category",
        "status",
        "error",
        "question_text",
        "latency_seconds",
        "has_context",
        "retrieved_count",
        "best_score",
        "lexical_fallback_used",
        "query_evidence",
        "main_keyword",
        "phrase_norm",
        "has_citations",
        "citations_count",
        "forbidden_markers",
    ]

    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            summary = row["retrieval_summary"]
            writer.writerow(
                {
                    "question_id": row["question_id"],
                    "category": row["category"],
                    "status": row["status"],
                    "error": row["error"],
                    "question_text": row["question_text"],
                    "latency_seconds": row["latency_seconds"],
                    "has_context": summary.get("has_context"),
                    "retrieved_count": summary.get("retrieved_count"),
                    "best_score": summary.get("best_score"),
                    "lexical_fallback_used": summary.get("lexical_fallback_used"),
                    "query_evidence": summary.get("query_evidence"),
                    "main_keyword": summary.get("main_keyword"),
                    "phrase_norm": summary.get("phrase_norm"),
                    "has_citations": row["has_citations"],
                    "citations_count": row["citations_count"],
                    "forbidden_markers": ",".join(row["forbidden_markers"]),
                }
            )


def _build_report(
    rows: list[dict[str, Any]],
    *,
    thread_id: str,
    questions_path: Path,
    settings: EvalSettings,
) -> str:
    total = len(rows)
    ok_rows = [row for row in rows if row["status"] == "ok"]
    timeout_rows = [row for row in rows if row["status"] == "timeout"]
    error_rows = [row for row in rows if row["status"] == "error"]

    context_true = [
        row
        for row in ok_rows
        if row["retrieval_summary"].get("has_context") is True
    ]
    context_false = [
        row
        for row in ok_rows
        if row["retrieval_summary"].get("has_context") is False
    ]

    with_citations_when_context = [row for row in context_true if row["has_citations"]]
    without_citations_when_no_context = [row for row in context_false if not row["has_citations"]]
    forbidden_rows = [row for row in ok_rows if row["forbidden_markers"]]

    ok_ratio = _safe_ratio(len(ok_rows), total)
    timeout_ratio = _safe_ratio(len(timeout_rows), total)

    metric_context_citation = _safe_ratio(len(with_citations_when_context), len(context_true))
    metric_no_context_no_citation = _safe_ratio(
        len(without_citations_when_no_context),
        len(context_false),
    )

    ok_latencies = [float(row["latency_seconds"]) for row in ok_rows]
    avg_latency = statistics.fmean(ok_latencies) if ok_latencies else 0.0
    median_latency = statistics.median(ok_latencies) if ok_latencies else 0.0

    category_metrics = _build_category_metrics(rows)
    worst_cases = _select_worst_cases_by_category(rows, max_total=10, max_per_category=2)

    lines = [
        "# E2E Chat Report",
        "",
        f"- Thread ID: `{thread_id}`",
        f"- Questions file: `{questions_path}`",
        f"- Total questions: **{total}**",
        (
            "- Eval settings: "
            f"seed={settings.seed}, timeout_seconds={settings.timeout_seconds}, "
            f"temperature={settings.temperature}, top_p={settings.top_p}, "
            f"max_questions={settings.max_questions}"
        ),
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| OK rate | {ok_ratio * 100:.1f}% ({len(ok_rows)}/{total}) |",
        f"| Timeout rate | {timeout_ratio * 100:.1f}% ({len(timeout_rows)}/{total}) |",
        f"| Timeout count | {len(timeout_rows)} |",
        f"| Error count | {len(error_rows)} |",
        (
            "| Citation rate when has_context=True "
            f"| {metric_context_citation * 100:.1f}% "
            f"({len(with_citations_when_context)}/{len(context_true)}) |"
        ),
        (
            "| No-citation rate when has_context=False "
            f"| {metric_no_context_no_citation * 100:.1f}% "
            f"({len(without_citations_when_no_context)}/{len(context_false)}) |"
        ),
        f"| Forbidden marker hits | {len(forbidden_rows)} |",
        f"| Avg latency (status=ok) | {avg_latency:.3f}s |",
        f"| Median latency (status=ok) | {median_latency:.3f}s |",
        "",
        "## Per Category Metrics",
        "",
        (
            "| Category | Count | citation_rate_when_has_context_true | "
            "no_citation_rate_when_has_context_false | timeout_count | "
            "forbidden_marker_hits | avg_latency_seconds (ok only) |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for metric in category_metrics:
        lines.append(
            "| "
            f"{metric['category']} | "
            f"{metric['count']} | "
            f"{metric['citation_rate_when_has_context_true'] * 100:.1f}% | "
            f"{metric['no_citation_rate_when_has_context_false'] * 100:.1f}% | "
            f"{metric['timeout_count']} | "
            f"{metric['forbidden_marker_hits']} | "
            f"{metric['avg_latency_seconds']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Worst Cases (Grouped by Category)",
            "",
            (
                "| Category | Question ID | Status | Reason | has_context | "
                "has_citations | citations_count | best_score |"
            ),
            "|---|---|---|---|---:|---:|---:|---:|",
        ]
    )

    if worst_cases:
        for case in worst_cases:
            summary = case["retrieval_summary"]
            lines.append(
                "| "
                f"{case['category']} | "
                f"{case['question_id']} | "
                f"{case['status']} | "
                f"{case['_reason']} | "
                f"{summary.get('has_context')} | "
                f"{case['has_citations']} | "
                f"{case['citations_count']} | "
                f"{summary.get('best_score')} |"
            )
    else:
        lines.append("| - | - | - | No severe failures detected | - | - | - | - |")

    lines.extend(["", "### Case Details", ""])
    for case in worst_cases:
        header = (
            f"- **{case['question_id']}** "
            f"[{case['category']}] "
            f"({case['_reason']}, {case['status']})"
        )
        lines.append(header)
        lines.append(f"  Question: {case['question_text']}")
        lines.append(f"  Reply: {_excerpt(case['reply_text'])}")

    return "\n".join(lines) + "\n"


def _build_category_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    categories = sorted({row["category"] for row in rows})
    metrics: list[dict[str, Any]] = []
    for category in categories:
        category_rows = [row for row in rows if row["category"] == category]
        category_ok = [row for row in category_rows if row["status"] == "ok"]
        category_ctx_true = [
            row
            for row in category_ok
            if row["retrieval_summary"].get("has_context") is True
        ]
        category_ctx_false = [
            row
            for row in category_ok
            if row["retrieval_summary"].get("has_context") is False
        ]
        category_ctx_true_with_citations = [
            row for row in category_ctx_true if row["has_citations"]
        ]
        category_ctx_false_without_citations = [
            row for row in category_ctx_false if not row["has_citations"]
        ]
        category_ok_latencies = [float(row["latency_seconds"]) for row in category_ok]

        metrics.append(
            {
                "category": category,
                "count": len(category_rows),
                "citation_rate_when_has_context_true": _safe_ratio(
                    len(category_ctx_true_with_citations),
                    len(category_ctx_true),
                ),
                "no_citation_rate_when_has_context_false": _safe_ratio(
                    len(category_ctx_false_without_citations),
                    len(category_ctx_false),
                ),
                "timeout_count": sum(
                    1 for row in category_rows if row["status"] == "timeout"
                ),
                "forbidden_marker_hits": sum(
                    1 for row in category_ok if row["forbidden_markers"]
                ),
                "avg_latency_seconds": (
                    statistics.fmean(category_ok_latencies) if category_ok_latencies else 0.0
                ),
            }
        )
    return metrics


def _select_worst_cases_by_category(
    rows: list[dict[str, Any]],
    *,
    max_total: int,
    max_per_category: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for row in rows:
        score, reason = _row_severity(row)
        if score <= 0:
            continue
        enriched = dict(row)
        enriched["_reason"] = reason
        grouped.setdefault(row["category"], []).append((score, enriched))

    if not grouped:
        return []

    for scored_rows in grouped.values():
        scored_rows.sort(key=lambda item: (-item[0], item[1]["question_id"]))

    selected: list[dict[str, Any]] = []
    per_category_taken = {category: 0 for category in grouped}
    category_indices = {category: 0 for category in grouped}
    categories = sorted(grouped.keys())

    while len(selected) < max_total:
        added = False
        for category in categories:
            if per_category_taken[category] >= max_per_category:
                continue
            index = category_indices[category]
            scored_rows = grouped[category]
            if index >= len(scored_rows):
                continue
            selected.append(scored_rows[index][1])
            per_category_taken[category] += 1
            category_indices[category] += 1
            added = True
            if len(selected) >= max_total:
                break
        if not added:
            break

    return selected


def _row_severity(row: dict[str, Any]) -> tuple[int, str]:
    status = row.get("status", "")
    if status == "timeout":
        return (120, "timeout")
    if status == "error":
        return (110, "error")

    summary = row["retrieval_summary"]
    has_context = summary.get("has_context")
    has_citations = bool(row["has_citations"])

    if has_context is True and not has_citations:
        return (100, "has_context=True but missing citations")
    if has_context is False and has_citations:
        return (95, "has_context=False but citations present")
    if row["forbidden_markers"] and not has_citations:
        return (85, "forbidden marker without citations")
    if has_context is False and not row["no_context_prefix_present"]:
        return (80, "no-context path without no-context prefix")
    return (0, "")


def _excerpt(text: str, limit: int = 280) -> str:
    clean = " ".join(text.split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3] + "..."


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _write_versions(
    cfg: AppConfig,
    output_path: Path,
    *,
    settings: EvalSettings,
    questions_path: Path,
    question_count: int,
) -> None:
    payload = {
        "updated_at": datetime.now(UTC).isoformat(),
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "uv": _get_uv_version(),
        "ollama_host": cfg.ollama_host,
        "model_name": cfg.model_name,
        "embed_model": cfg.embed_model,
        "eval": {
            "seed": settings.seed,
            "timeout_seconds": settings.timeout_seconds,
            "temperature": settings.temperature,
            "top_p": settings.top_p,
            "max_questions": settings.max_questions,
            "questions_path": str(questions_path),
            "question_count": question_count,
        },
        "dependencies": {name: _package_version(name) for name in KEY_DEPENDENCIES},
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _get_uv_version() -> str | None:
    uv_binary = shutil.which("uv")
    if uv_binary is None:
        candidate = Path.home() / ".local" / "bin" / "uv"
        if candidate.exists():
            uv_binary = str(candidate)

    if uv_binary is None:
        return None

    try:
        completed = subprocess.run(
            [uv_binary, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or completed.stderr.strip() or None


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


if __name__ == "__main__":
    main()
