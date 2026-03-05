"""Pre-flight support checker for eval question sets."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import AppConfig
from app.eval_io import load_question_items
from app.logging import configure_logging, get_logger
from app.retrieval.chroma_retriever import ChromaRetriever
from app.retrieval.hybrid import KeywordQuery, analyze_query_keywords

logger = get_logger(__name__)


@dataclass(slots=True)
class PreflightQuestion:
    question_id: str
    category: str
    question_text: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight support check for eval questions")
    parser.add_argument(
        "--questions",
        default="eval/questions_pl.jsonl",
        help="Path to eval questions JSONL",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="CSV output path (default: results/preflight_<timestamp>.csv)",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Markdown report path (default: results/preflight_<timestamp>.md)",
    )
    args = parser.parse_args()

    configure_logging()
    cfg = AppConfig.from_env()
    cfg.ensure_dirs()

    questions_path = Path(args.questions)
    questions = load_questions(questions_path)
    if not questions:
        raise RuntimeError(f"No questions loaded from {questions_path}")

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.out) if args.out else results_dir / f"preflight_{timestamp}.csv"
    report_path = Path(args.report) if args.report else results_dir / f"preflight_{timestamp}.md"

    retriever = ChromaRetriever(
        persist_dir=str(cfg.chroma_dir),
        collection_name=cfg.collection_name,
    )
    rows = run_preflight(questions=questions, retriever=retriever)
    write_csv(rows, csv_path)
    write_report(rows, report_path, questions_path)

    print(f"QUESTIONS: {questions_path}")
    print(f"CSV: {csv_path}")
    print(f"REPORT: {report_path}")


def load_questions(path: Path) -> list[PreflightQuestion]:
    return load_question_items(path, PreflightQuestion)


def run_preflight(
    *,
    questions: list[PreflightQuestion],
    retriever: ChromaRetriever,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for question in questions:
        keyword_query = analyze_query_keywords(question.question_text)
        search_terms = build_preflight_terms(keyword_query)

        hit_chunks: dict[str, dict[str, Any]] = {}
        hit_terms: dict[str, int] = {}
        for term in search_terms:
            matches = retriever.retrieve_by_document_contains(phrase=term, limit=12)
            if not matches:
                continue
            hit_terms[term] = len(matches)
            for chunk in matches:
                hit_chunks[chunk.chunk_id] = {
                    "chunk_id": chunk.chunk_id,
                    "page": chunk.page,
                    "source_file": chunk.source_file,
                }

        pages = sorted(
            {int(chunk["page"]) for chunk in hit_chunks.values() if chunk.get("page") is not None}
        )
        rows.append(
            {
                "question_id": question.question_id,
                "category": question.category,
                "question_text": question.question_text,
                "keywords": keyword_query.keywords,
                "phrase_norm": keyword_query.phrase_norm,
                "phrase_terms": keyword_query.phrase_terms,
                "main_keyword": keyword_query.main_keyword,
                "supported": bool(hit_chunks),
                "hits_count": len(hit_chunks),
                "hit_pages": pages,
                "hit_terms": hit_terms,
            }
        )
    return rows


def build_preflight_terms(keyword_query: KeywordQuery) -> list[str]:
    terms: list[str] = []
    terms.extend(keyword_query.phrase_terms[:8])
    if keyword_query.main_keyword:
        terms.append(keyword_query.main_keyword)
    for keyword in keyword_query.keywords[:4]:
        terms.append(keyword)
    return _dedupe_preserve([term for term in terms if term and term.strip()])


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "question_id",
        "category",
        "question_text",
        "supported",
        "hits_count",
        "main_keyword",
        "phrase_norm",
        "keywords",
        "phrase_terms",
        "hit_pages",
        "hit_terms",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "question_id": row["question_id"],
                    "category": row["category"],
                    "question_text": row["question_text"],
                    "supported": row["supported"],
                    "hits_count": row["hits_count"],
                    "main_keyword": row["main_keyword"] or "",
                    "phrase_norm": row["phrase_norm"] or "",
                    "keywords": "|".join(row["keywords"]),
                    "phrase_terms": "|".join(row["phrase_terms"]),
                    "hit_pages": ";".join(str(page) for page in row["hit_pages"]),
                    "hit_terms": json.dumps(row["hit_terms"], ensure_ascii=False),
                }
            )


def write_report(
    rows: list[dict[str, Any]],
    output_path: Path,
    questions_path: Path,
) -> None:
    total = len(rows)
    supported = [row for row in rows if row["supported"]]
    unsupported = [row for row in rows if not row["supported"]]
    support_rate = (len(supported) / total) if total else 0.0

    lines = [
        "# Preflight Questions Report",
        "",
        f"- questions_file: `{questions_path}`",
        f"- total_questions: **{total}**",
        f"- supported_questions: **{len(supported)}**",
        f"- unsupported_questions: **{len(unsupported)}**",
        f"- support_rate: **{support_rate * 100:.1f}%**",
        "",
        "## Per Question",
        "",
        "| id | category | supported | hits_count | phrase_norm | hit_pages |",
        "|---|---|---:|---:|---|---|",
    ]
    for row in rows:
        hit_pages = ";".join(str(page) for page in row["hit_pages"])
        lines.append(
            "| "
            f"{row['question_id']} | {row['category']} | {row['supported']} | "
            f"{row['hits_count']} | {row['phrase_norm'] or ''} | {hit_pages} |"
        )

    lines.extend(["", "## Unsupported Questions", ""])
    if not unsupported:
        lines.append("- (none)")
    else:
        for row in unsupported:
            lines.append(f"- `{row['question_id']}` [{row['category']}]: {row['question_text']}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _dedupe_preserve(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        output.append(value)
        seen.add(value)
    return output


if __name__ == "__main__":
    main()
