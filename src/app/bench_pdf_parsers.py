"""Benchmark PDF parsers for offline Phase 1 extraction quality."""

from __future__ import annotations

import argparse
import csv
import random
import re
import statistics
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.ingest.pdf_parsers import (
    SUPPORTED_PARSERS,
    count_pdf_pages,
    extract_pdf_pages,
    parser_available,
)
from app.logging import configure_logging, get_logger

logger = get_logger(__name__)

COVERAGE_KEYWORDS = [
    "Święty Wojciech",
    "Zjazd gnieźnieński",
    "Mieszko",
    "Bolesław Chrobry",
    "metropolia",
    "966",
    "1000",
]


@dataclass(slots=True)
class ParserBenchRow:
    parser: str
    status: str
    note: str
    total_time_seconds: float
    avg_time_per_page_seconds: float
    pages_total: int
    pages_non_empty: int
    empty_pages_count: int
    total_chars: int
    avg_chars_per_page: float
    median_chars_per_page: float
    lines_total: int
    lines_lt_3_tokens: int
    lines_lt_3_tokens_ratio: float
    avg_tokens_per_line: float
    single_char_tokens_count: int
    single_char_tokens_ratio: float
    coverage_hits_total: int
    coverage_pages_total: int
    keyword_hits: dict[str, int]
    keyword_pages: dict[str, list[int]]

    def to_csv_row(self) -> dict[str, Any]:
        row = {
            "parser": self.parser,
            "status": self.status,
            "note": self.note,
            "total_time_seconds": round(self.total_time_seconds, 6),
            "avg_time_per_page_seconds": round(self.avg_time_per_page_seconds, 6),
            "pages_total": self.pages_total,
            "pages_non_empty": self.pages_non_empty,
            "empty_pages_count": self.empty_pages_count,
            "total_chars": self.total_chars,
            "avg_chars_per_page": round(self.avg_chars_per_page, 4),
            "median_chars_per_page": round(self.median_chars_per_page, 4),
            "lines_total": self.lines_total,
            "lines_lt_3_tokens": self.lines_lt_3_tokens,
            "lines_lt_3_tokens_ratio": round(self.lines_lt_3_tokens_ratio, 6),
            "avg_tokens_per_line": round(self.avg_tokens_per_line, 6),
            "single_char_tokens_count": self.single_char_tokens_count,
            "single_char_tokens_ratio": round(self.single_char_tokens_ratio, 6),
            "coverage_hits_total": self.coverage_hits_total,
            "coverage_pages_total": self.coverage_pages_total,
        }
        for keyword in COVERAGE_KEYWORDS:
            slug = _keyword_slug(keyword)
            row[f"kw_{slug}_hits"] = self.keyword_hits.get(keyword, 0)
            pages = self.keyword_pages.get(keyword, [])
            row[f"kw_{slug}_pages"] = ";".join(str(page) for page in pages)
        return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark PDF parsers and compare extraction quality",
    )
    parser.add_argument(
        "--pdf",
        default=None,
        help="PDF path to benchmark (default: first PDF under data/)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="CSV output path (default: results/pdf_parser_bench_<timestamp>.csv)",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Markdown report path (default: results/pdf_parser_bench_<timestamp>.md)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Optional upper bound for pages to benchmark",
    )
    parser.add_argument(
        "--sample-pages",
        type=int,
        default=None,
        help="Optional deterministic sample size from selected pages",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Deterministic seed for page sampling",
    )
    parser.add_argument(
        "--anchor-pages",
        default="",
        help="Comma-separated 1-index page numbers always included in sample",
    )
    args = parser.parse_args()

    configure_logging()
    pdf_path = Path(args.pdf) if args.pdf else _default_pdf_path(Path("data"))
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = (
        Path(args.out)
        if args.out
        else results_dir / f"pdf_parser_bench_{timestamp}.csv"
    )
    report_path = (
        Path(args.report)
        if args.report
        else results_dir / f"pdf_parser_bench_{timestamp}.md"
    )

    anchor_pages = parse_anchor_pages(args.anchor_pages)
    selected_pages = build_selected_pages(
        pdf_path,
        max_pages=args.max_pages,
        sample_pages=args.sample_pages,
        seed=args.seed,
        anchor_pages=anchor_pages,
    )

    rows = run_pdf_parser_benchmark(
        pdf_path=pdf_path,
        parser_names=list(SUPPORTED_PARSERS),
        max_pages=args.max_pages,
        sample_pages=args.sample_pages,
        seed=args.seed,
        selected_pages=selected_pages,
    )

    write_csv(rows, csv_path)
    write_report(
        rows=rows,
        report_path=report_path,
        pdf_path=pdf_path,
        max_pages=args.max_pages,
        sample_pages=args.sample_pages,
        seed=args.seed,
        anchor_pages=anchor_pages,
        selected_pages=selected_pages,
    )

    print(f"PDF: {pdf_path}")
    print(f"CSV: {csv_path}")
    print(f"REPORT: {report_path}")


def run_pdf_parser_benchmark(
    *,
    pdf_path: Path,
    parser_names: list[str],
    max_pages: int | None,
    sample_pages: int | None,
    seed: int,
    selected_pages: list[int] | None = None,
) -> list[ParserBenchRow]:
    resolved_selected = selected_pages or build_selected_pages(
        pdf_path,
        max_pages=max_pages,
        sample_pages=sample_pages,
        seed=seed,
    )

    rows: list[ParserBenchRow] = []
    for parser_name in parser_names:
        normalized = parser_name.strip().lower()
        available = parser_available(normalized)
        if not available:
            rows.append(
                ParserBenchRow(
                    parser=normalized,
                    status="skipped",
                    note="parser not installed",
                    total_time_seconds=0.0,
                    avg_time_per_page_seconds=0.0,
                    pages_total=len(resolved_selected),
                    pages_non_empty=0,
                    empty_pages_count=len(resolved_selected),
                    total_chars=0,
                    avg_chars_per_page=0.0,
                    median_chars_per_page=0.0,
                    lines_total=0,
                    lines_lt_3_tokens=0,
                    lines_lt_3_tokens_ratio=0.0,
                    avg_tokens_per_line=0.0,
                    single_char_tokens_count=0,
                    single_char_tokens_ratio=0.0,
                    coverage_hits_total=0,
                    coverage_pages_total=0,
                    keyword_hits={keyword: 0 for keyword in COVERAGE_KEYWORDS},
                    keyword_pages={keyword: [] for keyword in COVERAGE_KEYWORDS},
                )
            )
            continue

        logger.info("Benchmark parser=%s pages=%s", normalized, len(resolved_selected))
        start_time = datetime.now(UTC)
        pages = extract_pdf_pages(
            pdf_path,
            parser_name=normalized,
            include_empty=True,
            max_pages=max_pages,
            selected_pages=resolved_selected,
        )
        elapsed = (datetime.now(UTC) - start_time).total_seconds()

        row = _build_parser_row(normalized, pages, elapsed)
        rows.append(row)
    return rows


def build_selected_pages(
    pdf_path: Path,
    *,
    max_pages: int | None,
    sample_pages: int | None,
    seed: int,
    anchor_pages: list[int] | None = None,
) -> list[int]:
    total_pages = count_pdf_pages(pdf_path)
    limit = total_pages if max_pages is None else max(0, min(total_pages, max_pages))
    candidates = list(range(1, limit + 1))
    anchors = sorted({page for page in (anchor_pages or []) if 1 <= page <= limit})
    if sample_pages is None or sample_pages >= len(candidates):
        return sorted(set(candidates).union(anchors))

    rng = random.Random(seed)
    remainder = [page for page in candidates if page not in anchors]
    if sample_pages <= len(anchors):
        return anchors

    extra_count = min(sample_pages - len(anchors), len(remainder))
    sampled = rng.sample(remainder, extra_count)
    return sorted(anchors + sampled)


def write_csv(rows: list[ParserBenchRow], output_path: Path) -> None:
    fieldnames = _csv_fieldnames()
    with output_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv_row())


def write_report(
    *,
    rows: list[ParserBenchRow],
    report_path: Path,
    pdf_path: Path,
    max_pages: int | None,
    sample_pages: int | None,
    seed: int,
    anchor_pages: list[int],
    selected_pages: list[int],
) -> None:
    best = choose_best_overall(rows)
    lines = [
        "# PDF Parser Benchmark",
        "",
        f"- PDF: `{pdf_path}`",
        f"- max_pages: `{max_pages}`",
        f"- sample_pages: `{sample_pages}`",
        f"- anchor_pages: `{anchor_pages}`",
        f"- seed: `{seed}`",
        f"- evaluated_pages_count: `{len(selected_pages)}`",
        f"- evaluated_pages: `{','.join(str(page) for page in selected_pages)}`",
        "",
        "## Results",
        "",
        (
            "| parser | status | total_time_s | pages_non_empty | empty_pages | "
            "total_chars | coverage_hits | coverage_pages | noise_ratio |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in rows:
        lines.append(
            "| "
            f"{row.parser} | "
            f"{row.status} | "
            f"{row.total_time_seconds:.3f} | "
            f"{row.pages_non_empty} | "
            f"{row.empty_pages_count} | "
            f"{row.total_chars} | "
            f"{row.coverage_hits_total} | "
            f"{row.coverage_pages_total} | "
            f"{row.lines_lt_3_tokens_ratio:.3f} |"
        )

    lines.extend(["", "## Keyword Coverage", ""])
    lines.append("| parser | keyword | hits | pages |")
    lines.append("|---|---|---:|---|")
    for row in rows:
        for keyword in COVERAGE_KEYWORDS:
            pages = ",".join(str(page) for page in row.keyword_pages.get(keyword, []))
            lines.append(
                "| "
                f"{row.parser} | {keyword} | {row.keyword_hits.get(keyword, 0)} | {pages} |"
            )

    lines.extend(["", "## Wnioski", ""])
    skipped = [row.parser for row in rows if row.status == "skipped"]
    if skipped:
        lines.append(f"- Pominięte parsery (brak zależności): {', '.join(skipped)}.")
    else:
        lines.append("- Wszystkie parsery były dostępne.")

    if best is None:
        lines.append("- Brak parsera ze statusem `ok`, nie można wybrać rekomendacji.")
    else:
        lines.append(
            "- Rekomendacja `best overall`: "
            f"`{best.parser}` (coverage={best.coverage_hits_total}, "
            f"empty_pages={best.empty_pages_count}, time={best.total_time_seconds:.3f}s)."
        )
        lines.append(
            "- Uzasadnienie: priorytet coverage keywordów, następnie mniej pustych stron, "
            "na końcu czas przetwarzania."
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def choose_best_overall(rows: list[ParserBenchRow]) -> ParserBenchRow | None:
    eligible = [row for row in rows if row.status == "ok"]
    if not eligible:
        return None
    return sorted(
        eligible,
        key=lambda row: (
            -row.coverage_hits_total,
            row.empty_pages_count,
            row.total_time_seconds,
            row.parser,
        ),
    )[0]


def _build_parser_row(
    parser_name: str,
    pages: list[tuple[int, str]],
    elapsed_seconds: float,
) -> ParserBenchRow:
    chars_by_page = [len(text) for _, text in pages]
    pages_total = len(pages)
    pages_non_empty = sum(1 for _, text in pages if text.strip())
    empty_pages_count = pages_total - pages_non_empty
    total_chars = sum(chars_by_page)
    avg_chars = (total_chars / pages_total) if pages_total else 0.0
    median_chars = statistics.median(chars_by_page) if chars_by_page else 0.0

    layout = _layout_noise(pages)
    keyword_hits, keyword_pages = keyword_coverage(pages, COVERAGE_KEYWORDS)
    coverage_hits_total = sum(keyword_hits.values())
    coverage_pages_total = sum(len(set(pages_list)) for pages_list in keyword_pages.values())
    avg_time_per_page = elapsed_seconds / pages_total if pages_total else 0.0

    return ParserBenchRow(
        parser=parser_name,
        status="ok",
        note="",
        total_time_seconds=elapsed_seconds,
        avg_time_per_page_seconds=avg_time_per_page,
        pages_total=pages_total,
        pages_non_empty=pages_non_empty,
        empty_pages_count=empty_pages_count,
        total_chars=total_chars,
        avg_chars_per_page=avg_chars,
        median_chars_per_page=median_chars,
        lines_total=layout["lines_total"],
        lines_lt_3_tokens=layout["lines_lt_3_tokens"],
        lines_lt_3_tokens_ratio=layout["lines_lt_3_tokens_ratio"],
        avg_tokens_per_line=layout["avg_tokens_per_line"],
        single_char_tokens_count=layout["single_char_tokens_count"],
        single_char_tokens_ratio=layout["single_char_tokens_ratio"],
        coverage_hits_total=coverage_hits_total,
        coverage_pages_total=coverage_pages_total,
        keyword_hits=keyword_hits,
        keyword_pages=keyword_pages,
    )


def keyword_coverage(
    pages: list[tuple[int, str]],
    keywords: list[str],
) -> tuple[dict[str, int], dict[str, list[int]]]:
    hits: dict[str, int] = {}
    keyword_pages: dict[str, list[int]] = {}

    for keyword in keywords:
        normalized_keyword = _normalize_text(keyword)
        total_hits = 0
        pages_with_hits: list[int] = []
        for page_no, text in pages:
            normalized_text = _normalize_text(text)
            occurrences = normalized_text.count(normalized_keyword)
            if occurrences > 0:
                total_hits += occurrences
                pages_with_hits.append(page_no)
        hits[keyword] = total_hits
        keyword_pages[keyword] = pages_with_hits

    return hits, keyword_pages


def _layout_noise(pages: list[tuple[int, str]]) -> dict[str, float | int]:
    lines_total = 0
    lines_lt_3_tokens = 0
    tokens_total = 0
    single_char_tokens_count = 0

    for _, text in pages:
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            tokens = line.split()
            token_count = len(tokens)
            if token_count == 0:
                continue
            lines_total += 1
            tokens_total += token_count
            if token_count < 3:
                lines_lt_3_tokens += 1
            single_char_tokens_count += sum(1 for token in tokens if len(token) == 1)

    lines_ratio = (lines_lt_3_tokens / lines_total) if lines_total else 0.0
    avg_tokens = (tokens_total / lines_total) if lines_total else 0.0
    single_char_ratio = (single_char_tokens_count / tokens_total) if tokens_total else 0.0
    return {
        "lines_total": lines_total,
        "lines_lt_3_tokens": lines_lt_3_tokens,
        "lines_lt_3_tokens_ratio": lines_ratio,
        "avg_tokens_per_line": avg_tokens,
        "single_char_tokens_count": single_char_tokens_count,
        "single_char_tokens_ratio": single_char_ratio,
    }


def _default_pdf_path(data_dir: Path) -> Path:
    pdf_files = sorted(data_dir.rglob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {data_dir}")
    return pdf_files[0]


def _normalize_text(text: str) -> str:
    if not text:
        return ""

    # Join words split by line hyphenation (e.g. "gnieź-\nnieński").
    no_hyphen_breaks = re.sub(r"[-‐‑‒–—]\s*\r?\n\s*", "", text)
    # Collapse all whitespace, so keyword matching is robust to line breaks/tabs.
    collapsed = re.sub(r"\s+", " ", no_hyphen_breaks).strip()

    decomposed = unicodedata.normalize("NFKD", collapsed)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return stripped.lower().strip()


def parse_anchor_pages(raw: str | None) -> list[int]:
    if raw is None:
        return []
    text = raw.strip()
    if not text:
        return []

    anchors: list[int] = []
    seen: set[int] = set()
    for part in text.split(","):
        value = part.strip()
        if not value:
            continue
        page = int(value)
        if page < 1:
            raise ValueError(f"anchor page must be >= 1, got: {page}")
        if page not in seen:
            anchors.append(page)
            seen.add(page)
    return sorted(anchors)


def _csv_fieldnames() -> list[str]:
    base = [
        "parser",
        "status",
        "note",
        "total_time_seconds",
        "avg_time_per_page_seconds",
        "pages_total",
        "pages_non_empty",
        "empty_pages_count",
        "total_chars",
        "avg_chars_per_page",
        "median_chars_per_page",
        "lines_total",
        "lines_lt_3_tokens",
        "lines_lt_3_tokens_ratio",
        "avg_tokens_per_line",
        "single_char_tokens_count",
        "single_char_tokens_ratio",
        "coverage_hits_total",
        "coverage_pages_total",
    ]
    dynamic: list[str] = []
    for keyword in COVERAGE_KEYWORDS:
        slug = _keyword_slug(keyword)
        dynamic.append(f"kw_{slug}_hits")
        dynamic.append(f"kw_{slug}_pages")
    return base + dynamic


def _keyword_slug(keyword: str) -> str:
    normalized = _normalize_text(keyword)
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    safe = normalized.replace(" ", "_").replace("-", "_")
    safe = "".join(ch for ch in safe if ch.isalnum() or ch == "_")
    return safe


if __name__ == "__main__":
    main()
