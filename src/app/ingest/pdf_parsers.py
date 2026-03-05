"""PDF parser backends for ingestion and benchmarking."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

ParserName = Literal["pymupdf", "pdfplumber", "pypdf", "pdfminer"]

SUPPORTED_PARSERS: tuple[ParserName, ...] = (
    "pymupdf",
    "pdfplumber",
    "pypdf",
    "pdfminer",
)
DEFAULT_PARSER: ParserName = "pymupdf"


def normalize_parser_name(name: str | None) -> str:
    if name is None:
        return DEFAULT_PARSER
    normalized = name.strip().lower()
    if not normalized:
        return DEFAULT_PARSER
    return normalized


def parser_available(name: str) -> bool:
    normalized = normalize_parser_name(name)
    if normalized == "pymupdf":
        try:
            import fitz  # noqa: F401
        except Exception:
            return False
        return True

    if normalized == "pdfplumber":
        try:
            import pdfplumber  # noqa: F401
        except Exception:
            return False
        return True

    if normalized == "pypdf":
        try:
            import pypdf  # noqa: F401
        except Exception:
            return False
        return True

    if normalized == "pdfminer":
        try:
            import pdfminer.high_level  # noqa: F401
        except Exception:
            return False
        return True

    return False


def resolve_ingest_parser(requested: str | None) -> tuple[ParserName, str | None]:
    normalized = normalize_parser_name(requested)
    if normalized not in SUPPORTED_PARSERS:
        return DEFAULT_PARSER, f"Unsupported PDF_PARSER={requested!r}; fallback to pymupdf"

    selected = normalized
    if not parser_available(selected):
        return DEFAULT_PARSER, f"Parser {selected!r} not installed; fallback to pymupdf"

    return selected, None


def load_pdf_pages_for_ingest(pdf_path: Path, *, parser_name: str) -> list[tuple[int, str]]:
    pages = extract_pdf_pages(
        pdf_path,
        parser_name=parser_name,
        include_empty=False,
        max_pages=None,
        selected_pages=None,
    )
    return [(page_no, clean_pdf_text(text)) for page_no, text in pages]


def clean_pdf_text(text: str) -> str:
    if not text:
        return ""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    # Join line-broken words like "misjo-\nnarz" before flattening whitespace.
    normalized = re.sub(
        r"(?<=[A-Za-zĄąĆćĘęŁłŃńÓóŚśŹźŻż])-\n(?=[A-Za-zĄąĆćĘęŁłŃńÓóŚśŹźŻż])",
        "",
        normalized,
    )

    cleaned_lines: list[str] = []
    for line in normalized.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        # Drop bare line-leading indices like "21 Święty Wojciech..."
        stripped = re.sub(r"^\d{1,3}\s+(?=[A-Za-zĄąĆćĘęŁłŃńÓóŚśŹźŻż])", "", stripped)
        cleaned_lines.append(stripped)

    compact = " ".join(cleaned_lines)
    compact = re.sub(r"\s+", " ", compact).strip()
    return compact


def extract_pdf_pages(
    pdf_path: Path,
    *,
    parser_name: str,
    include_empty: bool = True,
    max_pages: int | None = None,
    selected_pages: Sequence[int] | None = None,
) -> list[tuple[int, str]]:
    normalized = normalize_parser_name(parser_name)
    if normalized not in SUPPORTED_PARSERS:
        raise ValueError(f"Unsupported parser: {parser_name!r}")

    selected_set = set(selected_pages) if selected_pages else None
    target_pages = _resolve_target_pages(
        pdf_path,
        max_pages=max_pages,
        selected_pages=selected_set,
    )

    if normalized == "pymupdf":
        pages = _extract_with_pymupdf(pdf_path, target_pages)
    elif normalized == "pdfplumber":
        pages = _extract_with_pdfplumber(pdf_path, target_pages)
    elif normalized == "pypdf":
        pages = _extract_with_pypdf(pdf_path, target_pages)
    else:
        pages = _extract_with_pdfminer(pdf_path, target_pages)

    if include_empty:
        return pages

    return [(page_no, text) for page_no, text in pages if text.strip()]


def count_pdf_pages(pdf_path: Path) -> int:
    import fitz

    with fitz.open(pdf_path) as doc:
        return len(doc)


def _resolve_target_pages(
    pdf_path: Path,
    *,
    max_pages: int | None,
    selected_pages: set[int] | None,
) -> list[int]:
    total_pages = count_pdf_pages(pdf_path)
    page_limit = total_pages if max_pages is None else max(0, min(total_pages, max_pages))
    page_numbers = list(range(1, page_limit + 1))

    if selected_pages is None:
        return page_numbers

    filtered = [page_no for page_no in page_numbers if page_no in selected_pages]
    return filtered


def _extract_with_pymupdf(pdf_path: Path, page_numbers: list[int]) -> list[tuple[int, str]]:
    import fitz

    pages: list[tuple[int, str]] = []
    if not page_numbers:
        return pages

    wanted = set(page_numbers)
    with fitz.open(pdf_path) as doc:
        for page_no, page in enumerate(doc, start=1):
            if page_no not in wanted:
                continue
            text = (page.get_text("text") or "").strip()
            pages.append((page_no, text))
    return pages


def _extract_with_pdfplumber(pdf_path: Path, page_numbers: list[int]) -> list[tuple[int, str]]:
    import pdfplumber

    pages: list[tuple[int, str]] = []
    if not page_numbers:
        return pages

    with pdfplumber.open(pdf_path) as pdf:
        for page_no in page_numbers:
            if page_no > len(pdf.pages):
                continue
            text = (pdf.pages[page_no - 1].extract_text() or "").strip()
            pages.append((page_no, text))
    return pages


def _extract_with_pypdf(pdf_path: Path, page_numbers: list[int]) -> list[tuple[int, str]]:
    from pypdf import PdfReader

    pages: list[tuple[int, str]] = []
    if not page_numbers:
        return pages

    reader = PdfReader(str(pdf_path))
    for page_no in page_numbers:
        if page_no > len(reader.pages):
            continue
        text = (reader.pages[page_no - 1].extract_text() or "").strip()
        pages.append((page_no, text))
    return pages


def _extract_with_pdfminer(pdf_path: Path, page_numbers: list[int]) -> list[tuple[int, str]]:
    from pdfminer.high_level import extract_text

    pages: list[tuple[int, str]] = []
    for page_no in page_numbers:
        text = extract_text(str(pdf_path), page_numbers=[page_no - 1]).strip()
        pages.append((page_no, text))
    return pages
