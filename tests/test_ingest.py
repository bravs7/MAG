from __future__ import annotations

from app.ingest.chunker import ChunkingConfig, chunk_pages
from app.ingest.pdf_parsers import clean_pdf_text


def test_chunk_ids_are_stable_for_same_input() -> None:
    pages = [(1, "Ala ma kota i bardzo lubi historie Polski." * 20)]
    config = ChunkingConfig(chunk_size=120, chunk_overlap=20)

    first = chunk_pages(source_file="test.pdf", pages=pages, config=config)
    second = chunk_pages(source_file="test.pdf", pages=pages, config=config)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert len(first) > 1


def test_clean_pdf_text_normalizes_hyphenation_and_leading_numbers() -> None:
    raw = "21 Święty Wojciech był misjo-\nnarzem.\n\n  Zginął męczeńsko."
    cleaned = clean_pdf_text(raw)

    assert cleaned.startswith("Święty Wojciech był misjonarzem.")
    assert "21 " not in cleaned
    assert "\n" not in cleaned
