from __future__ import annotations

from app.chat.service import _has_unsupported_numeric_claims
from app.types import RetrievedChunk


def test_numeric_guard_detects_year_not_in_context() -> None:
    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            source_file="historia.pdf",
            page=23,
            text="Święty Wojciech (ok. 956-997) był biskupem praskim.",
            score=0.91,
        )
    ]

    generated = "Zjazd gnieźnieński odbył się w 1225 roku."
    assert _has_unsupported_numeric_claims(generated, chunks) is True


def test_numeric_guard_allows_numbers_present_in_context() -> None:
    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            source_file="historia.pdf",
            page=23,
            text="Święty Wojciech (ok. 956-997) był biskupem praskim.",
            score=0.91,
        )
    ]

    generated = "Święty Wojciech żył w latach 956-997."
    assert _has_unsupported_numeric_claims(generated, chunks) is False


def test_numeric_guard_allows_year_range_with_dash_variants() -> None:
    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            source_file="historia.pdf",
            page=23,
            text="Święty Wojciech (ok. 956–997) był biskupem praskim.",
            score=0.91,
        )
    ]

    generated_hyphen = "Święty Wojciech żył w latach 956-997."
    generated_em_dash = "Święty Wojciech żył w latach 956—997."

    assert _has_unsupported_numeric_claims(generated_hyphen, chunks) is False
    assert _has_unsupported_numeric_claims(generated_em_dash, chunks) is False
