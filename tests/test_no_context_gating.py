from __future__ import annotations

from app.dialogue.teacher_policy import (
    NO_CONTEXT_MESSAGE,
    build_no_context_response,
    validate_context,
)
from app.types import RetrievedChunk


def test_no_context_when_chunks_empty() -> None:
    assert not validate_context([], similarity_threshold=0.35)
    assert NO_CONTEXT_MESSAGE == "Nie wiem na podstawie dostarczonych materiałów."


def test_no_context_when_best_score_below_threshold() -> None:
    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            source_file="x.pdf",
            page=1,
            text="abc",
            score=0.2,
        )
    ]
    assert not validate_context(chunks, similarity_threshold=0.35)


def test_no_context_response_contains_clarifying_and_next_step() -> None:
    response = build_no_context_response()
    assert "Nie wiem na podstawie dostarczonych materiałów." in response
    assert "Pytanie doprecyzowujące" in response
    assert "wskaż nazwę PDF" in response
