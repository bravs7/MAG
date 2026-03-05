from __future__ import annotations

from app.retrieval.chroma_retriever import _cosine_distance_to_similarity


def test_cosine_distance_conversion() -> None:
    assert _cosine_distance_to_similarity(0.0) == 1.0
    assert _cosine_distance_to_similarity(0.65) == 0.35
    assert _cosine_distance_to_similarity(1.5) == 0.0
