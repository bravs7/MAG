from __future__ import annotations

from app.retrieval.hybrid import (
    analyze_query_keywords,
    chunk_contains_keyword,
    chunk_phrase_hit,
    ensure_top_k_contains_evidence,
    has_required_evidence,
    rerank_chunks,
    rerank_chunks_with_keyword_query,
    should_use_lexical_fallback,
)
from app.types import RetrievedChunk


def test_phrase_rerank_prioritizes_full_name_over_single_keyword() -> None:
    query = "Kto to byl Święty Wojciech"
    chunks = [
        RetrievedChunk(
            chunk_id="wojciech-only-high-score",
            source_file="historia.pdf",
            page=12,
            text="Wojciech Roszkowski jest autorem podręczników historycznych.",
            score=0.92,
        ),
        RetrievedChunk(
            chunk_id="swiety-wojciech-phrase",
            source_file="historia.pdf",
            page=23,
            text="Święty Wojciech (ok. 956–997) był biskupem praskim i misjonarzem.",
            score=0.61,
        ),
    ]

    reranked, keyword_query = rerank_chunks(chunks, query)

    assert keyword_query.phrase_norm == "swiety wojciech"
    assert reranked[0].chunk_id == "swiety-wojciech-phrase"
    assert chunk_phrase_hit(reranked[0], keyword_query.phrase_norm)
    assert chunk_contains_keyword(reranked[0], keyword_query.main_keyword)


def test_rerank_is_deterministic_for_same_input() -> None:
    query = "Kto to byl Swiety Wojciech"
    chunks = [
        RetrievedChunk(
            chunk_id="c2",
            source_file="historia.pdf",
            page=2,
            text="To jest fragment o Mieszku I.",
            score=0.70,
        ),
        RetrievedChunk(
            chunk_id="c1",
            source_file="historia.pdf",
            page=23,
            text="Swiety Wojciech i jego misja wsrod Prusow.",
            score=0.65,
        ),
        RetrievedChunk(
            chunk_id="c3",
            source_file="historia.pdf",
            page=5,
            text="Inny fragment historyczny.",
            score=0.88,
        ),
    ]

    keyword_query = analyze_query_keywords(query)
    order1 = [chunk.chunk_id for chunk in rerank_chunks_with_keyword_query(chunks, keyword_query)]
    order2 = [chunk.chunk_id for chunk in rerank_chunks_with_keyword_query(chunks, keyword_query)]

    assert order1 == order2
    assert order1[0] == "c1"


def test_fallback_condition_for_two_keywords() -> None:
    query = "Kto to był Święty Wojciech"
    keyword_query = analyze_query_keywords(query)
    chunks = [
        RetrievedChunk(
            chunk_id="only-one-keyword",
            source_file="historia.pdf",
            page=6,
            text="Wojciech Roszkowski i historia nowożytna.",
            score=0.87,
        )
    ]
    assert should_use_lexical_fallback(chunks, keyword_query) is True

    chunks_with_phrase = [
        RetrievedChunk(
            chunk_id="with-phrase",
            source_file="historia.pdf",
            page=23,
            text="Święty Wojciech był biskupem i męczennikiem.",
            score=0.55,
        )
    ]
    assert should_use_lexical_fallback(chunks_with_phrase, keyword_query) is False


def test_phrase_detection_targets_zjazd_gnieznienski() -> None:
    query = "Czym był Zjazd Gnieźnieński?"
    keyword_query = analyze_query_keywords(query)
    assert keyword_query.phrase_norm is not None
    assert "zjazd gnieznienski" in keyword_query.phrase_norm


def test_phrase_detection_targets_boleslaw_chrobry() -> None:
    query = "Kim był Bolesław Chrobry?"
    keyword_query = analyze_query_keywords(query)
    assert keyword_query.phrase_norm == "boleslaw chrobry"
    assert any(
        "Bolesław Chrobry" in term or "boleslaw chrobry" in term
        for term in keyword_query.phrase_terms
    )


def test_phrase_detection_targets_metropolia_gnieznienska() -> None:
    query = "Jak opisałbyś dziecku znaczenie metropolii gnieźnieńskiej?"
    keyword_query = analyze_query_keywords(query)
    assert keyword_query.phrase_norm is not None
    assert "metropolia gnieznienska" in keyword_query.phrase_norm


def test_ensure_top_k_contains_phrase_evidence_chunk() -> None:
    query = "Kto to był Święty Wojciech"
    keyword_query = analyze_query_keywords(query)
    chunks = [
        RetrievedChunk(
            chunk_id="c0",
            source_file="historia.pdf",
            page=1,
            text="Inny fragment bez tematu.",
            score=0.98,
        ),
        RetrievedChunk(
            chunk_id="c1",
            source_file="historia.pdf",
            page=2,
            text="Także nie ten temat.",
            score=0.95,
        ),
        RetrievedChunk(
            chunk_id="c2",
            source_file="historia.pdf",
            page=23,
            text="Święty Wojciech był biskupem i misjonarzem.",
            score=0.60,
        ),
    ]
    final_chunks = ensure_top_k_contains_evidence(chunks, keyword_query, top_k=2)
    assert len(final_chunks) == 2
    assert has_required_evidence(final_chunks, keyword_query) is True
