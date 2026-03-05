from __future__ import annotations

from app.retrieval.citations import citations_to_sources_json, format_citation
from app.types import SourceCitation


def test_citation_format_with_page() -> None:
    citation = SourceCitation(source_file="historia.pdf", page=12, chunk_id="c-1", score=0.7)
    text = format_citation(citation)
    assert text == "[Źródło: historia.pdf, s. 12, chunk c-1]"


def test_citation_format_without_page() -> None:
    citation = SourceCitation(source_file="historia.pdf", page=None, chunk_id="c-2", score=0.7)
    text = format_citation(citation)
    assert text == "[Źródło: historia.pdf, chunk c-2]"


def test_sources_json_shape() -> None:
    citation = SourceCitation(source_file="historia.pdf", page=None, chunk_id="c-2", score=0.42)
    payload = citations_to_sources_json([citation])
    assert payload == [
        {"source_file": "historia.pdf", "page": None, "chunk_id": "c-2", "score": 0.42}
    ]
