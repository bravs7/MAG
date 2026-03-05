"""Citation formatting and serialization helpers."""

from __future__ import annotations

from app.types import RetrievedChunk, SourceCitation


def from_retrieved(chunks: list[RetrievedChunk]) -> list[SourceCitation]:
    return [
        SourceCitation(
            source_file=chunk.source_file,
            page=chunk.page,
            chunk_id=chunk.chunk_id,
            score=chunk.score,
        )
        for chunk in chunks
    ]


def format_citation(citation: SourceCitation) -> str:
    if citation.page is None:
        return f"[Źródło: {citation.source_file}, chunk {citation.chunk_id}]"
    return f"[Źródło: {citation.source_file}, s. {citation.page}, chunk {citation.chunk_id}]"


def format_citations_block(citations: list[SourceCitation]) -> str:
    if not citations:
        return ""
    parts = [format_citation(citation) for citation in citations]
    return "\n" + "\n".join(parts)


def citations_to_sources_json(citations: list[SourceCitation]) -> list[dict[str, object]]:
    return [
        {
            "source_file": citation.source_file,
            "page": citation.page,
            "chunk_id": citation.chunk_id,
            "score": round(citation.score, 6),
        }
        for citation in citations
    ]
