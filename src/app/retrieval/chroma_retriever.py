"""Retrieve top-k chunks from ChromaDB.

Threshold-based no-context gating is handled in ChatService.
"""

from __future__ import annotations

from typing import Any

import chromadb

from app.types import RetrievedChunk


class ChromaRetriever:
    def __init__(self, *, persist_dir: str, collection_name: str) -> None:
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count_chunks(self) -> int:
        return int(self._collection.count())

    def retrieve(
        self,
        *,
        query_embedding: list[float],
        top_k: int,
    ) -> list[RetrievedChunk]:
        if top_k <= 0:
            return []

        response = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        docs = _first(response.get("documents"))
        metas = _first(response.get("metadatas"))
        distances = _first(response.get("distances"))

        results: list[RetrievedChunk] = []
        for doc, meta, distance in zip(docs, metas, distances, strict=False):
            similarity = _cosine_distance_to_similarity(distance)

            meta = meta or {}
            results.append(
                RetrievedChunk(
                    chunk_id=str(meta.get("chunk_id", "unknown")),
                    source_file=str(meta.get("source_file", "unknown")),
                    page=_to_int_or_none(meta.get("page")),
                    text=str(doc or ""),
                    score=similarity,
                )
            )

        return results

    def retrieve_by_document_contains(
        self,
        *,
        phrase: str,
        limit: int,
        fallback_score: float = 1.0,
    ) -> list[RetrievedChunk]:
        if not phrase or limit <= 0:
            return []

        response = self._collection.get(
            where_document={"$contains": phrase},
            limit=limit,
            include=["documents", "metadatas"],
        )

        ids = _first(response.get("ids"))
        docs = _first(response.get("documents"))
        metas = _first(response.get("metadatas"))

        results: list[RetrievedChunk] = []
        for raw_id, doc, meta in zip(ids, docs, metas, strict=False):
            meta = meta or {}
            chunk_id = str(meta.get("chunk_id", raw_id or "unknown"))
            results.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    source_file=str(meta.get("source_file", "unknown")),
                    page=_to_int_or_none(meta.get("page")),
                    text=str(doc or ""),
                    score=float(fallback_score),
                )
            )

        return results

    def list_chunks(self, *, limit: int, offset: int = 0) -> list[RetrievedChunk]:
        if limit <= 0:
            return []

        response = self._collection.get(
            limit=limit,
            offset=max(0, offset),
            include=["documents", "metadatas"],
        )
        ids = _first(response.get("ids"))
        docs = _first(response.get("documents"))
        metas = _first(response.get("metadatas"))

        results: list[RetrievedChunk] = []
        for raw_id, doc, meta in zip(ids, docs, metas, strict=False):
            meta = meta or {}
            chunk_id = str(meta.get("chunk_id", raw_id or "unknown"))
            results.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    source_file=str(meta.get("source_file", "unknown")),
                    page=_to_int_or_none(meta.get("page")),
                    text=str(doc or ""),
                    score=1.0,
                )
            )

        return results


def _first(value: Any) -> list[Any]:
    if not value:
        return []
    if isinstance(value, list) and value and isinstance(value[0], list):
        return list(value[0])
    if isinstance(value, list):
        return value
    return []


def _cosine_distance_to_similarity(distance: float | int | None) -> float:
    if distance is None:
        return 0.0
    numeric = float(distance)
    similarity = 1.0 - numeric
    return max(0.0, min(1.0, similarity))


def _to_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
