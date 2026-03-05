"""ChromaDB index writer for chunk records."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import suppress

import chromadb
from chromadb.api.models.Collection import Collection

from app.types import ChunkRecord


class ChromaIndexer:
    def __init__(self, *, persist_dir: str, collection_name: str) -> None:
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection_name = collection_name

    def get_collection(self) -> Collection:
        return self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def rebuild_collection(self) -> Collection:
        with suppress(Exception):
            self._client.delete_collection(self._collection_name)
        return self.get_collection()

    def add_chunks(
        self,
        *,
        records: Iterable[ChunkRecord],
        embeddings: list[list[float]],
    ) -> int:
        rows = list(records)
        if not rows:
            return 0
        if len(rows) != len(embeddings):
            raise ValueError("Number of records and embeddings must match")

        ids = [row.chunk_id for row in rows]
        docs = [row.text for row in rows]
        metas = [
            {
                "source_file": row.source_file,
                "page": row.page,
                "chunk_id": row.chunk_id,
                "char_start": row.char_start,
                "char_end": row.char_end,
            }
            for row in rows
        ]

        collection = self.get_collection()
        collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
        return len(rows)
