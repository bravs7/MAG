"""Chunking utilities for ingestion."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from app.types import ChunkRecord


@dataclass(slots=True)
class ChunkingConfig:
    chunk_size: int
    chunk_overlap: int


def chunk_pages(
    *, source_file: str, pages: list[tuple[int, str]], config: ChunkingConfig
) -> list[ChunkRecord]:
    if config.chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if config.chunk_overlap < 0 or config.chunk_overlap >= config.chunk_size:
        raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_size")

    records: list[ChunkRecord] = []
    step = config.chunk_size - config.chunk_overlap

    for page_num, text in pages:
        text_len = len(text)
        start = 0
        chunk_idx = 0
        while start < text_len:
            end = min(text_len, start + config.chunk_size)
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk_id = build_chunk_id(
                    source_file=source_file,
                    page=page_num,
                    chunk_index=chunk_idx,
                    chunk_text=chunk_text,
                )
                records.append(
                    ChunkRecord(
                        chunk_id=chunk_id,
                        source_file=source_file,
                        page=page_num,
                        text=chunk_text,
                        char_start=start,
                        char_end=end,
                    )
                )
            start += step
            chunk_idx += 1

    return records


def build_chunk_id(*, source_file: str, page: int, chunk_index: int, chunk_text: str) -> str:
    digest = hashlib.sha1(chunk_text.encode("utf-8")).hexdigest()[:10]
    return f"{source_file}:p{page}:c{chunk_index}:{digest}"
