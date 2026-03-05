"""Shared domain types used across modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    source_file: str
    page: int | None
    text: str
    char_start: int | None = None
    char_end: int | None = None


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: str
    source_file: str
    page: int | None
    text: str
    score: float


@dataclass(slots=True)
class SourceCitation:
    source_file: str
    page: int | None
    chunk_id: str
    score: float


Role = Literal["user", "assistant", "system"]


@dataclass(slots=True)
class ChatMessage:
    id: str
    thread_id: str
    role: Role
    content: str
    created_at: str = field(default_factory=utc_now_iso)
    model: str | None = None
    token_count: int | None = None
    sources: list[SourceCitation] = field(default_factory=list)
    config_fingerprint: dict[str, Any] | None = None


@dataclass(slots=True)
class AssistantReply:
    content: str
    sources: list[SourceCitation]
    token_count: int
    config_fingerprint: dict[str, Any]
