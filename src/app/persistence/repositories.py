"""Repository layer for threads, messages and thread state."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path

from app.persistence.db import Database
from app.retrieval.citations import citations_to_sources_json
from app.types import ChatMessage, SourceCitation, utc_now_iso


@dataclass(slots=True)
class ThreadStateRecord:
    thread_id: str
    summary: str | None
    memory_version: int
    updated_at: str


class ThreadRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def upsert_thread(self, thread_id: str, title: str | None = None) -> None:
        now = utc_now_iso()
        with self._db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO threads (id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                  title = COALESCE(excluded.title, threads.title),
                  updated_at = excluded.updated_at
                """,
                (thread_id, title, now, now),
            )

    def list_threads(self) -> list[dict[str, str | None]]:
        with self._db.connect() as conn:
            rows = conn.execute(
                "SELECT id, title, created_at, updated_at FROM threads ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]


class MessageRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def add_message(self, message: ChatMessage) -> None:
        sources_json = json.dumps(citations_to_sources_json(message.sources), ensure_ascii=False)
        fingerprint_json = json.dumps(message.config_fingerprint or {}, ensure_ascii=False)

        with self._db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO messages (
                  id, thread_id, role, content, created_at, model, token_count,
                  sources_json, config_fingerprint_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message.id,
                    message.thread_id,
                    message.role,
                    message.content,
                    message.created_at,
                    message.model,
                    message.token_count,
                    sources_json,
                    fingerprint_json,
                ),
            )
            conn.execute(
                "UPDATE threads SET updated_at = ? WHERE id = ?",
                (utc_now_iso(), message.thread_id),
            )

    def list_messages(self, thread_id: str) -> list[ChatMessage]:
        with self._db.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, thread_id, role, content, created_at, model, token_count,
                       sources_json, config_fingerprint_json
                FROM messages
                WHERE thread_id = ?
                ORDER BY created_at ASC
                """,
                (thread_id,),
            ).fetchall()

        messages: list[ChatMessage] = []
        for row in rows:
            raw_sources = json.loads(row["sources_json"] or "[]")
            sources = [
                SourceCitation(
                    source_file=str(item.get("source_file", "unknown")),
                    page=item.get("page"),
                    chunk_id=str(item.get("chunk_id", "unknown")),
                    score=float(item.get("score", 0.0)),
                )
                for item in raw_sources
            ]
            fingerprint = json.loads(row["config_fingerprint_json"] or "{}")
            messages.append(
                ChatMessage(
                    id=row["id"],
                    thread_id=row["thread_id"],
                    role=row["role"],
                    content=row["content"],
                    created_at=row["created_at"],
                    model=row["model"],
                    token_count=row["token_count"],
                    sources=sources,
                    config_fingerprint=fingerprint,
                )
            )
        return messages

    def add_user_message(self, thread_id: str, content: str) -> ChatMessage:
        msg = ChatMessage(
            id=uuid.uuid4().hex,
            thread_id=thread_id,
            role="user",
            content=content,
        )
        self.add_message(msg)
        return msg

    def add_assistant_message(
        self,
        *,
        thread_id: str,
        content: str,
        model: str,
        token_count: int,
        sources: list[SourceCitation],
        config_fingerprint: dict,
    ) -> ChatMessage:
        msg = ChatMessage(
            id=uuid.uuid4().hex,
            thread_id=thread_id,
            role="assistant",
            content=content,
            model=model,
            token_count=token_count,
            sources=sources,
            config_fingerprint=config_fingerprint,
        )
        self.add_message(msg)
        return msg


class ThreadStateRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def get_state(self, thread_id: str) -> ThreadStateRecord:
        query = (
            "SELECT thread_id, summary, memory_version, updated_at "
            "FROM thread_state "
            "WHERE thread_id = ?"
        )
        with self._db.connect() as conn:
            row = conn.execute(query, (thread_id,)).fetchone()

        if row is None:
            return ThreadStateRecord(
                thread_id=thread_id,
                summary=None,
                memory_version=0,
                updated_at=utc_now_iso(),
            )

        return ThreadStateRecord(
            thread_id=row["thread_id"],
            summary=row["summary"],
            memory_version=int(row["memory_version"]),
            updated_at=row["updated_at"],
        )

    def upsert_state(self, *, thread_id: str, summary: str | None, memory_version: int) -> None:
        now = utc_now_iso()
        with self._db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO thread_state (thread_id, summary, memory_version, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                  summary = excluded.summary,
                  memory_version = excluded.memory_version,
                  updated_at = excluded.updated_at
                """,
                (thread_id, summary, memory_version, now),
            )

    def get_preferences(self, thread_id: str) -> dict[str, object]:
        with self._db.connect() as conn:
            row = conn.execute(
                "SELECT preferences_json FROM thread_state WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
        if row is None:
            return {}
        payload = row["preferences_json"] or "{}"
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return {}
        if not isinstance(parsed, dict):
            return {}
        return parsed

    def upsert_preferences(self, *, thread_id: str, preferences: dict[str, object]) -> None:
        now = utc_now_iso()
        payload = json.dumps(preferences, ensure_ascii=False)
        with self._db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO thread_state (
                  thread_id, summary, memory_version, preferences_json, updated_at
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                  preferences_json = excluded.preferences_json,
                  updated_at = excluded.updated_at
                """,
                (thread_id, None, 0, payload, now),
            )


class Persistence:
    def __init__(self, db_path: Path) -> None:
        schema_path = Path(__file__).with_name("schema.sql")
        self.db = Database(db_path=db_path, schema_path=schema_path)
        self.db.init_schema()

        self.threads = ThreadRepository(self.db)
        self.messages = MessageRepository(self.db)
        self.thread_state = ThreadStateRepository(self.db)
