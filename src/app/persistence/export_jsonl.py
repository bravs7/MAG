"""Export thread messages to JSONL format."""

from __future__ import annotations

import json
from pathlib import Path

from app.persistence.repositories import Persistence
from app.retrieval.citations import citations_to_sources_json


def export_thread_to_jsonl(*, db_path: Path, thread_id: str, output_path: Path) -> int:
    persistence = Persistence(db_path=db_path)
    messages = persistence.messages.list_messages(thread_id)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as fp:
        for msg in messages:
            row = {
                "id": msg.id,
                "thread_id": msg.thread_id,
                "role": msg.role,
                "content": msg.content,
                "created_at": msg.created_at,
                "model": msg.model,
                "token_count": msg.token_count,
                "sources": citations_to_sources_json(msg.sources),
                "config_fingerprint": msg.config_fingerprint,
            }
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1

    return count
