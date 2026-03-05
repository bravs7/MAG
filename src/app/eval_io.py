"""Shared helpers for evaluation question files."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class QuestionRecord:
    question_id: str
    category: str
    question_text: str


def load_question_records(path: Path) -> list[QuestionRecord]:
    if not path.exists():
        raise FileNotFoundError(f"Questions file not found: {path}")

    records: list[QuestionRecord] = []
    with path.open("r", encoding="utf-8") as fp:
        for line_number, line in enumerate(fp, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_number} in {path}") from exc

            question = str(payload.get("question", "")).strip()
            question_id = str(payload.get("id", "")).strip()
            category = str(payload.get("category", "")).strip()

            missing: list[str] = []
            if not question_id:
                missing.append("id")
            if not category:
                missing.append("category")
            if not question:
                missing.append("question")
            if missing:
                fields = ", ".join(missing)
                raise ValueError(
                    f"Invalid question entry at line {line_number} in {path}: missing {fields}"
                )

            records.append(
                QuestionRecord(
                    question_id=question_id,
                    category=category,
                    question_text=question,
                )
            )

    return records


def load_question_items[T](
    path: Path,
    factory: Callable[[str, str, str], T],
) -> list[T]:
    return [
        factory(record.question_id, record.category, record.question_text)
        for record in load_question_records(path)
    ]
