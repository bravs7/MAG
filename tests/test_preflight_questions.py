from __future__ import annotations

import csv
from pathlib import Path

from app.preflight_questions import (
    PreflightQuestion,
    load_questions,
    run_preflight,
    write_csv,
    write_report,
)
from app.types import RetrievedChunk


class FakeRetriever:
    def retrieve_by_document_contains(self, *, phrase: str, limit: int) -> list[RetrievedChunk]:
        del limit
        normalized = phrase.lower()
        if "wojciech" in normalized:
            return [
                RetrievedChunk(
                    chunk_id="c1",
                    source_file="historia.pdf",
                    page=23,
                    text="Święty Wojciech był biskupem praskim.",
                    score=1.0,
                )
            ]
        return []


def test_preflight_outputs_csv_and_md(tmp_path: Path) -> None:
    questions = [
        PreflightQuestion(
            question_id="q1",
            category="in_scope_factual",
            question_text="Kto to był Święty Wojciech?",
        ),
        PreflightQuestion(
            question_id="q2",
            category="out_of_scope",
            question_text="Jak działa blockchain?",
        ),
    ]

    rows = run_preflight(questions=questions, retriever=FakeRetriever())  # type: ignore[arg-type]
    assert len(rows) == 2
    assert rows[0]["supported"] is True
    assert rows[1]["supported"] is False

    csv_path = tmp_path / "preflight.csv"
    report_path = tmp_path / "preflight.md"
    write_csv(rows, csv_path)
    write_report(rows, report_path, Path("eval/questions_pl.jsonl"))

    with csv_path.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        data = list(reader)
    assert len(data) == 2
    assert data[0]["supported"] == "True"
    assert "q2" in report_path.read_text(encoding="utf-8")


def test_load_questions_requires_category(tmp_path: Path) -> None:
    path = tmp_path / "questions.jsonl"
    path.write_text('{"id":"q1","question":"Kim był Mieszko I?"}\n', encoding="utf-8")
    try:
        load_questions(path)
    except ValueError as exc:
        assert "category" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError for missing category")
