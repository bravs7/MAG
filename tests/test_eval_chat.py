from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.config import AppConfig
from app.eval_chat import (
    EvalItem,
    EvalSettings,
    _load_questions,
    _run_eval_items,
    _run_single_question,
    _write_versions,
)


def _fake_reply() -> SimpleNamespace:
    return SimpleNamespace(
        content="Odpowiedź testowa",
        sources=[],
        config_fingerprint={"retrieval_summary": {"has_context": False}},
    )


def test_load_questions_requires_category(tmp_path: Path) -> None:
    path = tmp_path / "questions.jsonl"
    path.write_text('{"id":"q1","question":"Kim był Mieszko I?"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="missing category"):
        _load_questions(path)


def test_load_questions_parses_required_fields(tmp_path: Path) -> None:
    path = tmp_path / "questions.jsonl"
    path.write_text(
        '{"id":"q1","category":"in_scope_factual","question":"Kim był Mieszko I?"}\n',
        encoding="utf-8",
    )

    items = _load_questions(path)
    assert len(items) == 1
    assert items[0].question_id == "q1"
    assert items[0].category == "in_scope_factual"
    assert items[0].question_text == "Kim był Mieszko I?"


def test_run_single_question_passes_eval_overrides() -> None:
    captured_kwargs: dict[str, object] = {}

    class CaptureService:
        def respond(self, thread_id: str, user_text: str, **kwargs: object) -> SimpleNamespace:
            del thread_id, user_text
            captured_kwargs.update(kwargs)
            return _fake_reply()

    settings = EvalSettings(seed=42, timeout_seconds=15, temperature=0.0, top_p=1.0)
    item = EvalItem(
        question_id="q1",
        category="in_scope_factual",
        question_text="Kim był Mieszko I?",
    )
    row = _run_single_question(
        service=CaptureService(),  # type: ignore[arg-type]
        item=item,
        thread_id="thread-1",
        settings=settings,
        index=1,
        total=1,
    )

    assert row["status"] == "ok"
    assert captured_kwargs["temperature"] == 0.0
    assert captured_kwargs["top_p"] == 1.0
    assert captured_kwargs["seed"] == 42
    assert captured_kwargs["request_timeout_seconds"] == 15.0
    assert captured_kwargs["disable_summarization"] is True


def test_run_eval_items_continues_after_timeout() -> None:
    class TimeoutThenOkService:
        def __init__(self) -> None:
            self.calls = 0

        def respond(self, thread_id: str, user_text: str, **kwargs: object) -> SimpleNamespace:
            del thread_id, user_text, kwargs
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("simulated timeout")
            return _fake_reply()

    service = TimeoutThenOkService()
    settings = EvalSettings(seed=7, timeout_seconds=1, temperature=0.0, top_p=1.0)
    items = [
        EvalItem(question_id="q1", category="out_of_scope", question_text="Pytanie 1"),
        EvalItem(question_id="q2", category="out_of_scope", question_text="Pytanie 2"),
    ]

    rows = _run_eval_items(
        service=service,  # type: ignore[arg-type]
        items=items,
        thread_id="thread-1",
        settings=settings,
    )

    assert service.calls == 2
    assert [row["status"] for row in rows] == ["timeout", "ok"]


def test_write_versions_includes_eval_settings(tmp_path: Path) -> None:
    cfg = AppConfig()
    settings = EvalSettings(
        seed=1234,
        timeout_seconds=90,
        temperature=0.0,
        top_p=1.0,
        max_questions=10,
    )
    versions_path = tmp_path / "versions.json"

    _write_versions(
        cfg,
        versions_path,
        settings=settings,
        questions_path=Path("eval/questions_pl.jsonl"),
        question_count=10,
    )

    payload = json.loads(versions_path.read_text(encoding="utf-8"))
    assert payload["eval"]["seed"] == 1234
    assert payload["eval"]["timeout_seconds"] == 90
    assert payload["eval"]["temperature"] == 0.0
    assert payload["eval"]["top_p"] == 1.0
    assert payload["eval"]["max_questions"] == 10
    assert payload["eval"]["question_count"] == 10
