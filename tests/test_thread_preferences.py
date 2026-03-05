from __future__ import annotations

from pathlib import Path

from app.persistence.repositories import Persistence


def test_thread_preferences_are_persisted_per_thread(tmp_path: Path) -> None:
    db_path = tmp_path / "chat.db"
    persistence = Persistence(db_path=db_path)

    persistence.threads.upsert_thread("thread-a")
    persistence.threads.upsert_thread("thread-b")

    persistence.thread_state.upsert_preferences(
        thread_id="thread-a",
        preferences={"max_sentences": 2, "ask_check_question": False},
    )
    persistence.thread_state.upsert_preferences(
        thread_id="thread-b",
        preferences={"max_sentences": 3, "ask_check_question": True},
    )

    prefs_a = persistence.thread_state.get_preferences("thread-a")
    prefs_b = persistence.thread_state.get_preferences("thread-b")

    assert prefs_a["max_sentences"] == 2
    assert prefs_a["ask_check_question"] is False
    assert prefs_b["max_sentences"] == 3
    assert prefs_b["ask_check_question"] is True
