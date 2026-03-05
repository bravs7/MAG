from __future__ import annotations

from pathlib import Path

from app.persistence.repositories import Persistence


def test_thread_isolation_in_message_reads(tmp_path: Path) -> None:
    db_path = tmp_path / "chat.db"
    persistence = Persistence(db_path=db_path)

    persistence.threads.upsert_thread("thread-a")
    persistence.threads.upsert_thread("thread-b")

    persistence.messages.add_user_message("thread-a", "A")
    persistence.messages.add_user_message("thread-b", "B")

    messages_a = persistence.messages.list_messages("thread-a")
    messages_b = persistence.messages.list_messages("thread-b")

    assert len(messages_a) == 1
    assert len(messages_b) == 1
    assert messages_a[0].content == "A"
    assert messages_b[0].content == "B"


def test_thread_state_isolation(tmp_path: Path) -> None:
    db_path = tmp_path / "chat.db"
    persistence = Persistence(db_path=db_path)

    persistence.threads.upsert_thread("thread-a")
    persistence.threads.upsert_thread("thread-b")
    persistence.thread_state.upsert_state(
        thread_id="thread-a",
        summary="A-summary",
        memory_version=1,
    )
    persistence.thread_state.upsert_state(
        thread_id="thread-b",
        summary="B-summary",
        memory_version=2,
    )

    state_a = persistence.thread_state.get_state("thread-a")
    state_b = persistence.thread_state.get_state("thread-b")

    assert state_a.summary == "A-summary"
    assert state_b.summary == "B-summary"
    assert state_a.memory_version == 1
    assert state_b.memory_version == 2
