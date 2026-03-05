from __future__ import annotations

from app.memory.window import take_recent_turn_messages, trim_to_token_budget
from app.types import ChatMessage


def test_take_recent_turn_messages() -> None:
    messages = [
        ChatMessage(
            id=str(i), thread_id="t", role="user" if i % 2 == 0 else "assistant", content=f"m{i}"
        )
        for i in range(12)
    ]
    recent = take_recent_turn_messages(messages, n_turns=3)
    assert len(recent) == 6
    assert recent[0].content == "m6"


def test_trim_to_token_budget_removes_oldest_messages() -> None:
    messages = [
        ChatMessage(id=str(i), thread_id="t", role="user", content="x" * 400) for i in range(4)
    ]
    summary, kept = trim_to_token_budget(
        summary="", recent_messages=messages, max_prompt_tokens=150
    )
    assert summary == ""
    assert 1 <= len(kept) < len(messages)
