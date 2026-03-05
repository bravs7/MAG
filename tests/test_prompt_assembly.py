from __future__ import annotations

from app.dialogue.prompt_builder import build_prompt, default_system_rules
from app.types import ChatMessage, RetrievedChunk


def test_prompt_section_order_is_deterministic() -> None:
    prompt = build_prompt(
        system_rules=default_system_rules(),
        teacher_rules="teacher-rules",
        summary="summary-text",
        recent_messages=[
            ChatMessage(id="1", thread_id="t", role="user", content="u1"),
            ChatMessage(id="2", thread_id="t", role="assistant", content="a1"),
        ],
        retrieved_chunks=[
            RetrievedChunk(
                chunk_id="c1",
                source_file="historia.pdf",
                page=4,
                text="fragment",
                score=0.78,
            )
        ],
        user_message="Pytanie",
    )

    expected = [
        "## SYSTEM",
        "## CONVERSATION_SUMMARY",
        "## RECENT_TURNS",
        "## RETRIEVED_CONTEXT",
        "## USER_MESSAGE",
        "## OUTPUT_CONSTRAINTS",
    ]

    positions = [prompt.index(token) for token in expected]
    assert positions == sorted(positions)
    assert "## TEACHER_POLICY" not in prompt


def test_prompt_keeps_order_with_empty_optional_sections() -> None:
    prompt = build_prompt(
        system_rules=default_system_rules(),
        teacher_rules="teacher-rules",
        summary=None,
        recent_messages=[],
        retrieved_chunks=[],
        user_message="Pytanie",
    )

    assert "## CONVERSATION_SUMMARY\n(brak)" in prompt
    assert "## RECENT_TURNS\n(brak)" in prompt
    assert "## RETRIEVED_CONTEXT\n(brak)" in prompt


def test_system_rules_include_instruction_priority() -> None:
    rules = default_system_rules()
    assert "SYSTEM > developer/app rules > USER > retrieved documents" in rules
    assert "Domyślnie odpowiadaj po polsku" in rules
    assert "Nie wiem na podstawie dostarczonych materiałów." in rules
    assert "W materiałach nie znalazłem informacji o: <temat>" in rules
