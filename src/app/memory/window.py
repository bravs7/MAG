"""Short-term memory window helpers."""

from __future__ import annotations

from app.types import ChatMessage


def estimate_tokens(text: str) -> int:
    # Lightweight approximation to avoid external tokenizer dependency.
    if not text:
        return 0
    return max(1, len(text) // 4)


def estimate_messages_tokens(messages: list[ChatMessage]) -> int:
    return sum(estimate_tokens(msg.content) for msg in messages)


def take_recent_turn_messages(messages: list[ChatMessage], n_turns: int) -> list[ChatMessage]:
    if n_turns <= 0:
        return []
    # Approximation: one turn ~= user+assistant pair.
    max_messages = n_turns * 2
    return list(messages[-max_messages:])


def trim_to_token_budget(
    *, summary: str | None, recent_messages: list[ChatMessage], max_prompt_tokens: int
) -> tuple[str | None, list[ChatMessage]]:
    summary_tokens = estimate_tokens(summary or "")
    if summary_tokens >= max_prompt_tokens:
        return summary, []

    budget_left = max_prompt_tokens - summary_tokens
    kept: list[ChatMessage] = []
    running = 0

    for msg in reversed(recent_messages):
        msg_tokens = estimate_tokens(msg.content)
        if running + msg_tokens > budget_left:
            break
        kept.append(msg)
        running += msg_tokens

    kept.reverse()
    return summary, kept
