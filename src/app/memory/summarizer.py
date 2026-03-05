"""Long-term thread memory summarization policy."""

from __future__ import annotations

from dataclasses import dataclass

from app.memory.window import estimate_messages_tokens
from app.runtime.ollama_client import OllamaClient
from app.types import ChatMessage


@dataclass(slots=True)
class SummaryPolicy:
    summary_trigger_tokens: int
    summary_trigger_turns: int
    keep_last_turns: int


class ThreadSummarizer:
    def __init__(self, ollama: OllamaClient, *, model_name: str) -> None:
        self._ollama = ollama
        self._model_name = model_name

    def should_summarize(
        self,
        *,
        summary: str | None,
        messages: list[ChatMessage],
        policy: SummaryPolicy,
    ) -> bool:
        if not messages:
            return False

        total_tokens = estimate_messages_tokens(messages) + len(summary or "") // 4
        if total_tokens >= policy.summary_trigger_tokens:
            return True

        # Approximation: one turn is usually user+assistant pair.
        turns = len(messages) // 2
        return turns >= policy.summary_trigger_turns

    def split_for_summary(
        self, messages: list[ChatMessage], keep_last_turns: int
    ) -> tuple[list[ChatMessage], list[ChatMessage]]:
        keep = max(0, keep_last_turns * 2)
        if keep <= 0:
            return messages, []
        if len(messages) <= keep:
            return [], list(messages)
        return list(messages[:-keep]), list(messages[-keep:])

    def build_or_update_summary(
        self,
        *,
        existing_summary: str | None,
        older_messages: list[ChatMessage],
    ) -> str:
        if not older_messages:
            return existing_summary or ""

        conversation = "\n".join(
            f"{message.role.upper()}: {message.content}" for message in older_messages
        )

        prompt = (
            "Zaktualizuj streszczenie rozmowy ucznia z nauczycielem. "
            "Nie wymyslaj faktow ani zrodel. Uwzglednij tylko to, co padlo.\n\n"
            "Wymagane sekcje (jesli wystepuja):\n"
            "- cele ucznia i temat\n"
            "- ustalone definicje i zalozenia\n"
            "- co uczen rozumie vs co myli\n"
            "- bledy i poprawki\n"
            "- zadania i postep\n"
            "- preferencje ucznia\n\n"
            f"Poprzednie streszczenie:\n{existing_summary or '(brak)'}\n\n"
            f"Starsza historia:\n{conversation}\n"
        )

        summary = self._ollama.generate(
            model=self._model_name,
            prompt=prompt,
            temperature=0.1,
            top_p=0.9,
        )
        return summary.strip()
