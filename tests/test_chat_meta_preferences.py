from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from types import MethodType

from app.chat.service import ChatService, RetrievalBundle
from app.persistence.repositories import Persistence
from app.types import RetrievedChunk


@dataclass
class FakeConfig:
    similarity_threshold: float = 0.35
    debug_retrieval: bool = False
    retrieval_debug_top_n: int = 15
    n_turns: int = 8
    keep_last_turns: int = 6
    summary_trigger_tokens: int = 3200
    summary_trigger_turns: int = 14
    max_prompt_tokens: int = 6000
    model_name: str = "test-model"
    temperature: float = 0.2
    top_p: float = 0.9
    embed_model: str = "test-embed"
    top_k: int = 5

    def fingerprint(self) -> dict[str, object]:
        return {"llm": {"model": self.model_name}, "rag": {"top_k": self.top_k}}


class FakeOllama:
    def __init__(self, generated: str) -> None:
        self.generated = generated
        self.generate_calls = 0
        self.embed_calls = 0

    def generate(self, **_: object) -> str:
        self.generate_calls += 1
        return self.generated

    def embed_texts(self, *, model: str, texts: list[str]) -> list[list[float]]:
        _ = model, texts
        self.embed_calls += 1
        return [[0.1, 0.2, 0.3]]


class FakeSummarizer:
    def should_summarize(self, **_: object) -> bool:
        return False


def _build_service(
    tmp_path: Path,
    *,
    generated: str,
) -> tuple[ChatService, Persistence, FakeOllama]:
    service = ChatService.__new__(ChatService)
    service.config = FakeConfig()
    service.persistence = Persistence(db_path=tmp_path / "chat.db")
    service.ollama = FakeOllama(generated)
    service.summarizer = FakeSummarizer()
    service.retriever = object()
    return service, service.persistence, service.ollama


def test_meta_command_sets_preferences_without_retrieval_or_llm(tmp_path: Path) -> None:
    service, persistence, fake_ollama = _build_service(
        tmp_path,
        generated="To nie powinno się wykonać.",
    )
    thread_id = "thread-meta-set"

    reply = service.respond(
        thread_id=thread_id,
        user_text="Od teraz odpowiadaj krótko (maks 2 zdania). Potwierdź.",
    )

    prefs = persistence.thread_state.get_preferences(thread_id)
    assert "Ustawienia zapisane." in reply.content
    assert reply.sources == []
    assert fake_ollama.generate_calls == 0
    assert fake_ollama.embed_calls == 0
    assert prefs["max_sentences"] == 2
    assert reply.config_fingerprint["retrieval_summary"]["intent"] == "META"
    assert reply.config_fingerprint["retrieval_summary"]["has_context"] is False


def test_meta_query_reads_preferences_without_retrieval_or_llm(tmp_path: Path) -> None:
    service, persistence, fake_ollama = _build_service(
        tmp_path,
        generated="To nie powinno się wykonać.",
    )
    thread_id = "thread-meta-query"
    persistence.threads.upsert_thread(thread_id)
    persistence.thread_state.upsert_preferences(
        thread_id=thread_id,
        preferences={"max_sentences": 2, "ask_check_question": True, "style_short": True},
    )

    reply = service.respond(
        thread_id=thread_id,
        user_text="Jaką długość odpowiedzi mam ustawioną?",
    )

    assert "Aktualne preferencje:" in reply.content
    assert "2 zdania" in reply.content
    assert reply.sources == []
    assert fake_ollama.generate_calls == 0
    assert fake_ollama.embed_calls == 0
    assert reply.config_fingerprint["retrieval_summary"]["intent"] == "META"


def test_content_response_respects_max_sentences_and_keeps_citation(tmp_path: Path) -> None:
    service, persistence, _ = _build_service(
        tmp_path,
        generated=(
            "Mieszko I był księciem Polan. Zjednoczył część ziem. "
            "Przyjął chrzest w 966 roku. Czy to jest dla Ciebie jasne?"
        ),
    )
    thread_id = "thread-content-preferences"
    persistence.threads.upsert_thread(thread_id)
    persistence.thread_state.upsert_preferences(
        thread_id=thread_id,
        preferences={"max_sentences": 2, "ask_check_question": True, "style_short": True},
    )

    def fake_retrieve(
        self: ChatService,
        *,
        user_text: str,
        request_timeout_seconds: float | None = None,
    ) -> RetrievalBundle:
        _ = user_text, request_timeout_seconds
        return RetrievalBundle(
            final_chunks=[
                RetrievedChunk(
                    chunk_id="c-mieszko",
                    source_file="historia.pdf",
                    page=21,
                    text="Mieszko I był księciem Polan i przyjął chrzest w 966 roku.",
                    score=0.9,
                )
            ],
            candidates=[],
            keywords=["mieszko"],
            main_keyword="mieszko",
            phrase_norm=None,
            query_evidence=True,
            lexical_fallback_used=False,
        )

    service._retrieve = MethodType(fake_retrieve, service)

    reply = service.respond(thread_id=thread_id, user_text="Kim był Mieszko I?")

    assert len(reply.sources) == 1
    assert "[Źródło:" in reply.content

    body = reply.content.split("\n\n[Źródło:", 1)[0]
    parts = [part for part in body.split("\n\n") if part.strip()]
    answer_body = parts[0] if parts else body
    sentence_count = len([s for s in re.split(r"(?<=[.!?])\s+", answer_body) if s.strip()])
    assert sentence_count <= 2
