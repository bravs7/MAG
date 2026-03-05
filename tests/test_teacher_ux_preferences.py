from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from types import MethodType

from app.chat.service import ChatService, RetrievalBundle
from app.persistence.repositories import Persistence
from app.retrieval.hybrid import normalize_for_match
from app.types import RetrievedChunk, SourceCitation


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

    def generate(self, **_: object) -> str:
        return self.generated

    def embed_texts(self, *, model: str, texts: list[str]) -> list[list[float]]:
        _ = model, texts
        return [[0.1, 0.2, 0.3]]


class FakeSummarizer:
    def should_summarize(self, **_: object) -> bool:
        return False


class FakeHistoryRetriever:
    def __init__(self, history_chunk: RetrievedChunk) -> None:
        self.history_chunk = history_chunk
        self.calls = 0

    def retrieve_by_chunk_ids(self, *, chunk_ids: list[str]) -> list[RetrievedChunk]:
        self.calls += 1
        if self.history_chunk.chunk_id in chunk_ids:
            return [self.history_chunk]
        return []


def _build_service(tmp_path: Path, *, generated: str) -> tuple[ChatService, Persistence]:
    service = ChatService.__new__(ChatService)
    service.config = FakeConfig()
    service.persistence = Persistence(db_path=tmp_path / "chat.db")
    service.ollama = FakeOllama(generated)
    service.summarizer = FakeSummarizer()
    service.retriever = object()
    return service, service.persistence


def _bind_retrieve(service: ChatService, bundle: RetrievalBundle) -> None:
    def fake_retrieve(
        self: ChatService,
        *,
        user_text: str,
        request_timeout_seconds: float | None = None,
    ) -> RetrievalBundle:
        _ = user_text, request_timeout_seconds
        return bundle

    service._retrieve = MethodType(fake_retrieve, service)


def _seed_history_with_sources(
    *,
    persistence: Persistence,
    thread_id: str,
    chunks: list[RetrievedChunk],
) -> None:
    persistence.messages.add_user_message(thread_id, "Kto to był Święty Wojciech?")
    sources = [
        SourceCitation(
            source_file=chunk.source_file,
            page=chunk.page,
            chunk_id=chunk.chunk_id,
            score=chunk.score,
        )
        for chunk in chunks
    ]
    persistence.messages.add_assistant_message(
        thread_id=thread_id,
        content="Święty Wojciech był biskupem praskim.",
        model="test-model",
        token_count=24,
        sources=sources,
        config_fingerprint={},
    )


def _body_without_citations(content: str) -> str:
    return re.split(r"\n\[(?:Źródło|Zrodlo):", content, maxsplit=1)[0]


def _sentence_count(content: str) -> int:
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", content) if part.strip()]
    return len(sentences)


def test_preferences_remove_check_question_and_keep_citations(tmp_path: Path) -> None:
    chunk = RetrievedChunk(
        chunk_id="c-mieszko",
        source_file="historia.pdf",
        page=21,
        text="Mieszko I był księciem Polan i przyjął chrzest w 966 roku.",
        score=0.9,
    )
    bundle = RetrievalBundle(
        final_chunks=[chunk],
        candidates=[],
        keywords=["mieszko"],
        main_keyword="mieszko",
        phrase_norm=None,
        query_evidence=True,
        lexical_fallback_used=False,
    )
    service, persistence = _build_service(
        tmp_path,
        generated=(
            "Mieszko I był księciem Polan. Umocnił państwo. "
            "Przyjął chrzest w 966 roku. Czy to wyjaśnienie jest dla Ciebie zrozumiałe?"
        ),
    )
    _bind_retrieve(service, bundle)

    thread_id = "thread-prefs-generated"
    persistence.threads.upsert_thread(thread_id)
    persistence.thread_state.upsert_preferences(
        thread_id=thread_id,
        preferences={"max_sentences": 2, "ask_check_question": False, "style_short": True},
    )

    reply = service.respond(thread_id=thread_id, user_text="Kim był Mieszko I?")

    normalized_reply = normalize_for_match(reply.content)
    assert "czy to wyjasnienie jest dla ciebie zrozumiale" not in normalized_reply
    assert "czy zrozumial" not in normalized_reply
    assert len(reply.sources) == 1
    assert "[Źródło:" in reply.content
    assert _sentence_count(_body_without_citations(reply.content)) <= 2


def test_preferences_strip_check_question_in_extractive_fallback(tmp_path: Path) -> None:
    chunk = RetrievedChunk(
        chunk_id="c-fallback-mieszko",
        source_file="historia.pdf",
        page=21,
        text="Mieszko I przyjął chrzest i wzmocnił pozycję państwa.",
        score=0.84,
    )
    bundle = RetrievalBundle(
        final_chunks=[chunk],
        candidates=[],
        keywords=["mieszko"],
        main_keyword="mieszko",
        phrase_norm=None,
        query_evidence=True,
        lexical_fallback_used=False,
    )
    service, persistence = _build_service(
        tmp_path,
        generated="Nie wiem na podstawie dostarczonych materiałów.",
    )
    _bind_retrieve(service, bundle)

    thread_id = "thread-prefs-fallback"
    persistence.threads.upsert_thread(thread_id)
    persistence.thread_state.upsert_preferences(
        thread_id=thread_id,
        preferences={"max_sentences": 2, "ask_check_question": False, "style_short": True},
    )

    reply = service.respond(thread_id=thread_id, user_text="Kim był Mieszko I?")

    normalized_reply = normalize_for_match(reply.content)
    assert "czy to wyjasnienie jest dla ciebie zrozumiale" not in normalized_reply
    assert len(reply.sources) == 1
    assert reply.sources[0].chunk_id == "c-fallback-mieszko"
    assert "[Źródło:" in reply.content


def test_followup_entity_density_prefers_chunk_about_wojciech(tmp_path: Path) -> None:
    chunk_a = RetrievedChunk(
        chunk_id="c-wojciech-main",
        source_file="historia.pdf",
        page=23,
        text=(
            "Święty Wojciech był biskupem i misjonarzem. "
            "Wojciech prowadził misję wśród Prusów. "
            "Postać Wojciecha jest kluczowa dla tematu."
        ),
        score=0.78,
    )
    chunk_b = RetrievedChunk(
        chunk_id="c-zjazd-with-mention",
        source_file="historia.pdf",
        page=24,
        text=(
            "Zjazd gnieźnieński odbył się w 1000 roku. "
            "Cesarz odwiedził grób św. Wojciecha."
        ),
        score=0.96,
    )
    bundle = RetrievalBundle(
        final_chunks=[chunk_b, chunk_a],
        candidates=[],
        keywords=["wojciech"],
        main_keyword="wojciech",
        phrase_norm=None,
        query_evidence=True,
        lexical_fallback_used=False,
    )
    service, persistence = _build_service(
        tmp_path,
        generated="Święty Wojciech był biskupem i misjonarzem.",
    )
    _bind_retrieve(service, bundle)

    thread_id = "thread-followup-density"
    persistence.threads.upsert_thread(thread_id)
    _seed_history_with_sources(
        persistence=persistence,
        thread_id=thread_id,
        chunks=[chunk_b, chunk_a],
    )

    reply = service.respond(
        thread_id=thread_id,
        user_text="Wróćmy do Wojciecha: podsumuj najważniejsze informacje.",
    )

    assert reply.sources
    assert reply.sources[0].chunk_id == "c-wojciech-main"


def test_followup_prefers_history_chunk_when_topk_has_only_related_mention(tmp_path: Path) -> None:
    chunk_a = RetrievedChunk(
        chunk_id="c-wojciech-history",
        source_file="historia.pdf",
        page=23,
        text=(
            "Święty Wojciech był biskupem i misjonarzem. "
            "Wojciech zginął śmiercią męczeńską."
        ),
        score=1.0,
    )
    chunk_b = RetrievedChunk(
        chunk_id="c-zjazd-topk",
        source_file="historia.pdf",
        page=24,
        text=(
            "Zjazd gnieźnieński odbył się w 1000 roku. "
            "Wspomniano wtedy grób św. Wojciecha."
        ),
        score=0.98,
    )
    bundle = RetrievalBundle(
        final_chunks=[chunk_b],
        candidates=[],
        keywords=["wojciech"],
        main_keyword="wojciech",
        phrase_norm=None,
        query_evidence=True,
        lexical_fallback_used=False,
    )
    service, persistence = _build_service(
        tmp_path,
        generated="Nie wiem na podstawie dostarczonych materiałów.",
    )
    _bind_retrieve(service, bundle)

    fake_retriever = FakeHistoryRetriever(history_chunk=chunk_a)
    service.retriever = fake_retriever

    thread_id = "thread-followup-history-priority"
    persistence.threads.upsert_thread(thread_id)
    _seed_history_with_sources(
        persistence=persistence,
        thread_id=thread_id,
        chunks=[chunk_a],
    )

    reply = service.respond(
        thread_id=thread_id,
        user_text="Wróćmy do tego, co mówiłeś o Wojciechu: podsumuj w 2 zdaniach.",
    )

    assert fake_retriever.calls >= 1
    assert "Nie wiem na podstawie dostarczonych materiałów." not in reply.content
    assert reply.sources
    assert reply.sources[0].chunk_id == "c-wojciech-history"
