from __future__ import annotations

import re
from dataclasses import dataclass
from types import MethodType, SimpleNamespace

from app.chat.service import ChatService, RetrievalBundle, _build_extractive_fallback
from app.persistence.repositories import ThreadStateRecord
from app.types import ChatMessage, RetrievedChunk, SourceCitation, utc_now_iso


@dataclass
class FakeConfig:
    similarity_threshold: float = 0.35
    debug_retrieval: bool = False
    retrieval_debug_top_n: int = 30
    n_turns: int = 8
    keep_last_turns: int = 6
    summary_trigger_tokens: int = 3200
    summary_trigger_turns: int = 14
    max_prompt_tokens: int = 6000
    model_name: str = "test-model"
    temperature: float = 0.2
    top_p: float = 0.9
    embed_model: str = "test-embed"
    top_k: int = 3

    def fingerprint(self) -> dict[str, object]:
        return {"llm": {"model": self.model_name}}


class FakeThreadsRepo:
    def upsert_thread(self, thread_id: str, title: str | None = None) -> None:
        _ = thread_id, title

    def list_threads(self) -> list[dict[str, str | None]]:
        return []


class FakeMessagesRepo:
    def __init__(self) -> None:
        self.by_thread: dict[str, list[ChatMessage]] = {}

    def add_user_message(self, thread_id: str, content: str) -> ChatMessage:
        message = ChatMessage(
            id=f"u-{len(self.by_thread.get(thread_id, []))}",
            thread_id=thread_id,
            role="user",
            content=content,
        )
        self.by_thread.setdefault(thread_id, []).append(message)
        return message

    def add_assistant_message(
        self,
        *,
        thread_id: str,
        content: str,
        model: str,
        token_count: int,
        sources: list[SourceCitation],
        config_fingerprint: dict,
    ) -> ChatMessage:
        message = ChatMessage(
            id=f"a-{len(self.by_thread.get(thread_id, []))}",
            thread_id=thread_id,
            role="assistant",
            content=content,
            model=model,
            token_count=token_count,
            sources=sources,
            config_fingerprint=config_fingerprint,
        )
        self.by_thread.setdefault(thread_id, []).append(message)
        return message

    def list_messages(self, thread_id: str) -> list[ChatMessage]:
        return list(self.by_thread.get(thread_id, []))


class FakeThreadStateRepo:
    def get_state(self, thread_id: str) -> ThreadStateRecord:
        return ThreadStateRecord(
            thread_id=thread_id,
            summary=None,
            memory_version=0,
            updated_at=utc_now_iso(),
        )

    def upsert_state(self, *, thread_id: str, summary: str | None, memory_version: int) -> None:
        _ = thread_id, summary, memory_version


class FakeSummarizer:
    def should_summarize(self, **_: object) -> bool:
        return False


class FakeOllama:
    def __init__(self, generated: str) -> None:
        self.generated = generated
        self.prompts: list[str] = []

    def generate(self, **kwargs: object) -> str:
        prompt = kwargs.get("prompt")
        if isinstance(prompt, str):
            self.prompts.append(prompt)
        return self.generated


class FakeRetrieverWithHistoryChunk:
    def __init__(self, history_chunk: RetrievedChunk) -> None:
        self.history_chunk = history_chunk
        self.calls = 0

    def retrieve_by_chunk_ids(self, *, chunk_ids: list[str]) -> list[RetrievedChunk]:
        self.calls += 1
        if self.history_chunk.chunk_id in chunk_ids:
            return [self.history_chunk]
        return []


def _build_service(
    *,
    generated: str,
    retrieval_bundle: RetrievalBundle,
) -> tuple[ChatService, FakeMessagesRepo, FakeOllama]:
    messages_repo = FakeMessagesRepo()
    fake_ollama = FakeOllama(generated)
    service = ChatService.__new__(ChatService)
    service.config = FakeConfig()
    service.persistence = SimpleNamespace(
        threads=FakeThreadsRepo(),
        messages=messages_repo,
        thread_state=FakeThreadStateRepo(),
    )
    service.ollama = fake_ollama
    service.summarizer = FakeSummarizer()
    service.retriever = object()

    def fake_retrieve(
        self: ChatService,
        *,
        user_text: str,
        request_timeout_seconds: float | None = None,
    ) -> RetrievalBundle:
        _ = user_text, request_timeout_seconds
        return retrieval_bundle

    service._retrieve = MethodType(fake_retrieve, service)
    return service, messages_repo, fake_ollama


def test_proper_noun_guard_replaces_hallucination_with_extractive_fallback() -> None:
    evidence = RetrievedChunk(
        chunk_id="c-evidence",
        source_file="historia.pdf",
        page=21,
        text="Mieszko I był księciem Polan i przyjął chrzest w 966 roku.",
        score=0.71,
    )
    retrieval_bundle = RetrievalBundle(
        final_chunks=[
            evidence,
            RetrievedChunk(
                chunk_id="c-noise-1",
                source_file="historia.pdf",
                page=14,
                text="To nie jest fragment o Mieszku.",
                score=0.95,
            ),
            RetrievedChunk(
                chunk_id="c-noise-2",
                source_file="historia.pdf",
                page=116,
                text="Inny temat historyczny bez postaci z pytania.",
                score=0.89,
            ),
        ],
        candidates=[],
        keywords=["mieszko"],
        main_keyword="mieszko",
        phrase_norm=None,
        query_evidence=True,
        lexical_fallback_used=False,
    )
    service, _, _ = _build_service(
        generated="Mieszko I był synem Piasta Kołodzieja.",
        retrieval_bundle=retrieval_bundle,
    )

    reply = service.respond(thread_id="thread-1", user_text="Kim był Mieszko I?")

    assert "Piast" not in reply.content
    assert "Kołodziej" not in reply.content
    assert "Na podstawie dostarczonego materiału mogę powiedzieć:" in reply.content
    assert len(reply.sources) == 1
    assert reply.sources[0].chunk_id == "c-evidence"


def test_prompt_and_citations_use_only_hit_context_chunks() -> None:
    evidence = RetrievedChunk(
        chunk_id="c-evidence",
        source_file="historia.pdf",
        page=21,
        text="Mieszko I był księciem Polan i przyjął chrzest w 966 roku.",
        score=0.70,
    )
    noise_1 = RetrievedChunk(
        chunk_id="c-noise-1",
        source_file="historia.pdf",
        page=14,
        text="Piast Kołodziej to inna opowieść spoza tego pytania.",
        score=0.98,
    )
    noise_2 = RetrievedChunk(
        chunk_id="c-noise-2",
        source_file="historia.pdf",
        page=116,
        text="W tym fragmencie nie ma informacji o Mieszku.",
        score=0.97,
    )
    retrieval_bundle = RetrievalBundle(
        final_chunks=[evidence, noise_1, noise_2],
        candidates=[],
        keywords=["mieszko"],
        main_keyword="mieszko",
        phrase_norm=None,
        query_evidence=True,
        lexical_fallback_used=False,
    )
    service, _, fake_ollama = _build_service(
        generated="Mieszko I był księciem Polan.",
        retrieval_bundle=retrieval_bundle,
    )

    reply = service.respond(thread_id="thread-2", user_text="Kim był Mieszko I?")

    assert fake_ollama.prompts
    prompt = fake_ollama.prompts[0]
    assert evidence.text in prompt
    assert noise_1.text not in prompt
    assert noise_2.text not in prompt
    assert len(reply.sources) == 1
    assert reply.sources[0].chunk_id == "c-evidence"
    assert reply.content.count("[Źródło:") == 1


def test_phrase_norm_without_phrase_hit_falls_back_to_main_keyword_context() -> None:
    keyword_only_chunk = RetrievedChunk(
        chunk_id="c-boleslaw",
        source_file="historia.pdf",
        page=22,
        text="Bolesław umacniał pozycję państwa i prowadził wyprawy zbrojne.",
        score=0.72,
    )
    retrieval_bundle = RetrievalBundle(
        final_chunks=[
            keyword_only_chunk,
            RetrievedChunk(
                chunk_id="c-noise",
                source_file="historia.pdf",
                page=88,
                text="Fragment o zupełnie innym temacie bez tej postaci.",
                score=0.91,
            ),
        ],
        candidates=[],
        keywords=["boleslaw", "chrobry"],
        main_keyword="boleslaw",
        phrase_norm="boleslaw chrobry",
        query_evidence=True,
        lexical_fallback_used=False,
    )
    service, _, fake_ollama = _build_service(
        generated="Bolesław umacniał pozycję państwa.",
        retrieval_bundle=retrieval_bundle,
    )

    reply = service.respond(thread_id="thread-3", user_text="Kim był Bolesław Chrobry?")

    assert fake_ollama.prompts
    prompt = fake_ollama.prompts[0]
    assert keyword_only_chunk.text in prompt
    assert "zupełnie innym temacie" not in prompt
    assert len(reply.sources) == 1
    assert reply.sources[0].chunk_id == "c-boleslaw"


def test_extractive_fallback_picks_main_keyword_evidence_chunk() -> None:
    noise_chunk = RetrievedChunk(
        chunk_id="c-noise-first",
        source_file="historia.pdf",
        page=50,
        text="Rozważania ogólne bez nazwy postaci z pytania.",
        score=0.95,
    )
    evidence_chunk = RetrievedChunk(
        chunk_id="c-mieszko-evidence",
        source_file="historia.pdf",
        page=21,
        text="Mieszko I przyjął chrzest i wzmocnił pozycję państwa.",
        score=0.70,
    )

    content, citations = _build_extractive_fallback(
        context_chunks=[noise_chunk, evidence_chunk],
        phrase_norm="mieszko ksiaze",
        main_keyword="mieszko",
        keywords=["mieszko", "ksiaze"],
    )

    assert len(citations) == 1
    assert citations[0].chunk_id == "c-mieszko-evidence"
    assert "Mieszko" in content


def test_extractive_fallback_uses_clean_sentence_not_leading_page_number() -> None:
    evidence_chunk = RetrievedChunk(
        chunk_id="c-wojciech-evidence",
        source_file="historia.pdf",
        page=23,
        text=(
            "21 Święty Wojciech (ok. 956-997) był biskupem praskim i misjonarzem. "
            "Zginął śmiercią męczeńską podczas misji."
        ),
        score=0.92,
    )

    content, citations = _build_extractive_fallback(
        context_chunks=[evidence_chunk],
        phrase_norm="swiety wojciech",
        main_keyword="wojciech",
        keywords=["swiety", "wojciech"],
    )

    first_line = content.splitlines()[0]
    assert not first_line.startswith("Na podstawie dostarczonego materiału mogę powiedzieć: 21 ")
    assert "Święty Wojciech" in content
    assert len(citations) == 1
    assert citations[0].chunk_id == "c-wojciech-evidence"


def test_followup_summary_keeps_context_for_wojciech_topic() -> None:
    evidence = RetrievedChunk(
        chunk_id="c-wojciech",
        source_file="historia.pdf",
        page=23,
        text=(
            "Święty Wojciech był biskupem praskim i misjonarzem. "
            "Zginął śmiercią męczeńską podczas misji."
        ),
        score=1.0,
    )
    retrieval_bundle = RetrievalBundle(
        final_chunks=[evidence],
        candidates=[],
        keywords=["wrocmy", "podsumuj", "informacje"],
        main_keyword="informacje",
        phrase_norm=None,
        query_evidence=False,
        lexical_fallback_used=False,
    )
    service, messages_repo, _ = _build_service(
        generated=(
            "Święty Wojciech był biskupem i misjonarzem. "
            "Zginął śmiercią męczeńską. Czy to jest dla Ciebie jasne?"
        ),
        retrieval_bundle=retrieval_bundle,
    )
    thread_id = "thread-followup"
    messages_repo.by_thread[thread_id] = [
        ChatMessage(
            id="u-prev",
            thread_id=thread_id,
            role="user",
            content="Kto to był Święty Wojciech?",
        ),
        ChatMessage(
            id="a-prev",
            thread_id=thread_id,
            role="assistant",
            content="Święty Wojciech był biskupem praskim.",
            sources=[
                SourceCitation(
                    source_file="historia.pdf",
                    page=23,
                    chunk_id="c-wojciech",
                    score=0.91,
                )
            ],
        ),
    ]

    reply = service.respond(
        thread_id=thread_id,
        user_text="Wróćmy do tego, co mówiłeś o Wojciechu: podsumuj w 2 zdaniach.",
    )

    assert "Nie wiem na podstawie dostarczonych materiałów." not in reply.content
    assert len(reply.sources) == 1
    assert reply.sources[0].chunk_id == "c-wojciech"
    assert reply.config_fingerprint["retrieval_summary"]["has_context"] is True


def test_followup_summary_prefers_history_cited_chunk_over_irrelevant_top_chunk() -> None:
    irrelevant = RetrievedChunk(
        chunk_id="c-irrelevant",
        source_file="historia.pdf",
        page=77,
        text="Unia polsko-litewska została zawarta po długich negocjacjach.",
        score=1.0,
    )
    wojciech = RetrievedChunk(
        chunk_id="c-wojciech",
        source_file="historia.pdf",
        page=23,
        text="Święty Wojciech był biskupem praskim i misjonarzem.",
        score=0.74,
    )
    retrieval_bundle = RetrievalBundle(
        final_chunks=[irrelevant, wojciech],
        candidates=[],
        keywords=["wrocmy", "podsumuj", "informacje"],
        main_keyword="informacje",
        phrase_norm=None,
        query_evidence=False,
        lexical_fallback_used=False,
    )
    service, messages_repo, _ = _build_service(
        generated="Święty Wojciech żył około 1999 roku.",
        retrieval_bundle=retrieval_bundle,
    )
    thread_id = "thread-followup-history"
    messages_repo.by_thread[thread_id] = [
        ChatMessage(
            id="u-prev",
            thread_id=thread_id,
            role="user",
            content="Kto to był Święty Wojciech?",
        ),
        ChatMessage(
            id="a-prev",
            thread_id=thread_id,
            role="assistant",
            content="Święty Wojciech był biskupem praskim.",
            sources=[
                SourceCitation(
                    source_file="historia.pdf",
                    page=23,
                    chunk_id="c-wojciech",
                    score=0.91,
                )
            ],
        ),
    ]

    reply = service.respond(
        thread_id=thread_id,
        user_text="Wróćmy do tego i podsumuj w 2 zdaniach najważniejsze informacje.",
    )

    assert "Nie wiem na podstawie dostarczonych materiałów." not in reply.content
    assert len(reply.sources) == 1
    assert reply.sources[0].chunk_id == "c-wojciech"


def test_followup_summary_fetches_and_prefers_history_chunk_when_topk_is_irrelevant() -> None:
    retrieval_bundle = RetrievalBundle(
        final_chunks=[
            RetrievedChunk(
                chunk_id="c-irrelevant-top",
                source_file="historia.pdf",
                page=115,
                text="Jan Paweł II odwiedził Polskę wielokrotnie.",
                score=1.0,
            )
        ],
        candidates=[],
        keywords=["wojciech"],
        main_keyword="wojciech",
        phrase_norm=None,
        query_evidence=False,
        lexical_fallback_used=False,
    )
    service, messages_repo, _ = _build_service(
        generated="Nie wiem na podstawie dostarczonych materiałów.",
        retrieval_bundle=retrieval_bundle,
    )

    history_chunk = RetrievedChunk(
        chunk_id="c-history-wojciech",
        source_file="historia.pdf",
        page=23,
        text="Święty Wojciech był biskupem praskim i misjonarzem.",
        score=1.0,
    )
    fake_retriever = FakeRetrieverWithHistoryChunk(history_chunk=history_chunk)
    service.retriever = fake_retriever

    thread_id = "thread-followup-priority"
    messages_repo.by_thread[thread_id] = [
        ChatMessage(
            id="u-prev",
            thread_id=thread_id,
            role="user",
            content="Kto to był Święty Wojciech?",
        ),
        ChatMessage(
            id="a-prev",
            thread_id=thread_id,
            role="assistant",
            content="Święty Wojciech był biskupem praskim.",
            sources=[
                SourceCitation(
                    source_file="historia.pdf",
                    page=23,
                    chunk_id="c-history-wojciech",
                    score=0.91,
                )
            ],
        ),
    ]

    reply = service.respond(
        thread_id=thread_id,
        user_text="Wróćmy do tego, co mówiłeś o Wojciechu: podsumuj w 2 zdaniach.",
    )

    assert fake_retriever.calls >= 1
    assert "Nie wiem na podstawie dostarczonych materiałów." not in reply.content
    assert len(reply.sources) == 1
    assert reply.sources[0].chunk_id == "c-history-wojciech"


def test_followup_summary_without_history_returns_clarifying_no_context() -> None:
    retrieval_bundle = RetrievalBundle(
        final_chunks=[
            RetrievedChunk(
                chunk_id="c-kossak",
                source_file="historia.pdf",
                page=81,
                text="Wojciech Kossak był malarzem batalistą.",
                score=1.0,
            )
        ],
        candidates=[],
        keywords=["wojciech"],
        main_keyword="wojciech",
        phrase_norm=None,
        query_evidence=True,
        lexical_fallback_used=False,
    )
    service, _, fake_ollama = _build_service(
        generated="Wojciech Kossak był malarzem batalistą.",
        retrieval_bundle=retrieval_bundle,
    )

    reply = service.respond(
        thread_id="thread-empty-history",
        user_text="Podsumuj w 2 zdaniach, co mówiłeś o Wojciechu.",
    )

    assert "Nie wiem na podstawie dostarczonych materiałów." in reply.content
    assert "o którego" in reply.content
    assert len(reply.sources) == 0
    assert reply.config_fingerprint["retrieval_summary"]["has_context"] is False
    assert fake_ollama.prompts == []


def test_topic_summary_with_entity_keeps_context_even_if_query_evidence_false() -> None:
    evidence = RetrievedChunk(
        chunk_id="c-wojciech-topic",
        source_file="historia.pdf",
        page=23,
        text=(
            "Święty Wojciech był biskupem praskim i misjonarzem. "
            "Zginął śmiercią męczeńską podczas misji."
        ),
        score=1.0,
    )
    retrieval_bundle = RetrievalBundle(
        final_chunks=[evidence],
        candidates=[],
        keywords=["podsumuj", "swietego", "wojciecha"],
        main_keyword="podsumuj",
        phrase_norm=None,
        query_evidence=False,
        lexical_fallback_used=False,
    )
    service, _, _ = _build_service(
        generated=(
            "Święty Wojciech był biskupem i misjonarzem. "
            "Zginął śmiercią męczeńską. Czy to wyjaśnienie jest dla Ciebie zrozumiałe?"
        ),
        retrieval_bundle=retrieval_bundle,
    )
    service.persistence.thread_state.get_preferences = (
        lambda _thread_id: {
            "max_sentences": 2,
            "ask_check_question": False,
            "answer_style": "short",
            "style_short": True,
        }
    )
    service.persistence.thread_state.upsert_preferences = lambda **_: None

    reply = service.respond(
        thread_id="thread-topic-summary",
        user_text="Podsumuj w 2 zdaniach Świętego Wojciecha.",
    )

    assert "Nie wiem na podstawie dostarczonych materiałów." not in reply.content
    assert len(reply.sources) >= 1
    assert reply.config_fingerprint["retrieval_summary"]["has_context"] is True
    assert "Sprawdź się:" not in reply.content
    assert "Czy to wyjaśnienie jest dla Ciebie zrozumiałe?" not in reply.content

    body = reply.content.split("\n\n[Źródło:", 1)[0]
    sentence_count = len([part for part in re.split(r"(?<=[.!?])\s+", body) if part.strip()])
    assert sentence_count <= 2
