from __future__ import annotations

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
