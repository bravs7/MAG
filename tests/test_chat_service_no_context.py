from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from app.chat.service import ChatService
from app.persistence.repositories import ThreadStateRecord
from app.types import ChatMessage, RetrievedChunk, SourceCitation, utc_now_iso


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
        return {"llm": {"model": self.model_name}}


class FakeThreadsRepo:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, str | None]] = {}

    def upsert_thread(self, thread_id: str, title: str | None = None) -> None:
        self.rows[thread_id] = {
            "id": thread_id,
            "title": title,
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
        }

    def list_threads(self) -> list[dict[str, str | None]]:
        return list(self.rows.values())


class FakeMessagesRepo:
    def __init__(self) -> None:
        self.by_thread: dict[str, list[ChatMessage]] = {}

    def add_user_message(self, thread_id: str, content: str) -> ChatMessage:
        msg = ChatMessage(
            id=f"u-{len(self.by_thread.get(thread_id, []))}",
            thread_id=thread_id,
            role="user",
            content=content,
        )
        self.by_thread.setdefault(thread_id, []).append(msg)
        return msg

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
        msg = ChatMessage(
            id=f"a-{len(self.by_thread.get(thread_id, []))}",
            thread_id=thread_id,
            role="assistant",
            content=content,
            model=model,
            token_count=token_count,
            sources=sources,
            config_fingerprint=config_fingerprint,
        )
        self.by_thread.setdefault(thread_id, []).append(msg)
        return msg

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
        return None


class FakeOllama:
    def __init__(self) -> None:
        self.generate_calls = 0

    def embed_texts(self, *, model: str, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    def generate(self, **_: object) -> str:
        self.generate_calls += 1
        return "generated"


class FakeRetriever:
    def retrieve(self, **_: object) -> list[object]:
        return []

    def retrieve_by_document_contains(self, **_: object) -> list[RetrievedChunk]:
        return []


class FakeRetrieverLowScore:
    def retrieve(self, **_: object) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                chunk_id="c-low",
                source_file="historia.pdf",
                page=7,
                text="Niepowiązany fragment",
                score=0.12,
            )
        ]

    def retrieve_by_document_contains(self, **_: object) -> list[RetrievedChunk]:
        return []


class FakeRetrieverManyRelevant:
    def retrieve(self, **_: object) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                chunk_id=f"c-{idx}",
                source_file="historia.pdf",
                page=20 + idx,
                text=f"Mieszko I i Chrzest Polski 966 ({idx})",
                score=0.9 - (idx * 0.01),
            )
            for idx in range(6)
        ]

    def retrieve_by_document_contains(self, **_: object) -> list[RetrievedChunk]:
        return []


class FakeSummarizer:
    def should_summarize(self, **_: object) -> bool:
        return False


def test_chat_service_enforces_no_context_gating_without_generation() -> None:
    service = ChatService.__new__(ChatService)
    service.config = FakeConfig()
    service.persistence = SimpleNamespace(
        threads=FakeThreadsRepo(),
        messages=FakeMessagesRepo(),
        thread_state=FakeThreadStateRepo(),
    )
    service.ollama = FakeOllama()
    service.retriever = FakeRetriever()
    service.summarizer = FakeSummarizer()

    reply = service.respond(thread_id="thread-1", user_text="Kim był Mieszko I?")

    assert "Nie wiem na podstawie dostarczonych materiałów." in reply.content
    assert "Pytanie doprecyzowujące" in reply.content
    assert service.ollama.generate_calls == 0
    assert reply.sources == []
    assert reply.config_fingerprint["retrieval"] == []


def test_chat_service_enforces_no_context_when_best_score_below_threshold() -> None:
    service = ChatService.__new__(ChatService)
    service.config = FakeConfig(similarity_threshold=0.35)
    fake_messages = FakeMessagesRepo()
    service.persistence = SimpleNamespace(
        threads=FakeThreadsRepo(),
        messages=fake_messages,
        thread_state=FakeThreadStateRepo(),
    )
    service.ollama = FakeOllama()
    service.retriever = FakeRetrieverLowScore()
    service.summarizer = FakeSummarizer()

    reply = service.respond(thread_id="thread-1", user_text="kto to był Święty Wojciech")

    assert "Nie wiem na podstawie dostarczonych materiałów." in reply.content
    assert "[Źródło:" not in reply.content
    assert reply.sources == []
    assert service.ollama.generate_calls == 0
    assert reply.config_fingerprint["retrieval"] == [
        {"chunk_id": "c-low", "score": 0.12, "source_file": "historia.pdf"}
    ]
    assert reply.config_fingerprint["retrieval_summary"]["has_context"] is False
    assert reply.config_fingerprint["retrieval_summary"]["best_score"] == 0.12

    saved = fake_messages.by_thread["thread-1"][-1]
    assert saved.role == "assistant"
    assert saved.sources == []
    assert saved.token_count is not None
    assert saved.config_fingerprint is not None
    assert saved.config_fingerprint["retrieval_summary"]["has_context"] is False


def test_chat_service_limits_citations_to_three() -> None:
    service = ChatService.__new__(ChatService)
    service.config = FakeConfig(similarity_threshold=0.35, top_k=6)
    service.persistence = SimpleNamespace(
        threads=FakeThreadsRepo(),
        messages=FakeMessagesRepo(),
        thread_state=FakeThreadStateRepo(),
    )
    service.ollama = FakeOllama()
    service.retriever = FakeRetrieverManyRelevant()
    service.summarizer = FakeSummarizer()

    reply = service.respond(thread_id="thread-1", user_text="Kim był Mieszko I?")

    assert service.ollama.generate_calls == 1
    assert len(reply.sources) == 3
    assert reply.content.count("[Źródło:") == 3
