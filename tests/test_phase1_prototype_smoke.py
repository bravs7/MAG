from __future__ import annotations

from dataclasses import dataclass
from types import MethodType, SimpleNamespace

from app.chat.service import ChatService, RetrievalBundle
from app.persistence.repositories import ThreadStateRecord
from app.types import ChatMessage, RetrievedChunk, SourceCitation, utc_now_iso

NO_CONTEXT_PREFIX = "Nie wiem na podstawie dostarczonych materiałów."
FORBIDDEN_MARKERS = [
    "z tego co pamiętam",
    "z tego co pamietam",
    "wydaje mi się",
    "wydaje mi sie",
]


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
    def __init__(self, generated_responses: list[str]) -> None:
        self.generated_responses = list(generated_responses)
        self.generate_calls = 0

    def generate(self, **_: object) -> str:
        self.generate_calls += 1
        if not self.generated_responses:
            return "Brak odpowiedzi."
        return self.generated_responses.pop(0)


def test_phase1_prototype_smoke_chatservice() -> None:
    generated_responses = [
        "Mieszko I był synem Piasta Kołodzieja.",
        "Nie wiem na podstawie dostarczonych materiałów.",
        "Zjazd gnieźnieński odbył się w 1000 roku. Czy to dla Ciebie jasne?",
        "Chrzest Polski był w 1225 roku. Czy to jasne?",
    ]

    retrieval_map = {
        "Kim był Mieszko I?": RetrievalBundle(
            final_chunks=[
                RetrievedChunk(
                    chunk_id="c-mieszko",
                    source_file="historia.pdf",
                    page=21,
                    text="Mieszko I przyjął chrzest i wzmocnił pozycję państwa.",
                    score=0.71,
                ),
                RetrievedChunk(
                    chunk_id="c-noise",
                    source_file="historia.pdf",
                    page=50,
                    text="Niepowiązany fragment bez tej postaci.",
                    score=0.95,
                ),
            ],
            candidates=[],
            keywords=["mieszko"],
            main_keyword="mieszko",
            phrase_norm=None,
            query_evidence=True,
            lexical_fallback_used=False,
        ),
        "Kim był Bolesław Chrobry?": RetrievalBundle(
            final_chunks=[
                RetrievedChunk(
                    chunk_id="c-boleslaw",
                    source_file="historia.pdf",
                    page=22,
                    text="Bolesław prowadził działania wzmacniające państwo.",
                    score=0.72,
                )
            ],
            candidates=[],
            keywords=["boleslaw", "chrobry"],
            main_keyword="boleslaw",
            phrase_norm="boleslaw chrobry",
            query_evidence=True,
            lexical_fallback_used=False,
        ),
        "Czym był Zjazd Gnieźnieński?": RetrievalBundle(
            final_chunks=[
                RetrievedChunk(
                    chunk_id="c-zjazd",
                    source_file="historia.pdf",
                    page=22,
                    text="Zjazd gnieźnieński odbył się w 1000 roku.",
                    score=0.92,
                )
            ],
            candidates=[],
            keywords=["zjazd", "gnieznienski"],
            main_keyword="gnieznienski",
            phrase_norm="zjazd gnieznienski",
            query_evidence=True,
            lexical_fallback_used=False,
        ),
        "Jak działa blockchain?": RetrievalBundle(
            final_chunks=[],
            candidates=[],
            keywords=["blockchain"],
            main_keyword="blockchain",
            phrase_norm=None,
            query_evidence=False,
            lexical_fallback_used=False,
        ),
        "W którym roku był chrzest Polski?": RetrievalBundle(
            final_chunks=[
                RetrievedChunk(
                    chunk_id="c-chrzest",
                    source_file="historia.pdf",
                    page=21,
                    text="Chrzest Polski wiąże się z rokiem 966.",
                    score=0.78,
                )
            ],
            candidates=[],
            keywords=["chrzest", "polski"],
            main_keyword="chrzest",
            phrase_norm="chrzest polski",
            query_evidence=True,
            lexical_fallback_used=False,
        ),
    }

    service = ChatService.__new__(ChatService)
    service.config = FakeConfig()
    service.persistence = SimpleNamespace(
        threads=FakeThreadsRepo(),
        messages=FakeMessagesRepo(),
        thread_state=FakeThreadStateRepo(),
    )
    service.ollama = FakeOllama(generated_responses)
    service.summarizer = FakeSummarizer()
    service.retriever = object()

    def fake_retrieve(
        self: ChatService,
        *,
        user_text: str,
        request_timeout_seconds: float | None = None,
    ) -> RetrievalBundle:
        _ = request_timeout_seconds
        return retrieval_map[user_text]

    service._retrieve = MethodType(fake_retrieve, service)

    scenarios = [
        ("Kim był Mieszko I?", True),
        ("Kim był Bolesław Chrobry?", True),
        ("Czym był Zjazd Gnieźnieński?", True),
        ("Jak działa blockchain?", False),
        ("W którym roku był chrzest Polski?", False),
    ]

    for question, expect_citations in scenarios:
        reply = service.respond(thread_id="smoke-thread", user_text=question)

        lowered = reply.content.lower()
        for marker in FORBIDDEN_MARKERS:
            assert marker not in lowered

        non_citation_lines = [
            line for line in reply.content.splitlines() if line.strip() and "[Źródło:" not in line
        ]
        body_without_citations = "\n".join(non_citation_lines)
        assert body_without_citations.count("?") == 1

        if expect_citations:
            assert len(reply.sources) >= 1
            assert NO_CONTEXT_PREFIX not in reply.content
        else:
            assert NO_CONTEXT_PREFIX in reply.content
            assert reply.sources == []
