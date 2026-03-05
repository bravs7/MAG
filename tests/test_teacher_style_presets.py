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
        self.prompts: list[str] = []

    def generate(self, **kwargs: object) -> str:
        self.generate_calls += 1
        prompt = kwargs.get("prompt")
        if isinstance(prompt, str):
            self.prompts.append(prompt)
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


def _single_chunk_bundle(chunk: RetrievedChunk, *, keyword: str) -> RetrievalBundle:
    return RetrievalBundle(
        final_chunks=[chunk],
        candidates=[],
        keywords=[keyword],
        main_keyword=keyword,
        phrase_norm=None,
        query_evidence=True,
        lexical_fallback_used=False,
    )


def _body_without_citations(content: str) -> str:
    return re.split(r"\n\[(?:Źródło|Zrodlo):", content, maxsplit=1)[0]


def _extract_learning_section(content: str) -> str:
    match = re.search(
        r"Jak to zapamiętać:\n(?P<section>.*?)(?:\n\nSprawdź się:|\n\[(?:Źródło|Zrodlo):|$)",
        content,
        flags=re.DOTALL,
    )
    if match is None:
        return ""
    return match.group("section").strip()


def test_default_style_is_normal(tmp_path: Path) -> None:
    chunk = RetrievedChunk(
        chunk_id="c-wojciech",
        source_file="historia.pdf",
        page=23,
        text="Święty Wojciech był biskupem i misjonarzem.",
        score=0.88,
    )
    service, _, _ = _build_service(
        tmp_path,
        generated="Święty Wojciech był biskupem i misjonarzem.",
    )
    _bind_retrieve(service, _single_chunk_bundle(chunk, keyword="wojciech"))

    reply = service.respond(thread_id="thread-style-default", user_text="Kim był Święty Wojciech?")

    assert "Odpowiedź (z materiału):" in reply.content
    assert "Sprawdź się:" in reply.content
    assert "[Źródło:" in reply.content
    assert "Nie wiem na podstawie dostarczonych materiałów." not in reply.content


def test_short_style_disables_extra_sections(tmp_path: Path) -> None:
    chunk = RetrievedChunk(
        chunk_id="c-mieszko",
        source_file="historia.pdf",
        page=21,
        text="Mieszko I był księciem Polan i przyjął chrzest w 966 roku.",
        score=0.92,
    )
    service, persistence, _ = _build_service(
        tmp_path,
        generated=(
            "Mieszko I był księciem Polan. Umocnił władzę. "
            "Przyjął chrzest w 966 roku. Rozwinął państwo. "
            "Czy to wyjaśnienie jest dla Ciebie zrozumiałe?"
        ),
    )
    _bind_retrieve(service, _single_chunk_bundle(chunk, keyword="mieszko"))

    thread_id = "thread-style-short"
    persistence.threads.upsert_thread(thread_id)
    persistence.thread_state.upsert_preferences(
        thread_id=thread_id,
        preferences={"answer_style": "short", "ask_check_question": False},
    )

    reply = service.respond(thread_id=thread_id, user_text="Kim był Mieszko I?")
    body = _body_without_citations(reply.content)
    sentences = [part for part in re.split(r"(?<=[.!?])\s+", body) if part.strip()]

    assert "Jak to zapamiętać:" not in reply.content
    assert "Sprawdź się:" not in reply.content
    assert len(sentences) <= 3
    assert "[Źródło:" in reply.content


def test_extended_style_adds_learning_section_without_new_facts(tmp_path: Path) -> None:
    chunk = RetrievedChunk(
        chunk_id="c-mieszko-extended",
        source_file="historia.pdf",
        page=21,
        text="Mieszko I przyjął chrzest w 966 roku i umocnił państwo.",
        score=0.89,
    )
    service, persistence, _ = _build_service(
        tmp_path,
        generated=(
            "Mieszko I przyjął chrzest w 966 roku. "
            "Umocnił pozycję państwa. "
            "Czy to jest dla Ciebie zrozumiałe?"
        ),
    )
    _bind_retrieve(service, _single_chunk_bundle(chunk, keyword="mieszko"))

    thread_id = "thread-style-extended"
    persistence.threads.upsert_thread(thread_id)
    persistence.thread_state.upsert_preferences(
        thread_id=thread_id,
        preferences={"answer_style": "extended", "ask_check_question": True},
    )

    reply = service.respond(thread_id=thread_id, user_text="Kim był Mieszko I?")
    learning_section = _extract_learning_section(reply.content)

    assert "Jak to zapamiętać:" in reply.content
    assert learning_section
    assert re.search(r"\d", learning_section) is None
    assert "Mieszko" not in learning_section
    assert "[Źródło:" in reply.content


def test_meta_command_sets_style_without_llm(tmp_path: Path) -> None:
    service, persistence, fake_ollama = _build_service(
        tmp_path,
        generated="To nie powinno się wykonać.",
    )
    thread_id = "thread-meta-style"

    reply = service.respond(
        thread_id=thread_id,
        user_text="Ustaw styl odpowiedzi: extended.",
    )

    prefs = persistence.thread_state.get_preferences(thread_id)
    assert "Ustawienia zapisane." in reply.content
    assert "Styl odpowiedzi" in reply.content
    assert prefs["answer_style"] == "extended"
    assert reply.sources == []
    assert fake_ollama.generate_calls == 0
    assert fake_ollama.embed_calls == 0
