from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.chat.__main__ import handle_slash_command
from app.chat.service import ChatService
from app.persistence.repositories import Persistence


@dataclass
class FakeConfig:
    model_name: str = "test-model"
    top_k: int = 5

    def fingerprint(self) -> dict[str, object]:
        return {"llm": {"model": self.model_name}, "rag": {"top_k": self.top_k}}


class FakeOllama:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.embed_calls = 0

    def generate(self, **kwargs: object) -> str:
        prompt = kwargs.get("prompt")
        if isinstance(prompt, str):
            self.prompts.append(prompt)
        return "generated"

    def embed_texts(self, **_: object) -> list[list[float]]:
        self.embed_calls += 1
        return [[0.1, 0.2, 0.3]]


def _build_service(tmp_path: Path) -> tuple[ChatService, Persistence, FakeOllama, Path]:
    db_path = tmp_path / "chat.db"
    service = ChatService.__new__(ChatService)
    service.config = FakeConfig()
    service.persistence = Persistence(db_path=db_path)
    service.ollama = FakeOllama()
    service.summarizer = object()
    service.retriever = object()
    return service, service.persistence, service.ollama, db_path


def test_slash_style_command_does_not_call_llm(tmp_path: Path) -> None:
    service, persistence, fake_ollama, _ = _build_service(tmp_path)
    thread_id = "thread-slash-style"
    persistence.threads.upsert_thread(thread_id)

    response = handle_slash_command("/short", thread_id, service)

    assert response == "Ustawiono tryb: short"
    assert fake_ollama.prompts == []
    assert fake_ollama.embed_calls == 0


def test_slash_updates_thread_preferences_persistently(tmp_path: Path) -> None:
    service, persistence, _, db_path = _build_service(tmp_path)
    thread_id = "thread-slash-persist"
    persistence.threads.upsert_thread(thread_id)

    response = handle_slash_command("/short", thread_id, service)
    assert response == "Ustawiono tryb: short"

    reopened = Persistence(db_path=db_path)
    prefs = reopened.thread_state.get_preferences(thread_id)
    assert prefs["answer_style"] == "short"


def test_slash_sentences_validation(tmp_path: Path) -> None:
    service, persistence, _, _ = _build_service(tmp_path)
    thread_id = "thread-slash-sentences"
    persistence.threads.upsert_thread(thread_id)

    ok_message = handle_slash_command("/sentences 4", thread_id, service)
    err_zero = handle_slash_command("/sentences 0", thread_id, service)
    err_big = handle_slash_command("/sentences 999", thread_id, service)
    err_text = handle_slash_command("/sentences abc", thread_id, service)

    prefs = persistence.thread_state.get_preferences(thread_id)
    assert ok_message == "Maksymalna liczba zdań (część faktograficzna): 4"
    assert err_zero == "Błąd: dozwolony zakres to 1-20."
    assert err_big == "Błąd: dozwolony zakres to 1-20."
    assert err_text == "Błąd: podaj liczbę, np. /sentences 3"
    assert prefs["max_sentences"] == 4


def test_unknown_slash_command(tmp_path: Path) -> None:
    service, persistence, _, _ = _build_service(tmp_path)
    thread_id = "thread-slash-unknown"
    persistence.threads.upsert_thread(thread_id)

    response = handle_slash_command("/nope", thread_id, service)

    assert response == "Nieznana komenda. Użyj /help"
