"""Teacher persona policy utilities."""

from __future__ import annotations

from app.types import RetrievedChunk

NO_CONTEXT_MESSAGE = "Nie wiem na podstawie dostarczonych materiałów."


def is_open_learning_request(user_text: str) -> bool:
    lowered = user_text.lower()
    markers = ["pomoz", "naucz", "jak sie nauczyc", "wytlumacz krok po kroku", "nie rozumiem"]
    return any(marker in lowered for marker in markers)


def build_teacher_rules(*, has_context: bool, user_text: str) -> str:
    if not has_context:
        return (
            "Brak wystarczającego kontekstu z materiałów. "
            "Nie odpowiadaj faktograficznie bez źródeł. "
            "Napisz, że nie wiesz na podstawie materiałów, i zadaj pytanie doprecyzowujące."
        )

    if is_open_learning_request(user_text):
        return (
            "Tryb nauczyciela: zacznij od 1-3 pytań naprowadzających, "
            "następnie krótko wyjaśnij temat i daj mikro-zadanie."
        )

    return (
        "Tryb nauczyciela: udziel krótkiej odpowiedzi opartej na kontekście, "
        "dodaj cytowania i zakończ jednym pytaniem sprawdzającym zrozumienie."
    )


def validate_context(chunks: list[RetrievedChunk], *, similarity_threshold: float) -> bool:
    if not chunks:
        return False
    best = max(chunk.score for chunk in chunks)
    return best >= similarity_threshold


def build_no_context_response() -> str:
    return (
        f"{NO_CONTEXT_MESSAGE}\n\n"
        "Pytanie doprecyzowujące: o jaki okres, wydarzenie lub postać historyczną chodzi?\n"
        "Co możesz dodać: wskaż nazwę PDF lub wklej fragment materiału źródłowego, "
        "na którym mam się oprzeć."
    )
