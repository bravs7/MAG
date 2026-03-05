"""Console chat entrypoint."""

from __future__ import annotations

import logging
from pathlib import Path

from app.chat.service import ChatService
from app.config import AppConfig
from app.logging import configure_logging, get_logger
from app.persistence.export_jsonl import export_thread_to_jsonl

logger = get_logger(__name__)
CLI_HELP = (
    "Komendy CLI:\n"
    "/new | /threads | /export [sciezka] | /prefs\n"
    "/normal | /short | /extended\n"
    "/check on|off | /sentences N (1-20)\n"
    "/help | /exit"
)


def handle_slash_command(text: str, thread_id: str, service: ChatService) -> str | None:
    if not text.startswith("/"):
        return None

    command, _, args = text.partition(" ")
    cmd = command.strip().lower()
    arg = args.strip().lower()

    if cmd == "/help":
        return CLI_HELP
    if cmd == "/prefs":
        return service.format_thread_preferences(thread_id)
    if cmd == "/normal":
        service.update_thread_preferences(
            thread_id=thread_id,
            updates={"answer_style": "normal", "style_short": False},
        )
        return "Ustawiono tryb: normal"
    if cmd == "/short":
        service.update_thread_preferences(
            thread_id=thread_id,
            updates={"answer_style": "short", "style_short": True},
        )
        return "Ustawiono tryb: short"
    if cmd == "/extended":
        service.update_thread_preferences(
            thread_id=thread_id,
            updates={"answer_style": "extended", "style_short": False},
        )
        return "Ustawiono tryb: extended"
    if cmd == "/check":
        if arg == "on":
            service.update_thread_preferences(
                thread_id=thread_id,
                updates={"ask_check_question": True},
            )
            return "Pytanie kontrolne: włączone"
        if arg == "off":
            service.update_thread_preferences(
                thread_id=thread_id,
                updates={"ask_check_question": False},
            )
            return "Pytanie kontrolne: wyłączone"
        return "Błąd: użyj /check on lub /check off"
    if cmd == "/sentences":
        if not arg:
            return "Błąd: podaj liczbę, np. /sentences 3"
        try:
            limit = int(arg)
        except ValueError:
            return "Błąd: podaj liczbę, np. /sentences 3"
        if limit < 1 or limit > 20:
            return "Błąd: dozwolony zakres to 1-20."
        service.update_thread_preferences(
            thread_id=thread_id,
            updates={"max_sentences": limit},
        )
        return f"Maksymalna liczba zdań (część faktograficzna): {limit}"

    return "Nieznana komenda. Użyj /help"


def main() -> None:
    configure_logging(logging.INFO)
    config = AppConfig.from_env()
    service = ChatService(config)

    thread_id = service.create_thread(title="Nowa rozmowa")
    print("MAG teacher chat (Phase 1 console)")
    print("Komendy: /new, /threads, /export, /prefs, /normal, /short, /extended")
    print("         /check on|off, /sentences N, /help, /exit")
    print(f"Aktualny thread_id: {thread_id}")

    while True:
        try:
            user_text = input("\nTy: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nKoniec.")
            break

        if not user_text:
            continue

        if user_text == "/exit":
            print("Do zobaczenia.")
            break

        if user_text == "/new":
            thread_id = service.create_thread(title="Nowa rozmowa")
            print(f"Nowy thread_id: {thread_id}")
            continue

        if user_text == "/threads":
            rows = service.list_threads()
            for row in rows:
                print(
                    f"- {row['id']} | {row.get('title') or '(brak tytulu)'} "
                    f"| updated_at={row['updated_at']}"
                )
            continue

        if user_text.startswith("/export"):
            _, _, output = user_text.partition(" ")
            export_path = output.strip() or f"results/thread_{thread_id}.jsonl"
            output_path = Path(export_path)
            if not output_path.is_absolute():
                output_path = Path.cwd() / output_path
            count = export_thread_to_jsonl(
                db_path=config.db_path,
                thread_id=thread_id,
                output_path=output_path,
            )
            print(f"Wyeksportowano {count} wiadomosci do: {output_path}")
            continue

        slash_response = handle_slash_command(
            text=user_text,
            thread_id=thread_id,
            service=service,
        )
        if slash_response is not None:
            print(slash_response)
            continue

        try:
            reply = service.respond(thread_id=thread_id, user_text=user_text)
            print("\nNauczyciel:")
            print(reply.content)
        except Exception as exc:
            logger.exception("Chat step failed")
            print(f"Blad: {exc}")


if __name__ == "__main__":
    main()
