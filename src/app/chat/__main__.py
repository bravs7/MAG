"""Console chat entrypoint."""

from __future__ import annotations

import logging
from pathlib import Path

from app.chat.service import ChatService
from app.config import AppConfig
from app.logging import configure_logging, get_logger
from app.persistence.export_jsonl import export_thread_to_jsonl

logger = get_logger(__name__)


def main() -> None:
    configure_logging(logging.INFO)
    config = AppConfig.from_env()
    service = ChatService(config)

    thread_id = service.create_thread(title="Nowa rozmowa")
    print("MAG teacher chat (Phase 1 console)")
    print("Komendy: /new, /threads, /export, /exit")
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

        try:
            reply = service.respond(thread_id=thread_id, user_text=user_text)
            print("\nNauczyciel:")
            print(reply.content)
        except Exception as exc:
            logger.exception("Chat step failed")
            print(f"Blad: {exc}")


if __name__ == "__main__":
    main()
