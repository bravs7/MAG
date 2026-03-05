"""Chainlit UI entrypoint (Phase 2 optional)."""

from __future__ import annotations

try:
    import chainlit as cl
except ImportError as exc:  # pragma: no cover - optional dependency
    raise RuntimeError(
        "chainlit is not installed. Install optional dependencies with: uv sync --extra phase2"
    ) from exc

from app.chat.service import ChatService
from app.config import AppConfig

config = AppConfig.from_env()
service = ChatService(config)
THREAD_KEY = "thread_id"


@cl.on_chat_start
async def on_chat_start() -> None:
    thread_id = service.create_thread(title="Chainlit chat")
    cl.user_session.set(THREAD_KEY, thread_id)
    await cl.Message(content="Witaj. Mozesz zadawac pytania z historii Polski.").send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    thread_id = cl.user_session.get(THREAD_KEY)
    if not thread_id:
        thread_id = service.create_thread(title="Chainlit chat")
        cl.user_session.set(THREAD_KEY, thread_id)

    reply = service.respond(thread_id=thread_id, user_text=message.content)
    await cl.Message(content=reply.content).send()
