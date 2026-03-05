"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


@dataclass(slots=True)
class AppConfig:
    db_path: Path = Path("./data/chat.db")
    chroma_dir: Path = Path("./data/chroma")
    collection_name: str = "mag_chunks"

    ollama_host: str = "http://127.0.0.1:11434"
    model_name: str = "qwen2.5:7b-instruct-q4_K_M"
    embed_model: str = "nomic-embed-text"

    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 5
    similarity_threshold: float = 0.35
    debug_retrieval: bool = False
    retrieval_debug_top_n: int = 15

    n_turns: int = 8
    keep_last_turns: int = 6
    summary_trigger_tokens: int = 3200
    summary_trigger_turns: int = 14
    max_prompt_tokens: int = 6000

    temperature: float = 0.2
    top_p: float = 0.9

    @classmethod
    def from_env(cls, env_file: str | None = ".env") -> AppConfig:
        if env_file:
            load_dotenv(env_file, override=False)

        return cls(
            db_path=Path(os.getenv("DB_PATH", "./data/chat.db")),
            chroma_dir=Path(os.getenv("CHROMA_DIR", "./data/chroma")),
            collection_name=os.getenv("COLLECTION_NAME", "mag_chunks"),
            ollama_host=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
            model_name=os.getenv("MODEL_NAME", "qwen2.5:7b-instruct-q4_K_M"),
            embed_model=os.getenv("EMBED_MODEL", "nomic-embed-text"),
            chunk_size=_int_env("CHUNK_SIZE", 800),
            chunk_overlap=_int_env("CHUNK_OVERLAP", 120),
            top_k=_int_env("TOP_K", 5),
            similarity_threshold=_float_env("SIMILARITY_THRESHOLD", 0.35),
            debug_retrieval=_bool_env("DEBUG_RETRIEVAL", False),
            retrieval_debug_top_n=_int_env("RETRIEVAL_DEBUG_TOP_N", 15),
            n_turns=_int_env("N_TURNS", 8),
            keep_last_turns=_int_env("KEEP_LAST_TURNS", 6),
            summary_trigger_tokens=_int_env("SUMMARY_TRIGGER_TOKENS", 3200),
            summary_trigger_turns=_int_env("SUMMARY_TRIGGER_TURNS", 14),
            max_prompt_tokens=_int_env("MAX_PROMPT_TOKENS", 6000),
            temperature=_float_env("TEMPERATURE", 0.2),
            top_p=_float_env("TOP_P", 0.9),
        )

    def ensure_dirs(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)

    def fingerprint(self) -> dict[str, Any]:
        return {
            "llm": {
                "model": self.model_name,
                "host": self.ollama_host,
                "temperature": self.temperature,
                "top_p": self.top_p,
            },
            "embeddings": {
                "model": self.embed_model,
                "host": self.ollama_host,
            },
            "rag": {
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "top_k": self.top_k,
                "similarity_threshold": self.similarity_threshold,
                "debug_retrieval": self.debug_retrieval,
                "retrieval_debug_top_n": self.retrieval_debug_top_n,
                "collection_name": self.collection_name,
            },
            "memory": {
                "n_turns": self.n_turns,
                "keep_last_turns": self.keep_last_turns,
                "summary_trigger_tokens": self.summary_trigger_tokens,
                "summary_trigger_turns": self.summary_trigger_turns,
                "max_prompt_tokens": self.max_prompt_tokens,
            },
        }

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["db_path"] = str(self.db_path)
        payload["chroma_dir"] = str(self.chroma_dir)
        return payload


def _int_env(key: str, default: int) -> int:
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {key} must be int, got {value!r}") from exc


def _float_env(key: str, default: float) -> float:
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {key} must be float, got {value!r}") from exc


def _bool_env(key: str, default: bool) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Environment variable {key} must be bool-like, got {value!r}")
