"""SQLite database initialization and connection helpers."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class Database:
    def __init__(self, db_path: Path, schema_path: Path) -> None:
        self._db_path = db_path
        self._schema_path = schema_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def init_schema(self) -> None:
        sql = self._schema_path.read_text(encoding="utf-8")
        with self.connect() as conn:
            conn.executescript(sql)
            self._ensure_preferences_column(conn)

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        conn = self.connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _ensure_preferences_column(conn: sqlite3.Connection) -> None:
        rows = conn.execute("PRAGMA table_info(thread_state)").fetchall()
        existing = {str(row["name"]) for row in rows}
        if "preferences_json" in existing:
            return
        conn.execute(
            "ALTER TABLE thread_state ADD COLUMN preferences_json TEXT NOT NULL DEFAULT '{}'"
        )
