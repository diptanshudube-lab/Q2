"""SQLite storage. One connection behind one re-entrant lock, so each state change commits before the response goes out."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    principal TEXT NOT NULL,
    state TEXT NOT NULL,
    task_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS tasks_by_principal ON tasks (principal);
CREATE TABLE IF NOT EXISTS requests (
    principal TEXT NOT NULL,
    message_id TEXT NOT NULL,
    digest TEXT NOT NULL,
    task_id TEXT NOT NULL,
    PRIMARY KEY (principal, message_id)
);
CREATE TABLE IF NOT EXISTS decisions (
    cache_key TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    decision_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS model_calls (
    day TEXT PRIMARY KEY,
    calls INTEGER NOT NULL
);
"""


@dataclass(frozen=True)
class RequestRecord:
    digest: str
    task_id: str


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


class Store:
    def __init__(self, path: str) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self.lock = threading.RLock()
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA busy_timeout=5000")
        self._db.executescript(SCHEMA)

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Hold the lock and commit at the end. An exception rolls everything back."""
        with self.lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                yield
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
            self._db.execute("COMMIT")

    def _one(self, sql: str, params: tuple) -> tuple | None:
        with self.lock:
            return self._db.execute(sql, params).fetchone()

    def _all(self, sql: str, params: tuple) -> list[tuple]:
        with self.lock:
            return self._db.execute(sql, params).fetchall()

    def _write(self, sql: str, params: tuple) -> None:
        with self.lock:
            self._db.execute(sql, params)

    def find_request(self, principal: str, message_id: str) -> RequestRecord | None:
        row = self._one(
            "SELECT digest, task_id FROM requests WHERE principal = ? AND message_id = ?",
            (principal, message_id),
        )
        return RequestRecord(digest=row[0], task_id=row[1]) if row else None

    def add_request(self, principal: str, message_id: str, digest: str, task_id: str) -> None:
        self._write(
            "INSERT INTO requests (principal, message_id, digest, task_id) VALUES (?, ?, ?, ?)",
            (principal, message_id, digest, task_id),
        )

    def insert_task(self, principal: str, task: dict, message_id: str, digest: str) -> None:
        self._write(
            "INSERT INTO tasks (task_id, principal, state, task_json) VALUES (?, ?, ?, ?)",
            (task["id"], principal, task["status"]["state"], _dump(task)),
        )
        self.add_request(principal, message_id, digest, task["id"])

    def delete_task(self, principal: str, task_id: str, message_id: str) -> None:
        self._write(
            "DELETE FROM requests WHERE principal = ? AND message_id = ?",
            (principal, message_id),
        )
        self._write("DELETE FROM tasks WHERE task_id = ? AND principal = ?", (task_id, principal))

    def load_task(self, principal: str, task_id: str) -> dict | None:
        row = self._one(
            "SELECT task_json FROM tasks WHERE task_id = ? AND principal = ?",
            (task_id, principal),
        )
        return json.loads(row[0]) if row else None

    def save_task(self, principal: str, task: dict) -> None:
        self._write(
            "UPDATE tasks SET state = ?, task_json = ? WHERE task_id = ? AND principal = ?",
            (task["status"]["state"], _dump(task), task["id"], principal),
        )

    def list_tasks(self, principal: str, limit: int) -> list[dict]:
        rows = self._all(
            "SELECT task_json FROM tasks WHERE principal = ? ORDER BY rowid DESC LIMIT ?",
            (principal, limit),
        )
        return [json.loads(row[0]) for row in rows]

    def get_decision(self, cache_key: str) -> dict | None:
        row = self._one("SELECT decision_json FROM decisions WHERE cache_key = ?", (cache_key,))
        return json.loads(row[0]) if row else None

    def put_decision(self, cache_key: str, model: str, decision: dict) -> None:
        self._write(
            "INSERT OR REPLACE INTO decisions (cache_key, model, decision_json) VALUES (?, ?, ?)",
            (cache_key, model, _dump(decision)),
        )

    def reserve_model_call(self, day: str, limit: int) -> bool:
        """Count one model call for the day. Returns False when the daily limit is already used up."""
        with self.transaction():
            row = self._db.execute("SELECT calls FROM model_calls WHERE day = ?", (day,)).fetchone()
            calls = row[0] if row else 0
            if calls >= limit:
                return False
            self._db.execute(
                "INSERT OR REPLACE INTO model_calls (day, calls) VALUES (?, ?)",
                (day, calls + 1),
            )
            return True
