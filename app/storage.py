"""具名光路档的进程内 SQLite 存取。"""
from __future__ import annotations

import json
import sqlite3
import threading

from .validation import DuplicateSystem


class SystemStore:
    """光路档存取。单连接加锁，供并行请求安全读写。"""

    def __init__(self, path: str = "systems.db") -> None:
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS systems ("
                "  name TEXT PRIMARY KEY,"
                "  elements TEXT NOT NULL"
                ")"
            )
            self._conn.commit()

    def register(self, name: str, elements: list[dict]) -> None:
        """登记新档；同名已存在时抛 DuplicateSystem。"""
        payload = json.dumps(elements)
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO systems (name, elements) VALUES (?, ?)",
                    (name, payload),
                )
                self._conn.commit()
            except sqlite3.IntegrityError:
                raise DuplicateSystem(f"光路档已存在: {name!r}", field="name") from None

    def ensure(self, name: str, elements: list[dict]) -> None:
        """已存在则跳过（用于内置示范档）。"""
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO systems (name, elements) VALUES (?, ?)",
                (name, json.dumps(elements)),
            )
            self._conn.commit()

    def get(self, name: str) -> list[dict] | None:
        """按名取档，未登记返回 None。"""
        with self._lock:
            row = self._conn.execute(
                "SELECT elements FROM systems WHERE name = ?", (name,)
            ).fetchone()
        return None if row is None else json.loads(row[0])

    def list(self) -> list[dict]:
        """列出全部档，含元件种类与参数全文。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT name, elements FROM systems ORDER BY name"
            ).fetchall()
        return [{"name": name, "elements": json.loads(payload)} for name, payload in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
