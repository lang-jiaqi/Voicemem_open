"""用户交互偏好存储 — 前刺层组件。

user_interaction_profile 是长期稳定的偏好，不通过 anchor 动态检索，
每轮对话开始时无条件全量注入 system prompt。
"""
from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


class UserProfileStore:
    def __init__(self, db_path: Path | str) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self._path)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        return c

    def _ensure_schema(self) -> None:
        with self._conn() as c:
            c.executescript("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                id         TEXT PRIMARY KEY,
                user_id    TEXT NOT NULL,
                content    TEXT NOT NULL,
                priority   REAL NOT NULL DEFAULT 0.5,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_up_user ON user_profiles(user_id, priority);
            """)
            # 迁移：老库补上 kind 列——区分手动/外部种子的 profile（kind=NULL）
            # 和自动生成的长期人格摘要（kind='auto_persona'），互不干扰。
            cols = {row["name"] for row in c.execute("PRAGMA table_info(user_profiles)")}
            if "kind" not in cols:
                c.execute("ALTER TABLE user_profiles ADD COLUMN kind TEXT")

    def write(self, user_id: str, content: str, *, priority: float = 0.5, kind: str | None = None) -> str:
        mid = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as c:
            c.execute(
                "INSERT INTO user_profiles (id, user_id, content, priority, created_at, kind)"
                " VALUES (?,?,?,?,?,?)",
                (mid, user_id, content, priority, now, kind),
            )
        return mid

    def replace_auto_persona(self, user_id: str, content: str, *, priority: float = 0.7) -> str:
        """覆盖式写入自动生成的长期人格摘要（session 边界触发）。

        每次都是"当前最新汇总"，不是追加证据——旧的 auto_persona 快照先删掉
        再插新的，否则 get_all() 会攒一堆过期的历史版本。
        """
        with self._conn() as c:
            c.execute(
                "DELETE FROM user_profiles WHERE user_id=? AND kind='auto_persona'",
                (user_id,),
            )
        return self.write(user_id, content, priority=priority, kind="auto_persona")

    def get_all(self, user_id: str) -> list[str]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT content FROM user_profiles WHERE user_id=? ORDER BY priority DESC",
                (user_id,),
            ).fetchall()
        return [r["content"] for r in rows]

    def delete_user(self, user_id: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM user_profiles WHERE user_id=?", (user_id,))
