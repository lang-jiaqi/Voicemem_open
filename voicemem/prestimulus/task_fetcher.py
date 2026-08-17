"""前刺层 — 任务 todo 读取。

从左脑 cognitive graph 中查找 slot=tasks 的记忆，作为"近期需要处理的事项"
注入 system prompt。这些任务来自左脑存储，不另外维护独立表。
"""
from __future__ import annotations

import sqlite3
from pathlib import Path


class TaskFetcher:
    def __init__(self, cognitive_db_path: Path | str, leftbrain_db_path: Path | str) -> None:
        self._cog_db   = Path(cognitive_db_path)
        self._lb_db    = Path(leftbrain_db_path)

    def get_recent_tasks(self, user_id: str, limit: int = 5) -> list[str]:
        """返回最近标注为 tasks slot 的记忆文本，按写入顺序降序。"""
        if not self._cog_db.exists() or not self._lb_db.exists():
            return []
        try:
            # 1. 从 cognitive graph 拿到 tasks slot 对应的 memory ids
            cog = sqlite3.connect(self._cog_db)
            cog.row_factory = sqlite3.Row
            rows = cog.execute(
                "SELECT id FROM memories WHERE user_id=? AND slot=?",
                (user_id, "tasks"),
            ).fetchall()
            cog.close()
            task_mids = [r["id"] for r in rows]
            if not task_mids:
                return []
            # 2. 从左脑 SQLite 读取对应文本
            ph  = ",".join("?" * len(task_mids))
            lb  = sqlite3.connect(self._lb_db)
            texts = [r[0] for r in lb.execute(
                f"SELECT text FROM memories WHERE id IN ({ph}) ORDER BY rowid DESC LIMIT ?",
                [*task_mids, limit],
            ).fetchall()]
            lb.close()
            return texts
        except Exception:
            return []
