"""前刺层 (Pre-stimulus Layer)。

无条件在每轮对话开始时注入 system prompt 的上下文，包含：
  1. user_interaction_profile — 用户偏好（UserProfileStore）
  2. 近期 tasks/todo       — 从左脑 cognitive graph 读取（TaskFetcher）

与右脑不同：右脑是基于当前 anchor 动态检索的，前刺层是静态的、始终存在的。
"""
from .profile_store import UserProfileStore
from .task_fetcher import TaskFetcher
from pathlib import Path


class PreStimulusLayer:
    """统一封装前刺层两个数据源，供 demo 和真实系统使用。"""

    def __init__(
        self,
        *,
        profile_db_path: Path | str,
        cognitive_db_path: Path | str,
        leftbrain_db_path: Path | str,
    ) -> None:
        self._profiles = UserProfileStore(profile_db_path)
        self._tasks    = TaskFetcher(cognitive_db_path, leftbrain_db_path)

    def get_profiles(self, user_id: str) -> list[str]:
        return self._profiles.get_all(user_id)

    def get_tasks(self, user_id: str, limit: int = 5) -> list[str]:
        return self._tasks.get_recent_tasks(user_id, limit=limit)

    def build_text(self, user_id: str) -> str:
        """生成注入 system prompt 的前刺文本块。"""
        lines: list[str] = []
        profs = self.get_profiles(user_id)
        if profs:
            lines.append("=== How this user prefers responses ===")
            lines.extend(f"• {p}" for p in profs)
        tasks = self.get_tasks(user_id)
        if tasks:
            lines.append("=== Urgent tasks on the user's mind ===")
            lines.extend(f"• {t}" for t in tasks)
        return "\n".join(lines)

    def build_structured(self, user_id: str) -> dict:
        """返回结构化数据，供 UI /api/prefill 端点使用。"""
        return {
            "profiles": self.get_profiles(user_id),
            "tasks":    self.get_tasks(user_id),
        }

    # ── write helpers (seeding / runtime learning) ───────────────────────────

    def write_profile(self, user_id: str, content: str, *, priority: float = 0.5) -> str:
        return self._profiles.write(user_id, content, priority=priority)

    def replace_auto_persona(self, user_id: str, content: str, *, priority: float = 0.7) -> str:
        return self._profiles.replace_auto_persona(user_id, content, priority=priority)


__all__ = ["PreStimulusLayer", "UserProfileStore", "TaskFetcher"]
