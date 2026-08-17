from __future__ import annotations

import json
from pathlib import Path

from voicemem.leftbrain.local_memory_store import default_memory_root
from voicemem.persona.types import PersonaDocument

_PERSONA_JSON = "persona.json"


def persona_json_path(root: Path | None = None) -> Path:
    base = root if root is not None else default_memory_root()
    base.mkdir(parents=True, exist_ok=True)
    return base / _PERSONA_JSON


class PersonaStore:
    def __init__(self, *, root: Path | None = None) -> None:
        self._path = persona_json_path(root)

    @property
    def path(self) -> Path:
        return self._path

    def load(self, *, user_id: str = "default") -> PersonaDocument:
        if not self._path.is_file():
            return PersonaDocument(user_id=user_id)
        with self._path.open(encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return PersonaDocument(user_id=user_id)
        users = data.get("users")
        if isinstance(users, dict):
            raw = users.get(user_id)
            if isinstance(raw, dict):
                doc = PersonaDocument.from_dict(raw)
                doc.user_id = user_id
                return doc
        if data.get("user_id") == user_id or "impression_text" in data:
            return PersonaDocument.from_dict(data)
        return PersonaDocument(user_id=user_id)

    def save(self, doc: PersonaDocument) -> None:
        existing: dict[str, object] = {"users": {}}
        if self._path.is_file():
            with self._path.open(encoding="utf-8") as f:
                cur = json.load(f)
            if isinstance(cur, dict) and isinstance(cur.get("users"), dict):
                existing = cur
            elif isinstance(cur, dict) and "impression_text" in cur:
                uid = str(cur.get("user_id", "default"))
                existing = {"users": {uid: cur}}

        users = existing.setdefault("users", {})
        if not isinstance(users, dict):
            users = {}
            existing["users"] = users
        users[doc.user_id] = doc.to_dict()
        self._path.write_text(
            json.dumps(existing, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
