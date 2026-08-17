from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

PersonaConfidence = Literal["low", "medium", "high"]


@dataclass
class PersonaDocument:
    """慢更新的用户画像文档（Mem0 式 narrative + 可执行 hints）。"""

    version: int = 0
    user_id: str = "default"
    impression_text: str = ""
    interaction_hints: list[str] = field(default_factory=list)
    trait_vs_state: dict[str, list[str]] = field(default_factory=dict)
    confidence: PersonaConfidence = "low"
    changelog: str = ""
    updated_at: str | None = None
    sources: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "user_id": self.user_id,
            "impression_text": self.impression_text,
            "interaction_hints": list(self.interaction_hints),
            "trait_vs_state": dict(self.trait_vs_state),
            "confidence": self.confidence,
            "changelog": self.changelog,
            "updated_at": self.updated_at,
            "sources": dict(self.sources),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PersonaDocument:
        hints = raw.get("interaction_hints")
        tvs = raw.get("trait_vs_state")
        conf = raw.get("confidence")
        return cls(
            version=int(raw.get("version", 0)),
            user_id=str(raw.get("user_id", "default")),
            impression_text=str(raw.get("impression_text", "")),
            interaction_hints=[str(x) for x in hints] if isinstance(hints, list) else [],
            trait_vs_state=tvs if isinstance(tvs, dict) else {},
            confidence=conf if conf in ("low", "medium", "high") else "low",
            changelog=str(raw.get("changelog", "")),
            updated_at=raw.get("updated_at"),
            sources=raw.get("sources") if isinstance(raw.get("sources"), dict) else {},
        )


@dataclass
class PersonaUpdateSnapshot:
    """Persona 合成器输入快照。"""

    user_id: str
    stats: dict[str, Any]
    anchors: list[dict[str, str]]
    new_memories: list[dict[str, str]]
    #: 刚跨边界产生的 holistic 正文
    new_boundary_holistics: list[str]
    #: 更早 holistic 档案正文
    archival_holistics: list[str]
    previous_impression: str
    persona_version: int
