from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Protocol

from voicemem_core.emotion.persona_input import (
    RightBrainPersonaInput,
    build_right_brain_persona_input_from_closure,
)
from voicemem_core.emotion.types import SessionClosureResult
from voicemem_core.persona._time import utc_iso
from voicemem_core.persona.build_snapshot import build_persona_snapshot, build_persona_snapshot_from_emotion_result
from voicemem_core.persona.prompt import PERSONA_SYSTEM_PROMPT, build_persona_user_prompt
from voicemem_core.persona.store import PersonaStore
from voicemem_core.persona.types import PersonaDocument, PersonaUpdateSnapshot


@dataclass
class PersonaUpdaterConfig:
    #: Persona 仅在 session 边界更新：与 ``EmotionLayer`` 一致，由相邻轮次
    #: ``time_gap_from_prev_turn_s`` 相对 ``session_gap_threshold_s``（默认 6h）判定。
    #: 为 False 时不会自动更新（仍可 ``update_from_snapshot(..., force=True)``）。
    update_on_session_boundary: bool = True
    model: str | None = None

    def resolved_model(self) -> str:
        return (self.model or os.environ.get("OPENAI_MODEL", "").strip() or "gpt-4o-mini").strip()


class PersonaSynthesizer(Protocol):
    def synthesize(self, snapshot: PersonaUpdateSnapshot) -> dict[str, Any]: ...


class StubPersonaSynthesizer:
    def synthesize(self, snapshot: PersonaUpdateSnapshot) -> dict[str, Any]:
        anchor_lines = [a.get("memory", "") for a in snapshot.anchors[:3]]
        new_lines = [m.get("memory", "") for m in snapshot.new_memories[:3]]
        bits = [x for x in anchor_lines + new_lines if x]
        summary = "；".join(bits) if bits else "尚缺乏足够交互信息。"
        hol = snapshot.new_boundary_holistics[:1]
        if hol:
            summary += f" 上一段情绪复盘：{hol[0][:80]}"
        return {
            "impression_text": (
                f"该用户（画像 v{snapshot.persona_version + 1}，置信度偏低）。"
                f"基于现有记录：{summary}"
                "（占位；配置 OPENAI_API_KEY 后可生成完整 narrative。）"
            ),
            "interaction_hints": [
                "回复简洁，先确认用户意图再给建议。",
                "避免重复询问已在记忆中出现过的事实。",
            ],
            "trait_vs_state": {"stable_traits": [], "recent_states": []},
            "confidence": "low",
            "changelog": "stub persona synthesis",
        }


class OpenAIPersonaSynthesizer:
    def __init__(self, config: PersonaUpdaterConfig | None = None) -> None:
        self._cfg = config or PersonaUpdaterConfig()

    def synthesize(self, snapshot: PersonaUpdateSnapshot) -> dict[str, Any]:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("请安装: pip install openai>=1.0") from e

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("缺少 OPENAI_API_KEY")

        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=self._cfg.resolved_model(),
            messages=[
                {"role": "system", "content": PERSONA_SYSTEM_PROMPT},
                {"role": "user", "content": build_persona_user_prompt(snapshot)},
            ],
            response_format={"type": "json_object"},
            temperature=0.25,
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("persona synthesis must return a JSON object")
        return data


class PersonaUpdater:
    def __init__(
        self,
        *,
        persona_store: PersonaStore | None = None,
        synthesizer: PersonaSynthesizer | None = None,
        config: PersonaUpdaterConfig | None = None,
    ) -> None:
        self._persona = persona_store or PersonaStore()
        self._config = config or PersonaUpdaterConfig()
        self._synthesizer = synthesizer or self._default_synthesizer()

    def _default_synthesizer(self) -> PersonaSynthesizer:
        if (os.environ.get("OPENAI_API_KEY") or "").strip():
            return OpenAIPersonaSynthesizer(self._config)
        return StubPersonaSynthesizer()

    def should_update(
        self,
        snapshot: PersonaUpdateSnapshot,
        *,
        session_boundary_crossed: bool = False,
    ) -> bool:
        _ = snapshot  # 边界由右脑 time_gap 驱动，与左脑增量条数无关
        return bool(self._config.update_on_session_boundary and session_boundary_crossed)

    def update_from_snapshot(
        self,
        snapshot: PersonaUpdateSnapshot,
        *,
        force: bool = False,
        session_boundary_crossed: bool = False,
    ) -> PersonaDocument | None:
        if not force and not self.should_update(snapshot, session_boundary_crossed=session_boundary_crossed):
            return None

        raw = self._synthesizer.synthesize(snapshot)
        prev = self._persona.load(user_id=snapshot.user_id)

        doc = PersonaDocument(
            version=prev.version + 1,
            user_id=snapshot.user_id,
            impression_text=str(raw.get("impression_text", "")).strip(),
            interaction_hints=_as_str_list(raw.get("interaction_hints")),
            trait_vs_state=raw.get("trait_vs_state") if isinstance(raw.get("trait_vs_state"), dict) else {},
            confidence=raw.get("confidence") if raw.get("confidence") in ("low", "medium", "high") else "low",
            changelog=str(raw.get("changelog", "")).strip(),
            updated_at=utc_iso(),
            sources={
                "anchor_ids": [a["id"] for a in snapshot.anchors],
                "new_memory_ids": [m["id"] for m in snapshot.new_memories],
            },
        )
        self._persona.save(doc)
        return doc

    def update_after_session_closure(
        self,
        closure: SessionClosureResult,
        *,
        user_id: str = "default",
        left_memories: list[dict[str, Any]],
        emotion_layer: Any | None = None,
    ) -> PersonaDocument | None:
        """主动跨边界（无新用户轮次）后更新 Persona。"""
        prev = self._persona.load(user_id=user_id)
        archival = emotion_layer.archival_holistics if emotion_layer is not None else []
        rb = build_right_brain_persona_input_from_closure(closure, archival_holistics=archival)
        snapshot = build_persona_snapshot(
            user_id=user_id,
            previous=prev,
            left_memories=left_memories,
            right_brain=rb,
        )
        return self.update_from_snapshot(snapshot, session_boundary_crossed=True)

    def update(
        self,
        *,
        user_id: str = "default",
        left_memories: list[dict[str, Any]],
        right_brain: RightBrainPersonaInput | None = None,
        emotion_layer: Any | None = None,
        emotion_result: Any | None = None,
        force: bool = False,
    ) -> PersonaDocument | None:
        prev = self._persona.load(user_id=user_id)
        if emotion_result is not None:
            snapshot = build_persona_snapshot_from_emotion_result(
                user_id=user_id,
                previous=prev,
                left_memories=left_memories,
                emotion_result=emotion_result,
                emotion_layer=emotion_layer,
            )
            crossed = bool(getattr(emotion_result, "session_boundary_crossed", False))
        else:
            snapshot = build_persona_snapshot(
                user_id=user_id,
                previous=prev,
                left_memories=left_memories,
                right_brain=right_brain,
            )
            crossed = False

        return self.update_from_snapshot(
            snapshot,
            force=force,
            session_boundary_crossed=crossed,
        )


def _as_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(x).strip() for x in value if str(x).strip()]
