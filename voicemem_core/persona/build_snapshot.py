"""从已有左右脑组件组装 Persona 合成输入。"""

from __future__ import annotations

from typing import Any

from voicemem_core.emotion.persona_input import RightBrainPersonaInput, build_right_brain_persona_input
from voicemem_core.emotion.types import SessionHolisticSummary
from voicemem_core.persona._time import parse_iso
from voicemem_core.persona.types import PersonaDocument, PersonaUpdateSnapshot


def build_persona_snapshot(
    *,
    user_id: str,
    previous: PersonaDocument,
    left_memories: list[dict[str, Any]],
    right_brain: RightBrainPersonaInput | None = None,
    new_session_holistics: list[SessionHolisticSummary] | None = None,
    archival_holistics: list[SessionHolisticSummary] | None = None,
) -> PersonaUpdateSnapshot:
    since = parse_iso(previous.updated_at)
    all_norm = [_normalize_memory_row(r) for r in left_memories]
    all_norm = [r for r in all_norm if r is not None]

    new_rows: list[dict[str, str]] = []
    for r in all_norm:
        created = parse_iso(r.get("created_at"))
        if since is None or created is None or created >= since:
            new_rows.append(r)

    anchors = all_norm[-15:]

    rb = right_brain
    if rb is None:
        rb = RightBrainPersonaInput(
            new_session_holistics=list(new_session_holistics or []),
            archival_holistics=list(archival_holistics or []),
        )

    new_holistic_texts = _holistic_texts(rb.new_session_holistics)
    archival_texts = _holistic_texts(rb.archival_holistics)[-2:]

    return PersonaUpdateSnapshot(
        user_id=user_id,
        stats={
            "memory_count": len(all_norm),
            "new_memory_count": len(new_rows),
            "new_boundary_holistic_count": len(new_holistic_texts),
            "archival_holistic_count": len(archival_texts),
        },
        anchors=[{"id": r["id"], "memory": r["memory"]} for r in anchors],
        new_memories=[{"id": r["id"], "memory": r["memory"]} for r in new_rows[-30:]],
        new_boundary_holistics=new_holistic_texts,
        archival_holistics=archival_texts,
        previous_impression=previous.impression_text,
        persona_version=previous.version,
    )


def build_persona_snapshot_from_emotion_result(
    *,
    user_id: str,
    previous: PersonaDocument,
    left_memories: list[dict[str, Any]],
    emotion_result: Any,
    emotion_layer: Any | None = None,
) -> PersonaUpdateSnapshot:
    archival = emotion_layer.archival_holistics if emotion_layer is not None else []
    rb = build_right_brain_persona_input(emotion_result, archival_holistics=archival)
    return build_persona_snapshot(
        user_id=user_id,
        previous=previous,
        left_memories=left_memories,
        right_brain=rb,
    )


def _holistic_texts(items: list[SessionHolisticSummary]) -> list[str]:
    return [h.summary_text for h in items if h.summary_text]


def _normalize_memory_row(raw: dict[str, Any]) -> dict[str, str] | None:
    mid = str(raw.get("id", "")).strip()
    memory = str(raw.get("memory") or raw.get("text") or "").strip()
    if not mid or not memory:
        return None
    created = raw.get("created_at")
    return {
        "id": mid,
        "memory": memory,
        "created_at": str(created) if created else "",
    }
