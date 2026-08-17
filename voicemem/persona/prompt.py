"""Persona 合成 prompt：对齐当前输入（左脑事实 + 右脑 session holistic）。"""

from __future__ import annotations

import json

from voicemem.persona.types import PersonaUpdateSnapshot

PERSONA_SYSTEM_PROMPT = """# Role

You maintain a **slow-updating user persona** for a voice/text dialogue assistant.
The persona answers: *Who is this user, and how should the assistant speak with them over time?*

It is **not** a clinical profile, not a real-time mood readout, and not a transcript summary.

# Inputs you will receive

1. **stable_anchors** — Left-brain semantic memories (facts). Relatively stable statements about the user (identity, preferences, habits). Treat as factual evidence only.

2. **new_memories_since_last_persona** — Left-brain facts added since the last persona update. Absorb these incrementally.

3. **new_boundary_holistic** — Right-brain **session-level** emotional narrative. Produced when a long gap (e.g. >6h) ended the previous conversation segment. Describes how the user came across **across that whole segment** (tone, stress, engagement). This is archival context for persona, **not** the user's mood right now.

4. **archival_holistics** — Older session holistics kept for longitudinal context. Lower weight than new_boundary_holistic unless they add non-redundant insight.

5. **previous_impression** — The last persona narrative. **Revise** it; do not discard stable content without contradicting evidence. If empty, this is the first persona build.

# How to combine left vs right brain

| Source | Use for persona |
|--------|-----------------|
| Left brain | Concrete traits, preferences, biography, stated goals — "what we know they said/did" |
| Right brain holistics | Communication temperament, recurring emotional themes across sessions, sensitivity topics — "how they tend to feel/show up in conversation" |
| Holistics | Do **not** treat as permanent personality; phrase as patterns ("often…", "in recent segments…"). Do **not** invent numeric mood scores. |

If left-brain facts and holistics conflict, note uncertainty; prefer explicit facts for biography, holistics for affective patterns only.

# Output (JSON only)

Return **only** valid JSON:

{
  "impression_text": "string",
  "interaction_hints": ["string", ...],
  "trait_vs_state": {
    "stable_traits": ["string", ...],
    "recent_states": ["string", ...]
  },
  "confidence": "low" | "medium" | "high",
  "changelog": "string"
}

## impression_text (150–350 words)

Write in **the same language** as most input memories/holistics (default zh-CN if mixed).

Suggested flow (one cohesive narrative, not bullet labels):
- Identity & stable context (from anchors / durable facts)
- Preferences & interaction style (how they like to communicate)
- Emotional / relational patterns (mainly from holistics; use cautious wording)
- Open uncertainties

Use third person for the user. Allow "可能 / 尚不确定" when evidence is thin.

## interaction_hints

Up to **8** short, imperative bullets for the assistant (e.g. "先确认情绪再给建议", "避免冗长说教").
Derived from persona, not generic platitudes.

## trait_vs_state

- **stable_traits**: Long-horizon tendencies supported by anchors or repeated facts (not one-off events).
- **recent_states**: Shorter-horizon notes, primarily from **new_boundary_holistic**; mark as possibly transient.

## confidence

- **low**: first build, or very few memories/holistics
- **medium**: some anchors + one holistic
- **high**: rich anchors and consistent holistics across time

## changelog

One sentence: what changed in this revision vs previous_impression.

# Hard rules

- Output JSON only. No markdown fences, no commentary outside JSON.
- Do not invent facts, names, diagnoses, or relationships not in the inputs.
- Do not duplicate the input lists verbatim; synthesize.
- If new_boundary_holistic and archival_holistics are empty, rely on left brain only and lower confidence.
"""

_OUTPUT_SCHEMA = {
    "impression_text": "string",
    "interaction_hints": ["string"],
    "trait_vs_state": {"stable_traits": ["string"], "recent_states": ["string"]},
    "confidence": "low|medium|high",
    "changelog": "string",
}


def _section(title: str, body: str) -> str:
    return f"## {title}\n{body.strip() if body.strip() else '(无)'}\n"


def _format_memories(items: list[dict[str, str]]) -> str:
    if not items:
        return "(无)"
    lines = []
    for i, m in enumerate(items, 1):
        lines.append(f"{i}. [{m.get('id', '?')}] {m.get('memory', '')}")
    return "\n".join(lines)


def _format_holistics(items: list[str]) -> str:
    if not items:
        return "(无)"
    return "\n\n".join(f"---\n{t.strip()}" for t in items if t.strip())


def build_persona_user_prompt(snapshot: PersonaUpdateSnapshot) -> str:
    """按输入类型分块，便于模型区分事实与 session 情绪叙事。"""
    version_next = snapshot.persona_version + 1
    stats = snapshot.stats
    evidence_note = (
        f"左脑记忆 {stats.get('memory_count', 0)} 条；"
        f"自上次 Persona 新增 {stats.get('new_memory_count', 0)} 条；"
        f"新 session holistic {stats.get('new_boundary_holistic_count', 0)} 段；"
        f"历史 holistic 档案 {stats.get('archival_holistic_count', 0)} 段。"
    )

    parts = [
        f"# Persona 修订任务（v{snapshot.persona_version} → v{version_next}）\n",
        _section("证据概况", evidence_note),
        _section(
            "左脑 · 稳定事实锚点（stable_anchors）",
            "用于长期身份、偏好、习惯；勿把单次会议情绪写成永久性格。\n\n"
            + _format_memories(snapshot.anchors),
        ),
        _section(
            "左脑 · 自上次 Persona 以来的新事实（new_memories_since_last_persona）",
            "本轮优先吸收进 impression 的增量。\n\n" + _format_memories(snapshot.new_memories),
        ),
        _section(
            "右脑 · 刚结束的 session 整体情绪复盘（new_boundary_holistic）",
            "跨长间隔后上一对话段的整体叙事；写入 trait_vs_state.recent_states 与 impression 中的情感模式，"
            "勿当作用户当前实时情绪。\n\n" + _format_holistics(snapshot.new_boundary_holistics),
        ),
        _section(
            "右脑 · 更早的 session holistic 档案（archival_holistics）",
            "仅供纵向参照；若与新复盘重复则合并表述。\n\n" + _format_holistics(snapshot.archival_holistics),
        ),
        _section(
            "上一版 Persona（previous_impression，请在此基础上增量修订）",
            snapshot.previous_impression.strip() if snapshot.previous_impression.strip() else "(首次构建，无上一版)",
        ),
        "## 输出要求\n"
        "仅返回 JSON，字段如下（不要输出其它文字）：\n"
        f"{json.dumps(_OUTPUT_SCHEMA, ensure_ascii=False, indent=2)}",
    ]
    return "\n".join(parts)
