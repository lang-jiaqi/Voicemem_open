"""右脑组件 RightBrain。

从 VoiceMem 上帝类里抽出来的**右脑那一整块**——heartnote 情感记忆写入、
内心 OS 生成、右脑图层（情绪/关系/性格特质）写入、右脑检索（情境指导 rb_directive
的结构化 top-N），以及围绕它们的 LLM 清洁（重复删除 / 矛盾 supersede）。

参考 mem0 的组合模式：
  * **组件自持零件**——3 个右脑侧懒加载单例（rb_repo / rb_graph_store /
    attribution_manager）连同它们与宿主共享的缓存和锁，都在这个组件内部，
    engine 不再持有各自的 _get_*。
  * **依赖显式注入**——凡是需要用到"文本 embedding / LLM(JSON) / LLM(text) /
    会话追踪器 / 左脑仓库 / 内心OS 生成 / 特质抽取"这些**非右脑本域**能力的地方，
    一律在 __init__ 里以 getter/函数引用注入（懒加载语义保持不变），组件内部
    通过 self._dep() 调用。

logic 一字不改：方法体原样搬运，只改"怎么拿依赖"。

brain.py 不 import engine（避免循环）——RightBrainHit 数据类与模块级 _rb_*
辅助函数都落在本模块，engine 反向从这里 import。
"""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


# ── 结果容器 ───────────────────────────────────────────────────────────────────

@dataclass
class RightBrainHit:
    """右脑检索的单条结构化结果。rb_directive 由这个结构化列表渲染而来。"""
    content: str
    source: str                          # response_experience | situation_pattern | relation | emotion_trait | profile
    priority: float
    metadata: dict = field(default_factory=dict)


# ── 语言判定辅助 ───────────────────────────────────────────────────────────────

def _is_en_text(text: str) -> bool:
    """True if text is predominantly English (low CJK ratio)."""
    cjk = sum(1 for c in text if "一" <= c <= "鿿")
    alpha = sum(1 for c in text if c.isalpha())
    return alpha > 0 and cjk / max(alpha, 1) < 0.3


def _rb_lang(rb_ctx) -> bool:
    """Return True if right brain content is predominantly English."""
    samples = [m.content for m in rb_ctx.situation_patterns[:2]]
    samples += [m.content for m in rb_ctx.response_experiences[:1]]
    return _is_en_text(" ".join(samples))


def _rb_mem_date(m) -> str:
    """created_at ISO → '[YYYY-MM-DD] ' 前缀；没有日期返回空串（供时序推理用）。"""
    d = (getattr(m, "created_at", "") or "")[:10]
    return f"[{d}] " if d else ""


def _rb_blended_priority(m) -> float:
    """静态 priority + 本次检索的锚点相关度（归一化后加权）。

    anchor_score = SUM(link.weight*confidence)，用 s/(1+s) 压到 [0,1) 再乘 0.5：
    强命中锚点的具体证据能与 relation/emotion_trait 竞争，没命中的保持原状。"""
    s = getattr(m, "anchor_score", 0.0) or 0.0
    return m.priority + 0.5 * (s / (1.0 + s))


def _rb_ctx_to_hits(rb_ctx) -> list["RightBrainHit"]:
    """rb_ctx（heartnote / response_experience 检索结果）→ 结构化 hit 列表。
    当前信号（不满意/纠正/情绪提示）是本轮实时状态，给个偏高的固定优先级。"""
    en = _rb_lang(rb_ctx)
    hits: list[RightBrainHit] = []
    for m in rb_ctx.response_experiences:
        failed = (m.metadata or {}).get("previous_failure", False)
        prefix = (
            ("⚠ " + ("Avoid repeating: " if en else "避免重复："))
            if failed else
            ("✓ " + ("Effective approach: " if en else "有效方式："))
        )
        hits.append(RightBrainHit(
            content=f"{_rb_mem_date(m)}{prefix}{m.content}", source="response_experience",
            priority=_rb_blended_priority(m),
            metadata={"failed": failed, "anchor_score": getattr(m, "anchor_score", 0.0)},
        ))
    for m in rb_ctx.situation_patterns:
        meta = m.metadata or {}
        prefix = "Emotional note: " if en else "情感记录："
        inner = str(meta.get("inner_os") or "").strip()
        # content 存原话，inner_os 作为补充渲染拼在后面。超长原话（>400）会挤爆
        # prompt，降级用内心 OS 摘要，没有就截断。
        if len(m.content) > 400:
            body = inner if inner else (m.content[:400] + "…")
        else:
            body = m.content
            if inner and inner != m.content:
                body += f" (inner note: {inner})" if en else f"（内心OS：{inner}）"
        content = f"{_rb_mem_date(m)}{prefix}{body}"
        priority = _rb_blended_priority(m)
        # 被后续记录取代的旧况：保留但标注 + 降权，避免模型把旧况当现状。
        if meta.get("superseded_by"):
            until = str(meta.get("superseded_at") or "")[:10]
            tag = (
                f" [outdated{f', changed around {until}' if until else ''} — see newer note]"
                if en else
                f"【旧况{f'，约 {until} 已变化' if until else ''}，以更新的记录为准】"
            )
            content += tag
            priority *= 0.75
        hits.append(RightBrainHit(
            content=content, source="situation_pattern", priority=priority,
            metadata={"anchor_score": getattr(m, "anchor_score", 0.0)},
        ))
    sigs = rb_ctx.current_signals
    now: list[str] = []
    if sigs.dissatisfaction_signal:
        now.append("user is dissatisfied; get to the point" if en else "用户不满意，直接说重点")
    if sigs.correction_signal:
        now.append("user is correcting; just accept it" if en else "用户在纠正，接受即可")
    if sigs.affect_hint:
        now.append(f"current emotion={sigs.affect_hint}" if en else f"当前情绪={sigs.affect_hint}")
    if now:
        sep = "; " if en else "；"
        head = "Current signals: " if en else "当前信号："
        hits.append(RightBrainHit(
            content=head + sep.join(now), source="current_signal", priority=0.95,
        ))
    return hits


def _rb_graph_hits(rb_graph, user_id: str) -> list["RightBrainHit"]:
    """右脑图(情绪/喜好与厌恶/表达风格/思维模式/应对方式)的 slot description，
    每个有描述的 slot 各是一条画像观察。"""
    return [
        RightBrainHit(
            content=f"{slot.name}：{slot.description}",
            source="profile", priority=0.5, metadata={"slot_name": slot.name},
        )
        for slot in rb_graph.list_slots(user_id)
        if slot.description
    ]


def _rb_emotion_trait_hit(rb_graph, user_id: str, emotion: str | None) -> "RightBrainHit | None":
    """检索和当前情绪相近的用户固有性格节点：情绪是8个固定规范标签，按当前
    情绪精确查"情绪"slot 下同名的那一个 entity，不用向量、不扫描其余7个。"""
    if not emotion:
        return None
    from voicemem.rightbrain.anchor_router import normalize_emotion_strict
    emo_slot = rb_graph.get_slot_by_name(user_id, "情绪")
    if emo_slot is None:
        return None
    canonical = normalize_emotion_strict(emotion)
    if canonical is None:
        return None
    ent = rb_graph.get_entity_by_name(user_id, emo_slot.id, canonical)
    if ent is None or not ent.description:
        return None
    prefix = (
        "Trait observed around this emotion: " if _is_en_text(ent.description)
        else "当前情绪相关的性格观察："
    )
    return RightBrainHit(
        content=f"{prefix}{ent.description}", source="emotion_trait", priority=0.75,
    )


def _rb_relation_hits(rb_graph, user_id: str, anchors) -> list["RightBrainHit"]:
    """关系节点检索：左脑这次问题触发的实体（anchors 里带真实 entity.id 的
    锚点），直接按 ID 查对应的右脑关系节点——纯索引查表，不扫描、不算向量。"""
    from voicemem.rightbrain.anchor_router import _ENTITY_TYPE_TO_ANCHOR
    entity_anchor_types = set(_ENTITY_TYPE_TO_ANCHOR.values())
    rel_slot = rb_graph.get_slot_by_name(user_id, "人物地点态度")
    if rel_slot is None:
        return []
    hits: list[RightBrainHit] = []
    seen: set[str] = set()
    for a in anchors:
        if a.anchor_type not in entity_anchor_types or not a.anchor_id:
            continue
        if a.anchor_id in seen:
            continue
        seen.add(a.anchor_id)
        ent = rb_graph.get_entity_by_source_id(user_id, rel_slot.id, a.anchor_id)
        if ent is not None and ent.description:
            content = (
                f"Impression of {ent.name}: {ent.description}"
                if _is_en_text(ent.description)
                else f"对 {ent.name} 的印象：{ent.description}"
            )
            hits.append(RightBrainHit(
                content=content,
                source="relation", priority=0.85, metadata={"entity_name": ent.name},
            ))
    return hits


def _render_rb_directive(hits: list["RightBrainHit"]) -> str:
    """结构化 top-N → 拼进 prompt 的文本块。"""
    return "\n".join(h.content for h in hits) if hits else ""


# ── RightBrain 组件 ────────────────────────────────────────────────────────────

class RightBrain:
    """自持零件 + 依赖显式注入的右脑组件。

    构造参数分两类：

    组件自持的**运行参数**（右脑侧路径/身份）::

        memory_root, user_id, base_url, cognitive_db

    显式**注入的非右脑依赖**（全部以 getter/函数引用传入，懒加载语义不变）::

        embed              -> self._embed_text          文本 embedding（特质/图层写入）
        llm_json           -> self._llm_json            LLM(JSON)（特质抽取）
        llm_text           -> self._llm_text            LLM(text)（归因管理器）
        tracker            -> self._get_session_tracker 跨左右脑会话追踪器（touch）
        repo               -> self._get_repo            左脑仓库（写入时查左脑实体链接）
        generate_inner_os  -> self._generate_inner_os   内心OS 生成（延迟解析，可被测试 patch）
        extract_rb_traits  -> self._extract_rb_traits   特质抽取（延迟解析，可被测试 patch）

    右脑侧的 3 个懒加载单例（rb_repo / rb_graph_store / attribution_manager）
    连同与宿主共享的缓存/锁由本组件自持，见下方 self._rb_repo() 等。
    """

    def __init__(
        self,
        *,
        memory_root: Path,
        user_id: str,
        base_url: str | None,
        cognitive_db: Path,
        embed: Callable[[str], list[float]],
        llm_json: Callable[[str], str],
        llm_text: Callable[..., str],
        tracker: Callable[[], Any],
        repo: Callable[[], Any],
        generate_inner_os: Callable[..., str],
        extract_rb_traits: Callable[..., list[tuple[str, str]]],
        cache: dict[str, Any] | None = None,
        lock: Any = None,
    ) -> None:
        # ── 运行参数 ──
        self._memory_root = memory_root
        self._user_id = user_id
        self._base_url = base_url
        self._cognitive_db = cognitive_db

        # ── 注入的非右脑依赖（getter/函数引用）──
        self._embed = embed
        self._llm_json = llm_json
        self._llm_text = llm_text
        self._tracker = tracker
        self._repo = repo
        # generate_inner_os / extract_rb_traits 延迟解析（getter 形式），
        # 让既有测试对宿主实例的 patch.object 生效（write 走宿主暴露的入口）。
        self._generate_inner_os = generate_inner_os
        self._extract_rb_traits = extract_rb_traits

        # ── 组件自持的右脑零件缓存 ──
        # 允许宿主共享同一个 cache/lock（右脑侧懒加载单例与宿主 _get_* 落在同一
        # 字典，既存调用点/测试对宿主 _cache 的直接读写与本组件保持一致视图）。
        self._cache: dict[str, Any] = cache if cache is not None else {}
        self._lock = lock if lock is not None else threading.Lock()

    # ── 右脑懒加载单例 ──────────────────────────────────────────────────────────

    def _rb_repo(self):
        with self._lock:
            if "rb_repo" not in self._cache:
                from voicemem.leftbrain.cognitive_graph import CognitiveGraphStore
                from voicemem.rightbrain import ExperienceRepository
                cog_store = CognitiveGraphStore(self._cognitive_db)
                self._cache["rb_repo"] = ExperienceRepository.create(
                    self._memory_root / "right_brain.sqlite",
                    cognitive_store=cog_store,
                )
        return self._cache["rb_repo"]

    def _rb_graph_store(self):
        with self._lock:
            if "rb_graph_store" not in self._cache:
                from voicemem.rightbrain import RightBrainGraphStore
                store = RightBrainGraphStore(self._memory_root / "rb_graph.sqlite")
                store.ensure_seed_slots(self._user_id)
                self._cache["rb_graph_store"] = store
        return self._cache["rb_graph_store"]

    def _attribution_manager(self):
        rb_graph = self._rb_graph_store()
        rb_repo = self._rb_repo()
        with self._lock:
            if "attribution_manager" not in self._cache:
                from voicemem.rightbrain import AttributionManager
                self._cache["attribution_manager"] = AttributionManager(
                    rb_graph, rb_repo._store, llm_fn=self._llm_text,
                )
        return self._cache["attribution_manager"]

    # ── 右脑检索 ────────────────────────────────────────────────────────────────

    def search(
        self, query: str, activated_names: list[str], emotion: str | None, top_k: int,
    ) -> tuple[list["RightBrainHit"], str]:
        """右脑并发检索段（原 Search() 里的 _run_rb 闭包）：build_query_plan →
        retrieve → 关系/情绪特质/画像图层 → 按 priority 排序截断，返回
        (rb_hits, rb_directive)。异常时降级为无右脑（空列表 + 空指导）。"""
        try:
            from voicemem.rightbrain.types import CurrentSignals
            rb_repo = self._rb_repo()
            # 右脑接收左脑"已激活实体"作锚点（联合检索：右脑依赖左脑激活结果）
            plan    = rb_repo.build_query_plan(
                query, self._user_id,
                signals=CurrentSignals(),
                entities=activated_names or None,
                emotion=emotion,
            )
            rb_ctx = rb_repo.retrieve(plan)
            collected: list[RightBrainHit] = _rb_ctx_to_hits(rb_ctx) if not rb_ctx.is_empty() else []

            rb_graph = self._rb_graph_store()
            collected.extend(_rb_relation_hits(rb_graph, self._user_id, plan.anchors))
            trait_hit = _rb_emotion_trait_hit(rb_graph, self._user_id, emotion)
            if trait_hit is not None:
                collected.append(trait_hit)
            collected.extend(_rb_graph_hits(rb_graph, self._user_id))

            # 按 priority 排序截断，rb_directive 从截断后的列表渲染，保证一致。
            collected.sort(key=lambda h: h.priority, reverse=True)
            # 右脑结构化 top-N（默认 5，VOICEMEM_RB_TOPN 可调）。
            try:
                _rb_topn = max(1, int(os.environ.get("VOICEMEM_RB_TOPN", "5")))
            except ValueError:
                _rb_topn = 5
            rb_hits = collected[:_rb_topn]
            return rb_hits, _render_rb_directive(rb_hits)
        except Exception as e:
            import traceback as _tb
            print(f"[Search] 右脑检索失败（本轮降级为无右脑）: {e}\n{_tb.format_exc()}", flush=True)
            return [], ""

    # ── 右脑写入 ────────────────────────────────────────────────────────────────

    def write(self, emotion, result, text, entities, observed_at) -> None:
        """右脑写入段：每条 utterance 一条 heartnote，挂 emotion + entity anchors +
        关系节点 + 右脑 slot→entity 图层。gate 只看 emotion（不绑 result.memory_ids）——
        纯情绪句左脑可能抽不出事实但情绪仍值得记；mid 为空时不挂证据、不查左脑实体
        链接，但情绪锚点 + 文本实体名锚点仍正常写。"""
        if not emotion:
            return
        try:
            from voicemem.rightbrain.types import MemoryAnchor
            rb_repo = self._rb_repo()
            mid = result.memory_ids[0] if result.memory_ids else None

            # content 存原话，inner_os 进 metadata（渲染时作为补充拼在原话
            # 后面，见 _rb_ctx_to_hits）——避免共情改写抹掉数字/名字/时间等细节。
            inner_os = self._generate_inner_os(text, emotion, entities or [])
            content = text

            # 事件时间用 observed_at（与左脑 time_start 同源），不用写入墙钟
            _obs = str(observed_at) if observed_at and re.match(r"^\d{4}-\d{2}-\d{2}", str(observed_at)) else None
            rb_mem = rb_repo._store.upsert_memory(
                user_id=self._user_id,
                memory_class="heartnote",
                content=content,
                metadata={"emotion": emotion, "entities": entities or [],
                          "left_memory_id": mid, "inner_os": inner_os or ""},
                evidence_memory_ids=[mid] if mid else [],
                created_at=_obs,
            )
            # emotion anchor：按情感检索。strict 版，识别不出的情绪词不挂锚点。
            from voicemem.rightbrain.anchor_router import normalize_emotion_strict
            canonical_emotion = normalize_emotion_strict(emotion)
            if canonical_emotion is not None:
                rb_repo._store.link_anchor(
                    rb_mem.id, self._user_id,
                    MemoryAnchor(anchor_type="emotion", anchor_id=canonical_emotion,
                                 role="trigger", weight=1.0, confidence=1.0),
                )
            # entity anchors：优先用左脑这条记忆真正链上的 entity.id（稳定），
            # name 字符串锚点做兜底。同一循环里顺手给每个实体建/更新一个专属
            # 关系节点（source_entity_id 精确匹配），这里只负责挂证据 + touch，
            # description 的提炼交给 AttributionManager.run_short_term。
            try:
                from voicemem.rightbrain.anchor_router import _ENTITY_TYPE_TO_ANCHOR
                cog_store = self._repo()._cognitive_store
                if mid and cog_store is not None:
                    rb_graph = self._rb_graph_store()
                    tracker  = self._tracker()
                    rel_slot = rb_graph.get_or_create_slot(self._user_id, "人物地点态度")
                    for eid in cog_store.entity_ids_for_memory(mid):
                        ent = cog_store.get_entity(eid)
                        if ent is None:
                            continue
                        rb_repo._store.link_anchor(
                            rb_mem.id, self._user_id,
                            MemoryAnchor(
                                anchor_type=_ENTITY_TYPE_TO_ANCHOR.get(ent.entity_type.value, "knowledge"),
                                anchor_id=ent.id, role="subject",
                                weight=1.0, confidence=ent.confidence,
                            ),
                        )
                        rel_ent, _created = rb_graph.get_or_create_entity_by_source_id(
                            self._user_id, rel_slot.id, ent.id, ent.name,
                        )
                        rb_graph.link_memory(rel_ent.id, self._user_id, rb_mem.id)
                        tracker.touch(self._user_id, "rb_entity_short", rel_ent.id)
            except Exception as e:
                print(f"[RBAnchor] 实体ID锚点/关系节点写入失败: {e}")

            for name in (entities or []):
                rb_repo._store.link_anchor(
                    rb_mem.id, self._user_id,
                    MemoryAnchor(anchor_type="entity", anchor_id=name.lower().strip(),
                                 role="subject", weight=0.8, confidence=1.0),
                )

            # 右脑 slot→entity 图层：情绪(精确匹配8个固定标签) + 其余4类(语义匹配)
            try:
                rb_graph = self._rb_graph_store()
                tracker = self._tracker()
                emo_slot = rb_graph.get_slot_by_name(self._user_id, "情绪")
                if emo_slot is not None and canonical_emotion is not None:
                    emo_ent = rb_graph.get_or_create_entity(
                        self._user_id, emo_slot.id, canonical_emotion,
                    )
                    rb_graph.link_memory(emo_ent.id, self._user_id, rb_mem.id)
                    tracker.touch(self._user_id, "rb_entity_short", emo_ent.id)
                    tracker.touch(self._user_id, "rb_slot_long", emo_slot.id)

                for slot_name, label in self._extract_rb_traits(text, emotion):
                    slot = rb_graph.get_slot_by_name(self._user_id, slot_name)
                    if slot is None:
                        continue
                    label_emb = self._embed(label)
                    g_ent, _created = rb_graph.get_or_create_entity_semantic(
                        self._user_id, slot.id, label, label_emb,
                    )
                    rb_graph.link_memory(g_ent.id, self._user_id, rb_mem.id)
                    tracker.touch(self._user_id, "rb_entity_short", g_ent.id)
                    tracker.touch(self._user_id, "rb_slot_long", slot.id)
            except Exception as e:
                print(f"[RBGraph] 右脑图层写入失败: {e}")
        except Exception as e:
            print(f"[Ingest] right brain write skipped: {e}")

    # ── 右脑清洁 ────────────────────────────────────────────────────────────────

    def check_and_cleanup(self) -> None:
        """每增加 50 条 heartnote 触发一次右脑清洁。"""
        try:
            import json as _json
            state_path = self._memory_root / "cleanup_state.json"
            state = _json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"last_count": 0}

            rb_repo = self._rb_repo()
            all_mems = rb_repo._store.get_all(self._user_id)
            current_count = sum(1 for m in all_mems if m.memory_class == "heartnote")

            if current_count - state.get("last_count", 0) >= 50:
                state_path.write_text(
                    _json.dumps({"last_count": current_count}), encoding="utf-8"
                )
                self.run_cleanup()
        except Exception as e:
            print(f"[Cleanup] check error: {e}")

    def run_cleanup(self) -> None:
        """用 LLM 清洁右脑 heartnote：重复/无意义 → 删除；矛盾 → 标注 supersede。

        矛盾不"删旧留新"（偏好演化题需要新旧两条 + 先后关系）：旧条目打
        superseded_by/superseded_at 标记保留，渲染时标注"旧况"并降权。"""
        try:
            import json as _json
            import sqlite3

            rb_repo = self._rb_repo()
            heartnotes = [
                m for m in rb_repo._store.get_all(self._user_id)
                if m.memory_class == "heartnote"
            ]
            if len(heartnotes) < 10:
                return

            # 构造紧凑列表发给 LLM（用前8位 ID 节省 token）
            lines = []
            id_map: dict[str, str] = {}  # short_id -> full_id
            for i, m in enumerate(heartnotes):
                short = m.id[:8]
                id_map[short] = m.id
                emotion  = (m.metadata or {}).get("emotion", "")
                entities = (m.metadata or {}).get("entities", [])
                lines.append(
                    f"[{i}] ID:{short} | 情感:{emotion} | 实体:{','.join(entities)} | {m.content}"
                )

            from openai import OpenAI
            client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
                base_url=self._base_url,
                timeout=60.0,
            )
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": (
                        "你是记忆清洁助手。分析以下情感记忆列表，做两类判断。\n"
                        "一、删除（宁可少删，不要误删有价值的记录）：\n"
                        "1. 重复：内容高度相似，保留一条，删其余；\n"
                        "2. 无意义：信息量极低（如纯标点、单词、句子残缺）。\n"
                        "二、取代（不删除）：同一实体/同一偏好有前后矛盾的描述，"
                        "序号靠后的是新状态——旧的不删，标记为被新的取代，"
                        "以保留偏好演化轨迹。\n"
                        "返回 JSON：{\"delete_ids\": [\"8位ID\", ...], "
                        "\"supersede\": [{\"old_id\": \"8位ID\", \"new_id\": \"8位ID\"}, ...]}\n"
                        "若无需处理返回 {\"delete_ids\": [], \"supersede\": []}"
                    )},
                    {"role": "user", "content": "\n".join(lines)},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )

            result      = _json.loads(resp.choices[0].message.content)
            short_ids   = result.get("delete_ids", [])
            full_ids    = [id_map[s] for s in short_ids if s in id_map]

            if full_ids:
                with sqlite3.connect(rb_repo._store._path) as conn:
                    for mid in full_ids:
                        conn.execute(
                            "DELETE FROM right_brain_anchor_links WHERE right_memory_id=?", (mid,)
                        )
                        conn.execute(
                            "DELETE FROM right_brain_memories WHERE id=?", (mid,)
                        )
                print(f"[Cleanup] 清洁完成，删除 {len(full_ids)} 条右脑记忆")
            else:
                print("[Cleanup] 无需删除")

            # 矛盾对：旧条目打 superseded 标记（保留，渲染时标注+降权）。
            pairs = result.get("supersede", []) or []
            keep_old = True
            marked = 0
            from datetime import datetime, timezone
            now_iso = datetime.now(timezone.utc).isoformat()
            for p in pairs:
                old_full = id_map.get(str(p.get("old_id", "")))
                new_full = id_map.get(str(p.get("new_id", "")))
                if not old_full or not new_full or old_full == new_full:
                    continue
                if old_full in full_ids or new_full in full_ids:
                    continue  # 已被删除的不再标记
                if keep_old:
                    rb_repo._store.merge_metadata(
                        old_full, {"superseded_by": new_full, "superseded_at": now_iso},
                    )
                else:
                    with sqlite3.connect(rb_repo._store._path) as conn:
                        conn.execute(
                            "DELETE FROM right_brain_anchor_links WHERE right_memory_id=?",
                            (old_full,),
                        )
                        conn.execute(
                            "DELETE FROM right_brain_memories WHERE id=?", (old_full,)
                        )
                marked += 1
            if marked:
                action = "标记为 superseded（保留演化轨迹）" if keep_old else "删除（旧行为）"
                print(f"[Cleanup] {marked} 条矛盾旧况已{action}")

            # 更新 last_count
            remaining = sum(
                1 for m in rb_repo._store.get_all(self._user_id)
                if m.memory_class == "heartnote"
            )
            state_path = self._memory_root / "cleanup_state.json"
            state_path.write_text(
                _json.dumps({"last_count": remaining}), encoding="utf-8"
            )

        except Exception as e:
            print(f"[Cleanup] run error: {e}")


__all__ = ["RightBrain", "RightBrainHit"]
