"""VoiceMem 统一入口类。

用法::

    from voicemem import VoiceMem

    vm = VoiceMem()

    # 语音模块提供 slot 和 entities，直接传入：
    result = vm.Search(query, slots=["work"], entities=["阿里"])

    # 分步调用：
    slot_ids, clf = vm.SearchCogGraph(slots=["work"], entities=["阿里"])
    candidate_ids  = vm.SearchData(slot_ids, clf)
    hits           = vm.Rank(query, candidate_ids, top_k=5)
"""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from voicemem.leftbrain.cognitive_graph.slot_v2 import SLOT_RELATIONS
from voicemem.leftbrain.cognitive_graph.query_slot_classifier import QueryClassification
from voicemem.leftbrain.local_memory_store import MemorySearchHit


# ── 结果容器 ───────────────────────────────────────────────────────────────────

@dataclass
class RightBrainHit:
    """右脑检索的单条结构化结果。rb_directive 由这个结构化列表渲染而来。"""
    content: str
    source: str                          # response_experience | situation_pattern | relation | emotion_trait | profile
    priority: float
    metadata: dict = field(default_factory=dict)


@dataclass
class SearchResult:
    """Search() 的完整返回值。"""
    hits: list[MemorySearchHit]
    classification: QueryClassification
    related_summaries: dict[str, str]   # {slot: summary_text}
    slot_mem_ids: set[str]              # SearchCogGraph 返回的原始 slot IDs
    final_candidate_ids: set[str]       # SearchData 实体缩窄后的最终候选 IDs
    search_mode: str = "fallback"
    rb_directive: str = ""              # 右脑情境指导文字（由 rb_hits 渲染而来）
    rb_hits: list[RightBrainHit] = field(default_factory=list)  # 右脑结构化 top-N
    scene_directive: str = ""          # 当前声学场景的回复风格建议
    current_scene: str = ""            # 当前场景 tag，如 "transit"
    timing: dict = None                 # {slot_filter, entity_narrow, rank, rb, total} 单位 ms


# 靠词面/时间加分"救回"的记忆最多补几条（在 top_k 之外额外给，不占语义名额）
_RESCUE_K = 3

# 候选池构造模式（VOICEMEM_POOL_MODE）：
#   union  —— slot 池 ∪ 宏观关联 slot 池 ∪ 实体池 ∪ 一跳邻居池。
#   strict —— schema routing → entity narrowing → graph expansion：实体命中时取
#             slot 池 ∩ (实体 ∪ 一跳邻居)（交集太小则退回实体池，再空才退回 slot
#             池）；宏观 slot 扩散只在 slot 池小于 _STRICT_MACRO_MIN_POOL 时才开。
_POOL_MODE_ENV = "VOICEMEM_POOL_MODE"
_STRICT_MIN_INTERSECTION = 3      # 交集至少这么多条才用交集
_STRICT_MACRO_MIN_POOL = 30       # slot 池小于这个数才做宏观 slot 扩散


def _pool_mode() -> str:
    return os.environ.get(_POOL_MODE_ENV, "union").strip().lower()

# 场景切换时用于主动检索的查询词——选取该场景最常关联的记忆主题（audiomem）
_SCENE_RECALL_QUERY: dict[str, str] = {
    "office":  "工作任务截止日期会议",
    "transit": "通勤路上待办事项",
    "home":    "家里要做的事情购物清单",
    "café":    "灵感想法头脑风暴",
    "meeting": "会议议程项目进展",
    "outdoor": "运动健康目标",
    "quiet":   "学习计划专注任务",
}

# "回放原声"意图关键词：命中任意一个就认为用户在要求听回放，整句直接拿去语义搜索。
_PLAYBACK_PATTERNS = ["回放", "播放", "放一下", "听听", "重新放", "再放一遍"]


# ── 辅助函数 ───────────────────────────────────────────────────────────────────

def _search_mode(slot_ids: set, final_ids: set) -> str:
    if not slot_ids and not final_ids:
        return "fallback"
    if final_ids and final_ids < slot_ids:
        return "entity+slot-intersection"
    if final_ids == slot_ids:
        return "slot-only"
    return "entity+slot-union"


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


@dataclass
class AudioPerception:
    """preprocess() 的结构化产物：一段音频跑完所有声学感知模块后的汇总。
    speaker/emotion 可能被感知结果覆盖（声纹 person_id、韵律情绪），其余字段
    供 Ingest 组装 ctx 与写 audiomem 标签使用。"""
    speaker: str
    emotion: str
    environment: str = ""              # AST/CLAP 背景音描述
    environment_hint: str = ""         # 即时场景提示（始终来自 AST）
    scene_tag: str | None = None       # 归类后的场景 tag，如 "transit"
    scene_raw_labels: list[str] = field(default_factory=list)
    person_id: str | None = None       # 声纹识别出的说话人 id
    tune_result: Any = None            # 背景音乐/哼唱识别结果
    abnormal_hits: list = field(default_factory=list)  # 异常环境音
    detection: dict = field(default_factory=dict)      # AST 原始一次推理结果


# ── VoiceMem 类 ────────────────────────────────────────────────────────────────

class VoiceMem:
    """Left-brain + right-brain personal memory system.

    Parameters
    ----------
    memory_root:
        Memory storage directory.  Defaults to
        ``<package_root>/results/voice_memory``.
    user_id:
        Owner of all memories managed by this instance.
    base_url:
        OpenAI-compatible API base URL (e.g. a proxy).  Falls back to the
        ``OPENAI_BASE_URL`` environment variable.
    """

    def __init__(
        self,
        memory_root: Path | str | None = None,
        user_id: str = "voice_user",
        base_url: str | None = None,
        enable_scene: bool = True,
        enable_music: bool = True,
        enable_abnormal_sound: bool = True,
        enable_voiceprint: bool = True,
        enable_emotion: bool = True,
        embedder: Any = None,
        vector_store: Any = None,
        classifier: Any = None,
    ) -> None:
        _pkg_root = Path(__file__).resolve().parent.parent
        self._vector_store = vector_store   # 注入的 memory engine（默认 None → mem0）
        self._memory_root = Path(memory_root) if memory_root else (
            _pkg_root / "results" / "voice_memory"
        )
        self._memory_root.mkdir(parents=True, exist_ok=True)
        self._cognitive_db = self._memory_root / "cognitive_graph.sqlite"
        self._user_id = user_id
        self._base_url = base_url or os.environ.get("OPENAI_BASE_URL") or None
        # Official/default is OpenAI embeddings (OpenAILocalEmbedder, built
        # lazily in _get_repo() below); pass a different TextEmbedder-
        # conforming object here to use something else for the left-brain
        # store's raw-fact embedding (used for both ingest and Rank()'s
        # search-time ranking). openai_voice_demo uses this to swap in a
        # local model for speed -- see that demo's local_embedder.py.
        self._embedder = embedder
        # query→slots+entities 的分类器（Classify 用）。默认 None → 内置
        # QuerySlotClassifier（单次 LLM）。传一个 .classify(query)->QueryClassification
        # 的实现即可切成本地模型；可选的 .classify_child(...) 存在时才做子 slot 下钻。
        self._classifier = classifier

        # 5 个音频能力开关，全部默认开；可按需精确关掉某几个。
        self._enable_scene = enable_scene
        self._enable_music = enable_music
        self._enable_abnormal_sound = enable_abnormal_sound
        self._enable_voiceprint = enable_voiceprint
        self._enable_emotion = enable_emotion

        self._cache: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._ingest_count = 0
        # session_id → person_id：session 里一旦靠自我介绍确认过是谁，后续同
        # session 的每一句都优先信这个人，而非逐句按纯声纹分数投票。
        self._session_person_pin: dict[str, str] = {}
        # person_id → session_id：这个 id 第一次被创建时所在的 session。跟
        # _session_person_pin 结合，是 _reconcile_speaker_candidates 里除声学
        # cross_score 之外的第二个独立信号：出生 session 后来被钉死成另一个人的
        # id，大概率只是那人某句噪声话的误判碎片，不该被合并进声学匹配的第三方身份。
        self._person_origin_session: dict[str, str] = {}

    # ── 懒加载单例 ──────────────────────────────────────────────────────────────

    def _get_repo(self):
        with self._lock:
            if "repo" not in self._cache:
                from voicemem.leftbrain.cognitive_graph import CognitiveAnnotator, CognitiveAnnotatorConfig
                from voicemem.leftbrain.local_memory_store import OpenAILocalEmbedder, OpenAILocalEmbedderConfig
                from voicemem.leftbrain.memory_repository_v2 import LeftBrainMemoryRepositoryConfig, LeftBrainMemoryRepositoryV2
                annotator = CognitiveAnnotator(CognitiveAnnotatorConfig(base_url=self._base_url))
                embedder  = self._embedder or OpenAILocalEmbedder(OpenAILocalEmbedderConfig(base_url=self._base_url))
                cfg = LeftBrainMemoryRepositoryConfig(
                    json_path=self._memory_root / "memories.json",
                    db_path=self._memory_root / "voicemem_leftbrain.sqlite",
                    cognitive_db_path=self._cognitive_db,
                    enable_cognitive_graph=True,
                )
                self._cache["repo"] = LeftBrainMemoryRepositoryV2(
                    embedder, config=cfg, cognitive_annotator=annotator,
                    vector_store=self._vector_store,
                )
        return self._cache["repo"]

    def _get_rb_repo(self):
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

    def _get_extractor(self):
        with self._lock:
            if "extractor" not in self._cache:
                from voicemem.leftbrain.extract_facts_openai import (
                    OpenAIAdditiveExtractorConfig,
                    OpenAIMem0V3AdditiveExtractor,
                )
                self._cache["extractor"] = OpenAIMem0V3AdditiveExtractor(
                    OpenAIAdditiveExtractorConfig(base_url=self._base_url)
                )
        return self._cache["extractor"]

    def _get_registry(self):
        with self._lock:
            if "registry" not in self._cache:
                from voicemem.utils.common.voice_input import VoiceprintRegistry
                self._cache["registry"] = VoiceprintRegistry(
                    self._memory_root / "voiceprint_registry.json"
                )
        return self._cache["registry"]

    # ── audiomem：场景 + 声纹相关懒加载单例 ─────────────────────────────────────

    def _get_env_detector(self):
        with self._lock:
            if "env_detector" not in self._cache:
                from voicemem.utils.audio.environment.environment_detector_ast import ASTEnvironmentDetector
                self._cache["env_detector"] = ASTEnvironmentDetector()
        return self._cache["env_detector"]

    def _clap_memory_enabled(self) -> bool:
        # AST always supplies the immediate hint. Once a CLAP checkpoint is
        # configured, the 4s-segmented CLAP pass takes over the background-sound
        # description memory write; set VOICEMEM_ENVIRONMENT_MEMORY_BACKEND=ast
        # to opt back out.
        return (
            os.environ.get("VOICEMEM_ENVIRONMENT_MEMORY_BACKEND", "clap").lower() == "clap"
            and bool(os.environ.get("VOICEMEM_CLAP_CHECKPOINT"))
        )

    def _get_clap_env_detector(self):
        with self._lock:
            if "clap_env_detector" not in self._cache:
                from voicemem.utils.audio.environment.environment_detector_clap import CLAPEnvironmentDetector
                self._cache["clap_env_detector"] = CLAPEnvironmentDetector(
                    checkpoint=os.environ["VOICEMEM_CLAP_CHECKPOINT"]
                )
        return self._cache["clap_env_detector"]

    def _finish_clap_environment(self, audio_path, text, session_id, environment_hint="") -> None:
        """后台用长窗口 CLAP 复核环境音，并写入独立环境记忆。

        CLAP 的候选词表是固定集合，真实环境音类别不在表里时 pairs 会是空的；
        此时退回用 AST 的 hint 兜底（CLAP 模式已把 ctx["environment"] 清空）。
        """
        try:
            detection = self._get_clap_env_detector().detect_full(Path(audio_path))
            pairs = detection.get("pairs") or []
            if not pairs:
                if environment_hint:
                    self.IngestEnv(
                        audio_path,
                        recent_context=[{"role": "user", "content": text}] if text else None,
                        session_id=session_id,
                        environment_override=environment_hint,
                    )
                return
            env_str = "background sounds: " + ", ".join(
                f"{label}({score:.2f})" for label, score in pairs
            )
            self.IngestEnv(
                audio_path,
                recent_context=[{"role": "user", "content": text}] if text else None,
                session_id=session_id,
                environment_override=env_str,
            )
        except Exception as exc:
            print(f"  [clap-env] background write skipped: {exc}", flush=True)

    def _get_trigger_store(self):
        with self._lock:
            if "trigger_store" not in self._cache:
                from voicemem.utils.audio.environment.scene_trigger import SceneTriggerStore
                self._cache["trigger_store"] = SceneTriggerStore(
                    self._memory_root / "scene_triggers.sqlite"
                )
        return self._cache["trigger_store"]

    def _get_audio_archive(self):
        with self._lock:
            if "audio_archive" not in self._cache:
                from voicemem.utils.audio.audio_archive import AudioArchive
                self._cache["audio_archive"] = AudioArchive(
                    self._memory_root / "audio_archive.sqlite"
                )
        return self._cache["audio_archive"]

    def _get_speaker_encoder(self):
        with self._lock:
            if "speaker_encoder" not in self._cache:
                from voicemem.utils.audio.voiceprint.speaker_encoder import SpeakerEncoder
                self._cache["speaker_encoder"] = SpeakerEncoder()
        return self._cache["speaker_encoder"]

    def _get_vp_store(self):
        with self._lock:
            if "vp_store" not in self._cache:
                from voicemem.utils.audio.voiceprint.voiceprint_store import VoiceprintStore
                from voicemem.utils.common.voice_config import VoiceStoreConfig
                voice_cfg = VoiceStoreConfig.from_env()
                self._cache["vp_store"] = VoiceprintStore(
                    self._memory_root / "voiceprints",
                    match_threshold=voice_cfg.match_threshold,
                    candidate_threshold=voice_cfg.candidate_threshold,
                    merge_threshold=voice_cfg.merge_threshold,
                )
        return self._cache["vp_store"]

    def _claimed_by_other_identity(self, candidate_pid: str, other_pid: str) -> bool:
        """``candidate_pid`` 的出生 session 是否已被自我介绍确认成一个跟这次拟
        合并的两个 id（``candidate_pid`` 和 ``other_pid``）都不沾边的第三方身份。

        如果是，``candidate_pid`` 大概率是那个第三方一句噪声话的误判碎片，不该
        被并进 ``other_pid``。出生 session 钉死成 ``other_pid`` 自己属于正常情形，不拦。
        """
        origin_session = self._person_origin_session.get(candidate_pid)
        if origin_session is None:
            return False
        pinned = self._session_person_pin.get(origin_session)
        return pinned is not None and pinned not in (candidate_pid, other_pid)

    def _reconcile_speaker_candidates(self, person_id: str, speaker: str) -> tuple[str, str]:
        """在一次真实 match 之后，核对是否有候选声纹现在能确认并回。

        只处理声学层面：``VoiceprintStore`` 不认姓名，判断"两边攒起来的画像
        是否真的是同一个人声"；姓名冲突守卫在这里做——已经被明确报过不同姓名
        的两个 person_id，绝不因为声学分数够高就强行合并。返回可能被更新过的
        ``(person_id, speaker)``（如果当前这条正好是被合并掉的那个）。
        """
        vp_store = self._get_vp_store()
        registry = self._get_registry()
        for _cid, absorb_pid, into_pid, cross in vp_store.find_resolvable_candidates(person_id):
            absorb_name = registry.display_name(absorb_pid)
            into_name = registry.display_name(into_pid)
            absorb_named = absorb_name != absorb_pid
            into_named = into_name != into_pid
            if absorb_named and into_named and absorb_name != into_name:
                continue
            # 声学 cross_score 之外的第二独立信号：任意一侧的出生 session 若已被
            # 自我介绍钉死给第三方，说明这个 id 另有主人，不能因声学分数够高就合并。
            if self._claimed_by_other_identity(absorb_pid, into_pid) or \
                    self._claimed_by_other_identity(into_pid, absorb_pid):
                continue
            vp_store.merge_persons(absorb_pid, into_pid)
            if absorb_named and not into_named:
                registry.bind(into_pid, name=absorb_name)
            # merge_persons() 只合并声纹画像，不碰 memory_tags；把旧
            # speaker:{absorb_pid} 标签回填成 into_pid，否则检索按人过滤会漏。
            try:
                cog_store = self._get_repo()._cognitive_store
                if cog_store and hasattr(cog_store, "rename_tag_value"):
                    cog_store.rename_tag_value(
                        self._user_id, f"speaker:{absorb_pid}", f"speaker:{into_pid}"
                    )
            except Exception as _e:
                print(f"  [speaker_identity] 标签回填失败: {_e}", flush=True)
            if person_id == absorb_pid:
                person_id = into_pid
                if speaker == absorb_pid:
                    speaker = into_pid
        return person_id, speaker

    def _get_emotion_detector(self):
        with self._lock:
            if "emotion_detector" not in self._cache:
                # 韵律 VAD + 负面显著轮次 Qwen2.5-Omni 归因；接口 detect(audio_path)->str。
                from voicemem.utils.audio.emotion.paper_emotion_detector import PaperAlignedEmotionDetector
                self._cache["emotion_detector"] = PaperAlignedEmotionDetector()
        return self._cache["emotion_detector"]

    def _get_music_store(self):
        with self._lock:
            if "music_store" not in self._cache:
                from voicemem.utils.audio.environment.music_memory import MusicMemoryStore
                self._cache["music_store"] = MusicMemoryStore(
                    self._memory_root / "music_profiles"
                )
        return self._cache["music_store"]

    def _get_routine_store(self):
        with self._lock:
            if "routine_store" not in self._cache:
                from voicemem.utils.audio.environment.routine_memory import RoutineStore
                self._cache["routine_store"] = RoutineStore(
                    self._memory_root / "routine_memory.sqlite"
                )
        return self._cache["routine_store"]

    def _get_place_store(self):
        with self._lock:
            if "place_store" not in self._cache:
                from voicemem.utils.audio.environment.place_memory import PlaceMemoryStore
                self._cache["place_store"] = PlaceMemoryStore(
                    self._memory_root / "place_profiles"
                )
        return self._cache["place_store"]

    # ── audiomem：场景触发提醒 ───────────────────────────────────────────────────

    def CreateSceneTrigger(self, text: str) -> dict:
        """从用户语句中解析场景触发意图并存储提醒。

        Parameters
        ----------
        text:
            用户语音转写文本，如"到公交上提醒我打电话给妈妈"。

        Returns
        -------
        dict
            ``{created: bool, scene: str, message: str}``
        """
        from voicemem.utils.audio.environment.scene_trigger import parse_trigger_intent
        scene, message, required_label = parse_trigger_intent(text)
        if scene is None:
            return {"created": False, "scene": "", "message": ""}

        store = self._get_trigger_store()
        trigger = store.create(self._user_id, scene.value, message, required_label=required_label)
        return {"created": True, "scene": scene.value, "message": message, "id": trigger.id}

    def GetOriginalAudio(self, memory_id: str) -> dict:
        """给定 memory_id，返回可用于"回放确认"的原始录音信息。

        原声可能已经过了保留期被清理（见 audio_archive.cleanup_expired），
        所以除了路径，还要显式校验文件是否仍然存在。

        Returns
        -------
        dict
            ``{found: bool, audio_path: str|None}``——found=False 时说明从未
            归档过，或者原声已经超过保留期被清理掉了。
        """
        path = self._get_audio_archive().get_audio_path(memory_id, self._user_id)
        if not path:
            return {"found": False, "audio_path": None}
        p = Path(path)
        if not p.exists():
            return {"found": False, "audio_path": None}
        return {"found": True, "audio_path": str(p)}

    def TryPlayback(self, text: str) -> dict | None:
        """检测这句话是不是在要求"回放原声"，命中就找最相关的记忆并取出原始录音。

        跟 GetOriginalAudio(memory_id) 不同——那个要求调用方已经知道具体是
        哪条记忆；这个是从"回放一下当时讨论价格的原声"这种自然语言里，先
        用语义搜索猜出用户说的是哪条记忆，再去查它的原声。

        Returns
        -------
        dict | None
            没有回放意图、搜不到相关记忆、或者原声已经过了保留期，返回 None；
            命中时返回 ``{"memory_id", "memory_text", "audio_path"}``。
        """
        if not any(p in text for p in _PLAYBACK_PATTERNS):
            return None
        hits = self.Rank(text, set(), top_k=3)
        for h in hits:
            audio = self.GetOriginalAudio(h.memory_id)
            if audio["found"]:
                return {
                    "memory_id": h.memory_id,
                    "memory_text": h.text,
                    "audio_path": audio["audio_path"],
                }
        return None

    # ── Dynamic slot（子图机制涌现的新 slot） ──────────────────────────────────

    def _get_dynamic_slot_store(self):
        with self._lock:
            if "dynamic_slot_store" not in self._cache:
                from voicemem.leftbrain.slot_split import DynamicSlotStore
                self._cache["dynamic_slot_store"] = DynamicSlotStore(
                    self._memory_root / "slot_splits.sqlite"
                )
        return self._cache["dynamic_slot_store"]

    def _get_dynamic_slots(self) -> list[tuple[str, str]]:
        """返回该用户已涌现的动态 slot [(name, description), ...]。"""
        try:
            return [(s.name, s.description)
                    for s in self._get_dynamic_slot_store().get_dynamic_slots(self._user_id)]
        except Exception:
            return []

    # ── slot→entity 图层（左脑：挂在 SlotV2 下；右脑：5个感性slot） ─────────────

    def _get_graph_entity_store(self):
        with self._lock:
            if "graph_entity_store" not in self._cache:
                from voicemem.leftbrain.slot_split import GraphEntityStore
                self._cache["graph_entity_store"] = GraphEntityStore(
                    self._memory_root / "graph_entities.sqlite"
                )
        return self._cache["graph_entity_store"]

    def _get_rb_graph_store(self):
        with self._lock:
            if "rb_graph_store" not in self._cache:
                from voicemem.rightbrain import RightBrainGraphStore
                store = RightBrainGraphStore(self._memory_root / "rb_graph.sqlite")
                store.ensure_seed_slots(self._user_id)
                self._cache["rb_graph_store"] = store
        return self._cache["rb_graph_store"]

    def _get_session_tracker(self):
        with self._lock:
            if "session_tracker" not in self._cache:
                from voicemem.utils.common.session_tracker import SessionTracker
                self._cache["session_tracker"] = SessionTracker(
                    self._memory_root / "session_tracker.sqlite"
                )
        return self._cache["session_tracker"]

    def _get_subgraph_manager(self):
        graph_store = self._get_graph_entity_store()   # 在 lock 外先拿，避免嵌套 acquire
        dyn_store = self._get_dynamic_slot_store()
        with self._lock:
            if "subgraph_manager" not in self._cache:
                from voicemem.leftbrain.slot_split import SubgraphManager

                def _tag_new_slot(user_id: str, memory_id: str, slot_name: str) -> None:
                    cog_store = self._get_repo()._cognitive_store
                    if cog_store is not None and hasattr(cog_store, "upsert_memory_tags"):
                        cog_store.upsert_memory_tags(memory_id, user_id, [(slot_name, 0.9)])

                self._cache["subgraph_manager"] = SubgraphManager(
                    graph_store, dyn_store, llm_fn=self._llm_json, tag_fn=_tag_new_slot,
                )
        return self._cache["subgraph_manager"]

    def _get_attribution_manager(self):
        rb_graph = self._get_rb_graph_store()
        rb_repo = self._get_rb_repo()
        with self._lock:
            if "attribution_manager" not in self._cache:
                from voicemem.rightbrain import AttributionManager
                self._cache["attribution_manager"] = AttributionManager(
                    rb_graph, rb_repo._store, llm_fn=self._llm_text,
                )
        return self._cache["attribution_manager"]

    def _extract_rb_traits(self, text: str, emotion: str) -> list[tuple[str, str]]:
        """LLM 从这句话里判断有没有透露"喜好与厌恶/表达风格/思维模式/应对方式"，
        有就提炼一个简短标签。返回 [(slot_name, label), ...]，可能是空列表。
        """
        import json as _json

        prompt = (
            f"用户说了这句话（当前情绪：{emotion or '未知'}）：\n「{text[:300]}」\n\n"
            "判断这句话有没有透露出以下几类主观信息，每类最多提炼一条简短标签（中文，5-15字）：\n"
            "- 喜好与厌恶：本能的喜欢/讨厌/偏好\n"
            "- 表达风格：说话/沟通方式和习惯\n"
            "- 思维模式：思考、判断、决策的习惯\n"
            "- 应对方式：面对压力/负面情绪时怎么自我调节\n\n"
            "没有清晰体现的类别就不要输出。\n"
            '只输出 JSON：{"items": [{"slot": "喜好与厌恶", "label": "讨厌被打断"}, ...]}'
            '（items 可以是空列表 []）'
        )
        raw = self._llm_json(prompt)
        if not raw:
            return []
        try:
            items = _json.loads(raw).get("items", [])
        except Exception:
            return []
        valid_slots = {"喜好与厌恶", "表达风格", "思维模式", "应对方式"}
        result = []
        for it in items:
            slot = str(it.get("slot", "")).strip()
            label = str(it.get("label", "")).strip()
            if slot in valid_slots and label:
                result.append((slot, label))
        return result

    def _embed_text(self, text: str) -> list[float]:
        from openai import OpenAI
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=self._base_url,
            timeout=15.0,
        )
        _kw = {
            "model": os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            "input": text,
            "encoding_format": "float",   # 部分兼容后端不支持 base64
        }
        if "openrouter" in str(self._base_url or os.environ.get("OPENAI_BASE_URL", "")).lower():
            _kw["extra_body"] = {"provider": {"order": ["OpenAI"], "allow_fallbacks": False}}
        resp = client.embeddings.create(**_kw)
        _exp = int(os.environ.get("VOICEMEM_EMBED_DIM", "1536"))
        if len(resp.data[0].embedding) != _exp:
            raise RuntimeError(f"embedding 维度 {len(resp.data[0].embedding)} != {_exp}，供应商被换掉了")
        return resp.data[0].embedding

    def _llm_json(self, prompt: str) -> str:
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
                base_url=self._base_url,
                timeout=15.0,
            )
            resp = client.chat.completions.create(
                model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=512,
            )
            from voicemem.utils.common.cost_log import log_usage
            log_usage("llm_json", resp.model, getattr(resp, "usage", None))
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"[SplitMgr] LLM 失败: {e}")
            return ""

    def _llm_text(self, prompt: str, max_tokens: int = 300) -> str:
        """跟 _llm_json 不同：不强制 JSON 输出，给归因总结这类要纯文本的场景用。"""
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
                base_url=self._base_url,
                timeout=15.0,
            )
            resp = client.chat.completions.create(
                model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=max_tokens,
            )
            from voicemem.utils.common.cost_log import log_usage
            log_usage("llm_text", resp.model, getattr(resp, "usage", None))
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"[Attribution] LLM 失败: {e}")
            return ""

    # ── Step 1: slot 过滤 ──────────────────────────────────────────────────────

    def SearchCogGraph(
        self,
        slots: list[str],
        entities: list[str] | None = None,
        scene_filter: str | None = None,
        speaker_filter: str | None = None,
    ) -> tuple[set[str], QueryClassification]:
        """slot 过滤，返回该 slot 下所有记忆 ID。

        Parameters
        ----------
        slots:
            由语音模块提供的 slot 列表，如 ``["work"]``。
        entities:
            由语音模块提供的实体列表，如 ``["阿里"]``。可为空。
        scene_filter:
            可选场景过滤（audiomem），如 ``"office"``。
        speaker_filter:
            可选说话人过滤（audiomem），传入 person_id（如 ``"person_3a2f1b"``）。
            只返回该说话人说过的记忆。

        Returns
        -------
        (slot_mem_ids, classification)
            ``slot_mem_ids`` — 候选 memory_id 集合。
            ``classification`` — 封装了 slots 和 entities 的数据容器。
        """
        classification = QueryClassification(
            slots=slots,
            entities=entities or [],
        )
        store = self._get_repo()._cognitive_store

        # 所有 slot 的记忆池取并集：Classify() 既返回精确子 slot 也返回宽父 slot。
        slot_mem_ids: set[str] = set()
        if classification.slots and store is not None and hasattr(store, "memory_ids_for_slots_v2"):
            slot_mem_ids = set(store.memory_ids_for_slots_v2(self._user_id, classification.slots))

        # 场景过滤（audiomem）：取 scene:<tag> 标签的记忆与 slot 结果的交集
        if scene_filter and slot_mem_ids:
            try:
                if store and hasattr(store, "memory_ids_for_slots_v2"):
                    scene_ids = set(
                        store.memory_ids_for_slots_v2(
                            self._user_id, [f"scene:{scene_filter}"]
                        )
                    )
                    narrowed = slot_mem_ids & scene_ids
                    if narrowed:
                        slot_mem_ids = narrowed
            except Exception:
                pass

        # 说话人过滤（audiomem）：取 speaker:<person_id> 标签与 slot 结果的交集。
        # slot_mem_ids 为空时（未指定 slot）直接把该说话人的记忆当作基础候选池。
        if speaker_filter:
            try:
                if store and hasattr(store, "memory_ids_for_slots_v2"):
                    spk_ids = set(
                        store.memory_ids_for_slots_v2(
                            self._user_id, [f"speaker:{speaker_filter}"]
                        )
                    )
                    if slot_mem_ids:
                        narrowed = slot_mem_ids & spk_ids
                        if narrowed:
                            slot_mem_ids = narrowed
                    elif spk_ids:
                        slot_mem_ids = spk_ids
            except Exception:
                pass

        return slot_mem_ids, classification

    # ── Step 2: 实体匹配（纯认知图，不碰向量） ──────────────────────────────────

    def SearchData(
        self,
        slot_mem_ids: set[str],
        classification: QueryClassification,
    ) -> set[str]:
        """在 slot_mem_ids 基础上用实体名称做交集缩窄，返回最终候选 ID 集合。

        纯认知图操作，不调用向量库，不需要原始 query。

        Parameters
        ----------
        slot_mem_ids:
            SearchCogGraph 返回的 slot 候选 ID 集合。
        classification:
            SearchCogGraph 返回的分类结果（使用其中的 entities 字段）。

        Returns
        -------
        set[str]
            最终候选 ID 集合。
            - 有实体 → slot ∪ entity
            - 无实体 → 直接返回 slot_mem_ids
        """
        final_ids, _activated_names = self._search_data_impl(slot_mem_ids, classification)
        return final_ids

    def _search_data_impl(
        self, slot_mem_ids: set[str], classification: QueryClassification,
    ) -> tuple[set[str], list[str]]:
        """SearchData() 的真正实现，多返回一个"左脑真正激活的实体名字列表"
        （含模糊匹配命中 + 一跳邻居扩散），供 Search() 内部传给右脑用。

        这跟 classification.entities（query 文本里的字面实体提及）不同——右脑
        依赖的是左脑检索管线真正确认/扩散出来的实体集合。SearchData() 公开方法
        只返回 memory id，维持原有 step-by-step 管线契约不变。
        """
        store = self._get_repo()._cognitive_store
        if not classification.entities or store is None:
            return set(slot_mem_ids), []

        entity_mids: set[str] = set()
        matched_entity_ids: set[str] = set()
        activated_names: list[str] = []
        if hasattr(store, "find_entities_by_name_fuzzy"):
            for ent_name in classification.entities:
                ents = store.find_entities_by_name_fuzzy(self._user_id, ent_name)
                if ents:
                    ids = [e.id for e in ents]
                    matched_entity_ids.update(ids)
                    activated_names.extend(e.name for e in ents)
                    mids = store.memory_ids_for_entities(ids)
                    entity_mids.update(mids)

        # 一跳邻居扩散：把直接匹配实体的一跳邻居（entity_edges）的记忆也并进候选池，
        # 一视同仁不加权，排序交给 Rank() 的向量相似度；邻居也计入 activated_names。
        if matched_entity_ids and hasattr(store, "neighbor_entity_ids"):
            neighbor_ids = store.neighbor_entity_ids(self._user_id, list(matched_entity_ids))
            if neighbor_ids:
                entity_mids.update(store.memory_ids_for_entities(neighbor_ids))
                for nid in neighbor_ids:
                    ne = store.get_entity(nid)
                    if ne:
                        activated_names.append(ne.name)

        if not entity_mids:
            return set(slot_mem_ids), activated_names

        if slot_mem_ids:
            if _pool_mode() == "strict":
                # entity narrowing：实体池对 slot 池做交集缩窄；交集太小就信实体不信 slot。
                inter = entity_mids & slot_mem_ids
                if len(inter) >= _STRICT_MIN_INTERSECTION:
                    return inter, activated_names
                return entity_mids, activated_names
            return entity_mids | slot_mem_ids, activated_names
        return entity_mids, activated_names

    # ── Step 2.5: 时间类问题扩候选 ────────────────────────────────────────────

    def _widen_for_time_question(self, query: str, final_ids: set[str]) -> set[str]:
        """问"多久 / 什么时候"时，把库里含时长或日期表达的记忆并进候选池。

        entity 和 slot 都按语义内容建索引，抓不住时间表达。这里按问题类型补一次
        正则扫库，把含时长/日期表达的记忆并进候选。final_ids 为空时走全库兜底，不扩。
        """
        if not final_ids:
            return final_ids
        from voicemem.leftbrain.local_memory_store import time_question_kind

        kind = time_question_kind(query)
        if kind is None:
            return final_ids
        store = self._get_repo()._vector_store
        if not hasattr(store, "memory_ids_with_time_expr"):
            return final_ids
        extra = store.memory_ids_with_time_expr(self._user_id, kind=kind)
        return (final_ids | extra) if extra else final_ids

    # ── Step 3: 向量排序 ──────────────────────────────────────────────────────

    def Rank(
        self,
        query: str,
        candidate_ids: set[str],
        top_k: int = 5,
        speaker_filter: str | None = None,
    ) -> list[MemorySearchHit]:
        """在 candidate_ids 范围内做向量相似度排序，返回 top-N 记忆。"""
        fetch_k = max(top_k * 3, 20)   # 全库兜底时多拉候选
        repo = self._get_repo()

        if candidate_ids:
            # 名额选择交给存储层：top_k 个按纯余弦发，额外补最多 _RESCUE_K 条被
            # 词面/时间加分救回来的（必须在完整候选集上做，避免二次截断丢分）。
            hits = repo._vector_store.search(
                query,
                user_id=self._user_id,
                top_k=top_k,
                rescue_k=_RESCUE_K,
                memory_id_filter=candidate_ids,
            )
            # 不足时从全库补齐——但按人过滤时不能这样做（会把其他人的记忆混进来），
            # 这种情况下宁可结果数少于 top_k。
            if len(hits) < top_k and not speaker_filter:
                seen = {h.memory_id for h in hits}
                for h in repo.search(query, user_id=self._user_id, top_k=fetch_k):
                    if h.memory_id not in seen:
                        hits.append(h)
                        seen.add(h.memory_id)
                        if len(hits) >= top_k:
                            break
        else:
            hits = repo.search(query, user_id=self._user_id, top_k=fetch_k)[:top_k]

        final_hits = hits[:top_k]
        # 记忆生命周期：检索命中增加热度，读取时按 last_hit_at 指数衰减、低热度归档。
        cog_store = repo._cognitive_store
        if cog_store is not None and hasattr(cog_store, "record_memory_hits"):
            try:
                cog_store.record_memory_hits([h.memory_id for h in final_hits])
            except Exception as e:
                print(f"[MemoryHeat] 记录失败: {e}")
        return final_hits

    # ── v5：LLM 打标签（替代 embedding 相似度） ───────────────────────────────

    # base-7 slot 的中文别名——用于构造 "english / 中文" 短锚点文本算 embedding。
    # 短标签对短标签的余弦相似度才够高，能让翻译变体折叠回同一个 slot。
    _BASE_SLOT_ALIASES: dict[str, str] = {
        "work": "工作", "finance": "财务", "relationships": "关系",
        "health": "健康", "goals": "目标", "daily_life": "日常生活",
        "knowledge": "知识",
    }

    def _get_slot_base_embeddings(self) -> dict[str, list[float]]:
        """base-7 slot 的 embedding，缓存一次。key 用字面枚举值（"relationships"），
        不能直接 str(枚举成员)——SlotV2.RELATIONSHIPS 的 __str__ 是 "SlotV2.RELATIONSHIPS"
        不是 "relationships"，会导致折叠命中后写回一个不存在的 slot 名字。"""
        with self._lock:
            if "slot_base_embeddings" not in self._cache:
                self._cache["slot_base_embeddings"] = {
                    value: self._embed_text(f"{value} / {alias}")
                    for value, alias in self._BASE_SLOT_ALIASES.items()
                }
        return self._cache["slot_base_embeddings"]

    def _get_slot_dyn_embeddings(self, dynamic: list[tuple[str, str]]) -> dict[str, list[float]]:
        """已涌现动态 slot 的 embedding，增量缓存（新 slot 出现才补算）。"""
        with self._lock:
            cache = self._cache.setdefault("slot_dyn_embeddings", {})
        for name, desc in dynamic:
            if name not in cache:
                cache[name] = self._embed_text(f"{name}：{desc}" if desc else name)
        return cache

    def _normalize_slot_name(
        self, candidate: str, known_all: set[str], dynamic: list[tuple[str, str]],
        threshold: float = 0.65,
    ) -> str:
        """精确匹配失败时按语义相似度把候选 slot 折叠回最接近的已知 slot（避免翻译/
        措辞漂移把同一类别拆成两份），只有真正找不到相近的才当作全新 slot。
        """
        if candidate in known_all:
            return candidate

        from voicemem.leftbrain.slot_split.split_manager import cosine_sim
        cand_emb = self._embed_text(candidate)

        best_name, best_sim = None, -1.0
        for name, emb in self._get_slot_base_embeddings().items():
            sim = cosine_sim(cand_emb, emb)
            if sim > best_sim:
                best_sim, best_name = sim, name
        for name, emb in self._get_slot_dyn_embeddings(dynamic).items():
            sim = cosine_sim(cand_emb, emb)
            if sim > best_sim:
                best_sim, best_name = sim, name

        return best_name if best_name is not None and best_sim >= threshold else candidate

    def _llm_tag_memories(self, text: str, memory_ids: list[str]) -> list[str]:
        """用 LLM 给这批记忆打 slot 标签，只能从已知 slot（固定 + 已建好的动态
        slot）里选 1-2 个，不允许 LLM 自造新类别（新 slot 只能由 SubgraphManager
        的共现子图判定产生）。返回实际打上的 slot 名称列表。
        """
        import json as _json
        from voicemem.leftbrain.cognitive_graph.slot_v2 import ALL_SLOT_V2_VALUES, SLOT_V2_DESCRIPTIONS

        dynamic = self._get_dynamic_slots()  # [(name, description), ...]
        dyn_names = {n for n, _ in dynamic}
        known_all = set(ALL_SLOT_V2_VALUES) | dyn_names

        # 构建 slot 列表描述
        slot_lines = [f"- {s}: {SLOT_V2_DESCRIPTIONS[s][:60]}" for s in ALL_SLOT_V2_VALUES]
        if dynamic:
            slot_lines += [f"- {n}: {d}" for n, d in dynamic]
        slot_desc = "\n".join(slot_lines)

        prompt = (
            f"用户说了这句话：\n「{text}」\n\n"
            f"请从下面列表里选最贴近的 1-2 个生活领域（必须选列表里已有的，"
            f"选最接近的即可，不要自创新类别）：\n{slot_desc}\n\n"
            '只输出 JSON：{"slots": ["类别1", "类别2"]}'
        )
        raw = self._llm_json(prompt)
        if not raw:
            return []

        try:
            slots = _json.loads(raw).get("slots", [])
        except Exception:
            return []

        slots = [s.strip() for s in slots if s.strip()][:2]
        if not slots:
            return []

        # 精确匹配失败的候选先按语义相似度折叠回已知 slot；折叠后仍不在已知列表
        # 里的（LLM 自造了新名字）直接丢弃——新 slot 的创造完全交给子图机制。
        slots = [self._normalize_slot_name(s, known_all, dynamic) for s in slots]
        slots = [s for s in slots if s in known_all]
        slots = list(dict.fromkeys(slots))  # 去重保序
        if not slots:
            return []

        cog_store = self._get_repo()._cognitive_store

        # 覆盖写入标签（覆盖 embedding 打的旧标签）
        if cog_store and hasattr(cog_store, "upsert_memory_tags"):
            for mid in memory_ids:
                cog_store.upsert_memory_tags(
                    mid, self._user_id, [(s, 0.95) for s in slots]
                )
        return slots

    # ── 查询分类（含动态 slot） ────────────────────────────────────────────────

    def Classify(self, query: str) -> QueryClassification:
        """LLM 分类 query → slots + entities，分层进行：
        1. 先只在 base-7 里选（不摊平全部动态 slot，避免列表越滚越长）。
        2. 每选中一个 slot，就往它的子 slot（子图机制分裂出来的）再钻一层，
           有比当前层更精确的子 slot 就往下钻，没有就停。
        3. 钻到的子 slot 追加进结果，父 slot 保留不丢——父 slot 兜住召回，
           子 slot 提供指向性，检索端对多 slot 取并集。
        entities 只在第 1 步提取一次。
        """
        from voicemem.leftbrain.cognitive_graph.query_slot_classifier import (
            QuerySlotClassifier, SlotClassifierConfig, QueryClassification,
        )
        # 可注入分类器（默认内置 LLM 版）。和 embedder 对称：传本地实现即切成
        # 本地模型，不碰 LLM/网络——这一步（抽 slot + entity）从此可 OpenAI 可本地。
        clf = self._classifier or QuerySlotClassifier(SlotClassifierConfig(base_url=self._base_url))
        top = clf.classify(query)

        dyn_store = self._get_dynamic_slot_store()
        final_slots = []

        def _add(name: str) -> None:
            if name not in final_slots:
                final_slots.append(name)

        # 子 slot 下钻需要分类器支持 classify_child（本地版没有就整体跳过）。
        _emergence_on = hasattr(clf, "classify_child")
        for slot in top.slots:
            _add(slot)
            if not _emergence_on:
                continue
            current = slot
            seen = {current}
            while True:
                children = dyn_store.get_children(self._user_id, current)
                children = [c for c in children if c.name not in seen]
                if not children:
                    break
                choice = clf.classify_child(
                    query, current, [(c.name, c.description) for c in children]
                )
                if choice is None:
                    break
                current = choice
                seen.add(current)
                _add(current)

        return QueryClassification(slots=final_slots, entities=top.entities)

    _SUBGRAPH_POOL_NS = "subgraph_pool"

    def PrimeSubgraphFromQuery(self, query: str, top_k: int = 10) -> dict:
        """Classify()+Search() 的便捷封装，返回这次检索记账的条数。

        子图判定分两层：Search() 每次检索完自动把查到的 memory_id 记进累积名单
        （便宜，无 LLM 调用，见 _record_subgraph_activation）；真正"建图→算密度→
        判断"那步很贵，只在 RunSubgraphCheckpoint() 里攒够一批后才做一次。

        记账是 Search() 自动做的副作用，直接调 Search() 效果一样；这个方法只为
        兼容"一次性 Classify+Search+拿记账条数"的调用方。
        """
        classification = self.Classify(query)
        result = self.Search(
            query=query, slots=classification.slots, entities=classification.entities,
            top_k=top_k,
        )
        return {"status": "recorded", "count": len({h.memory_id for h in result.hits})}

    def _record_subgraph_activation(self, hits: list) -> None:
        """检索结果记账：把命中 memory 对应的 graph_entity 记进 session 的子图
        候选池 + 查询激活历史（供簇涌现的密度公式用）。便宜，无 LLM 调用。
        由 Search() 本体每次真实检索后自动执行。
        """
        memory_ids = {h.memory_id for h in hits}
        if not memory_ids:
            return
        tracker = self._get_session_tracker()
        for mid in memory_ids:
            tracker.touch(self._user_id, self._SUBGRAPH_POOL_NS, mid)
        try:
            graph_store = self._get_graph_entity_store()
            activated: set[str] = set()
            for mid in memory_ids:
                activated.update(e.id for e in graph_store.get_entities_for_memory(self._user_id, mid))
            if activated:
                import uuid as _uuid
                session_id = self._get_session_tracker().get_current_session(self._user_id)
                graph_store.record_query_activation(
                    self._user_id, _uuid.uuid4().hex, list(activated), session_id=session_id,
                )
        except Exception as e:
            print(f"[QueryActivation] 记录失败: {e}")

    def RunSubgraphCheckpoint(self) -> dict:
        """把攒下的 memory_id 名单整个取出（并清空），做一次真正的建图→判断——
        这是子图判定"贵"的那一步，真实产品里应在每个 session 结束时调一次。
        """
        tracker = self._get_session_tracker()
        memory_ids = set(tracker.pop_touched(self._user_id, self._SUBGRAPH_POOL_NS))
        if not memory_ids:
            return {"status": "no_memories"}

        cog_store = self._get_repo()._cognitive_store

        def _mem_lookup(mid: str) -> str | None:
            if cog_store is None:
                return None
            rec = cog_store.get_memory_record(mid)
            return rec.content if rec else None

        session_id = self._get_session_tracker().get_current_session(self._user_id)
        return self._get_subgraph_manager().run_for_retrieved_pool(
            self._user_id, memory_ids, memory_content_lookup=_mem_lookup, session_id=session_id,
        )

    def ArchiveColdMemories(
        self, *, min_age_days: float = 30.0, heat_threshold: float | None = None,
    ) -> dict:
        """记忆生命周期的归档一步：扫衰减后热度低于阈值、且存在够久的记忆，
        调 mem0 的 expiration_date 归档（mem0 的 search()/get_all() 会自动隐藏
        过期记忆）。判定在 list_archivable_memories，这里只负责执行；显式调用
        的批处理操作，不在每次 Ingest()/Search() 里自动跑。
        """
        cog_store = self._get_repo()._cognitive_store
        if cog_store is None or not hasattr(cog_store, "list_archivable_memories"):
            return {"status": "no_cognitive_store", "archived": []}

        from voicemem.leftbrain.cognitive_graph.store import ARCHIVE_HEAT_THRESHOLD
        threshold = ARCHIVE_HEAT_THRESHOLD if heat_threshold is None else heat_threshold

        candidate_ids = cog_store.list_archivable_memories(
            self._user_id, min_age_days=min_age_days, heat_threshold=threshold,
        )
        if not candidate_ids:
            return {"status": "nothing_to_archive", "archived": []}

        vector_store = self._get_repo()._vector_store
        archived: list[str] = []
        for mid in candidate_ids:
            try:
                if hasattr(vector_store, "archive_memory") and vector_store.archive_memory(mid):
                    archived.append(mid)
            except Exception as e:
                print(f"[Archive] {mid} 归档失败: {e}")
        return {"status": "archived" if archived else "archive_failed", "archived": archived}

    # ── 完整 pipeline ──────────────────────────────────────────────────────────

    def Search(
        self,
        query: str,
        slots: list[str] | None = None,
        entities: list[str] | None = None,
        emotion: str | None = None,
        top_k: int = 5,
        scene_filter: str | None = None,
        speaker_filter: str | None = None,
    ) -> SearchResult:
        """完整检索 pipeline：SearchCogGraph → SearchData → Rank → 右脑 → 摘要。

        Parameters
        ----------
        query:
            用户语句，用于向量排序。
        slots:
            由语音模块提供的 slot 列表，如 ``["work"]``。空时降级为全库搜索。
        entities:
            由语音模块提供的实体列表，如 ``["阿里"]``。可为空。
        top_k:
            最多返回几条记忆。
        scene_filter:
            可选场景过滤（audiomem），如 ``"office"``。
        speaker_filter:
            可选说话人过滤（audiomem），传入 person_id。
        """
        import time
        import concurrent.futures

        # 情景绑定记忆：调用方没显式传 scene_filter 时，先从 query 文本反推场景意图。
        if scene_filter is None:
            from voicemem.utils.audio.environment.scene_classifier import infer_scene_from_text
            inferred_scene = infer_scene_from_text(query)
            if inferred_scene is not None:
                scene_filter = inferred_scene.value

        # query 里也没提到场景时，用当前/最近检测到的场景做软优先（narrow 不出
        # 结果会自动还原，不会真把其它场景的记忆过滤没）。
        if scene_filter is None:
            try:
                current_scene = self._get_trigger_store().get_last_scene(self._user_id)
                if current_scene:
                    scene_filter = current_scene
            except Exception:
                pass

        # ① slot 过滤
        t0 = time.time()
        slot_mem_ids, classification = self.SearchCogGraph(
            slots or [], entities, scene_filter=scene_filter, speaker_filter=speaker_filter,
        )
        t1 = time.time()

        # ② 实体缩窄——先于右脑跑完。右脑依赖左脑"已激活"的实体集合，必须等
        # _search_data_impl() 产出真正在左脑图里查到/扩散出来的那批实体。
        final_ids, activated_names = self._search_data_impl(slot_mem_ids, classification)
        final_ids = self._widen_for_time_question(query, final_ids)
        t2 = time.time()

        # ③ 右脑（依赖 activated_names）与 Rank（向量排序，依赖 final_ids）并发执行——
        # 两者互不依赖对方输出，可以并发。
        rb_hits: list[RightBrainHit] = []
        rb_directive = ""
        rb_duration  = 0.0

        def _run_rb() -> list[RightBrainHit]:
            try:
                from voicemem.rightbrain.types import CurrentSignals
                rb_repo = self._get_rb_repo()
                # 右脑接收左脑"已激活实体"作锚点（联合检索：右脑依赖左脑激活结果）
                plan    = rb_repo.build_query_plan(
                    query, self._user_id,
                    signals=CurrentSignals(),
                    entities=activated_names or None,
                    emotion=emotion,
                )
                rb_ctx = rb_repo.retrieve(plan)
                collected: list[RightBrainHit] = _rb_ctx_to_hits(rb_ctx) if not rb_ctx.is_empty() else []

                rb_graph = self._get_rb_graph_store()
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
                return collected[:_rb_topn]
            except Exception as e:
                import traceback as _tb
                print(f"[Search] 右脑检索失败（本轮降级为无右脑）: {e}\n{_tb.format_exc()}", flush=True)
                return []

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            rb_future = pool.submit(_run_rb)       # 右脑并发开跑

            hits = self.Rank(query, final_ids, top_k, speaker_filter=speaker_filter)
            t3 = time.time()

            rb_hits = rb_future.result()                 # 等右脑完成（通常已经跑完了）
            rb_directive = _render_rb_directive(rb_hits)
            t4 = time.time()
            rb_duration = t4 - t2                        # 右脑总耗时（从 activated_names 就绪开始）

        # 低置信弃权提示：左右脑都没有针对这个问题的具体证据时（左脑无命中/实体
        # 不存在，右脑只剩泛化 fallback），明确告诉 responder"证据不足就说不知道"。
        _specific_rb = {"response_experience", "situation_pattern", "relation"}
        rb_specific = any(h.source in _specific_rb for h in rb_hits)
        left_weak = (not hits) or (not activated_names)
        if left_weak and not rb_specific:
            hint = (
                "Note: the memory system found no specific evidence for this query "
                "(only generic profile context). If the retrieved content does not "
                "actually answer the question, say you don't know instead of guessing."
                if _is_en_text(query) else
                "注意：记忆系统没有为该问题找到具体证据（只有泛化画像信息）。"
                "若检索内容不能真正回答问题，请直接说不知道，不要猜测。"
            )
            rb_directive = f"{rb_directive}\n{hint}".strip()

        # 相关槽摘要：优先用从数据共现自动学出来的宏观连接；学出来的关联不够
        # （冷启动）时退回静态表兜底（动态 slot 静态表查不到，关联回其父 slot）。
        primary = classification.primary_slot()
        related_summaries: dict[str, str] = {}
        if primary:
            store = self._get_repo()._cognitive_store
            # 路由到的全部 slot + 主 slot 的 ≤3 个强连接 slot，各附一句 schema 描述。
            wanted: list[str] = list(classification.slots or [primary])
            related_slots: list[str] = []
            if store is not None and hasattr(store, "get_macro_related_slots"):
                related_slots = store.get_macro_related_slots(self._user_id, primary)
            if not related_slots:
                if primary in SLOT_RELATIONS:
                    related_slots = SLOT_RELATIONS[primary]
                else:
                    related_slots = self._get_dynamic_slot_store().get_parent_slots(self._user_id, primary)
            for r_ in related_slots[:3]:
                if r_ not in wanted:
                    wanted.append(r_)
            if wanted and store is not None and hasattr(store, "get_slot_summaries"):
                got = store.get_slot_summaries(self._user_id, wanted)
                related_summaries = {s_: got[s_] for s_ in wanted if got.get(s_)}


        # 场景自适应回复风格（audiomem）：读取用户当前场景，生成 directive
        scene_directive = ""
        current_scene = ""
        try:
            from voicemem.utils.audio.environment.scene_classifier import SceneTag, scene_to_response_directive
            last_scene = self._get_trigger_store().get_last_scene(self._user_id)
            if last_scene:
                current_scene = last_scene
                try:
                    scene_directive = scene_to_response_directive(SceneTag(last_scene))
                except ValueError:
                    pass
        except Exception:
            pass

        # 每次真实检索都自动记账（子图簇涌现需要），见 _record_subgraph_activation。
        self._record_subgraph_activation(hits)

        return SearchResult(
            hits=hits,
            classification=classification,
            related_summaries=related_summaries,
            slot_mem_ids=slot_mem_ids,
            final_candidate_ids=final_ids,
            search_mode=_search_mode(slot_mem_ids, final_ids),
            rb_directive=rb_directive,
            rb_hits=rb_hits,
            scene_directive=scene_directive,
            current_scene=current_scene,
            timing={
                "slot_filter_ms":    round((t1 - t0) * 1000, 1),
                "entity_narrow_ms":  round((t2 - t1) * 1000, 1),
                "rank_ms":           round((t3 - t2) * 1000, 1),
                "rb_ms":             round(rb_duration * 1000, 1),
                "total_ms":          round((t4 - t0) * 1000, 1),
            },
        )

    # ── 写入 ──────────────────────────────────────────────────────────────────

    def _get_user_name(self) -> str | None:
        """从左脑记忆中提取用户名字，命中后缓存。"""
        with self._lock:
            if "user_name" in self._cache:
                return self._cache["user_name"]

        import re, sqlite3 as _sql
        name: str | None = None
        try:
            db_path = self._memory_root / "voicemem_leftbrain.sqlite"
            if db_path.exists():
                conn = _sql.connect(db_path)
                rows = conn.execute(
                    "SELECT text FROM memories WHERE user_id=? LIMIT 300",
                    (self._user_id,),
                ).fetchall()
                conn.close()
                patterns = [
                    r"我叫([^\s，。！？,.]{1,6})",
                    r"叫我([^\s，。！？,.]{1,6})",
                    r"我的名字[叫是]([^\s，。！？,.]{1,6})",
                    r"[Mm]y name is ([A-Za-z]{2,15})",
                    r"[Ii]'?m ([A-Z][a-z]{1,14})",
                ]
                for (text,) in rows:
                    for pat in patterns:
                        m = re.search(pat, text)
                        if m:
                            name = m.group(1).strip()
                            break
                    if name:
                        break
        except Exception:
            pass

        with self._lock:
            self._cache["user_name"] = name
        return name

    @staticmethod
    def _is_english(text: str) -> bool:
        """True if text is predominantly English (ASCII letters dominate over CJK)."""
        cjk = sum(1 for c in text if "一" <= c <= "鿿" or "぀" <= c <= "ヿ")
        alpha = sum(1 for c in text if c.isalpha())
        return alpha > 0 and cjk / max(alpha, 1) < 0.3

    def _generate_inner_os(self, text: str, emotion: str, entities: list[str]) -> str:
        """用 LLM 把原句转成 AI 第三人称内心 OS 风格，带情绪标签；失败返回空串。"""
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
                base_url=self._base_url,
                timeout=10.0,
            )
            user_name = self._get_user_name()
            entity_hint = f", involving: {', '.join(entities)}" if entities else ""
            is_chinese = self._is_english(text) is False and any("一" <= c <= "鿿" for c in text)
            pronoun = user_name if user_name else ("用户" if is_chinese else "they")
            if is_chinese:
                system_prompt = (
                    f"你是一个有共情能力的AI助手，用第三人称记录你对用户情绪状态的内心感受。"
                    f"根据用户说的话，写出你（AI）的内心反应——就像你悄悄感受到了TA的情绪并被打动。"
                    f"要求：第三人称（称呼用户为『{pronoun}』），口语化，温暖，15-25字，"
                    f"开头用【情绪词】格式标注情绪。只输出一句话，不加任何解释。"
                    f"示例：\n"
                    f"输入：今天被老板当众批评了，好委屈\n"
                    f"输出：【心疼】{pronoun}强撑着没崩，但被这样当众说，心里一定很难受。\n"
                    f"输入：最好的朋友要搬走了\n"
                    f"输出：【担心】{pronoun}要失去身边最近的人了——以后难过的时候找谁说呢。"
                )
            else:
                system_prompt = (
                    "You are an empathetic AI assistant recording your inner observations about the user's emotional state. "
                    "Based on what the user said, write your (the AI's) internal reaction — "
                    "as if you quietly sensed their emotion and were moved by it. "
                    f"Requirements: third person (refer to the user as '{pronoun}'), "
                    "conversational, warm, 15-25 words, start with [emotion word] in brackets. "
                    f"Examples:\n"
                    f"Input: Got yelled at by my boss today, emotion: sad\n"
                    f"Output: [heartache] {pronoun} is holding it together on the outside, but being called out like that must really sting.\n"
                    f"Input: My best friend is moving away, emotion: longing\n"
                    f"Output: [worried] {pronoun} is losing someone close — once they're gone, who do they call on a hard day?\n"
                    "Output only that one sentence, nothing else."
                )
            user_content = f"What the user said: {text}\nEmotion: {emotion}{entity_hint}"

            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_content},
                ],
                max_tokens=80,
                temperature=0.7,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            return ""

    def _detect_scene(self, _apath):
        """声学场景检测（AST）：一次 detect_full() 推理拿到场景标签对、音乐/哼唱、
        异常环境音、原始 embedding；三个开关任一开就跑，后处理各自按开关判断。
        返回 (environment, environment_hint, scene_tag, scene_raw_labels,
        tune_result, abnormal_hits, detection)。"""
        environment = ""
        environment_hint = ""
        scene_tag: str | None = None
        scene_raw_labels: list[str] = []
        tune_result = None
        abnormal_hits: list[tuple[str, float]] = []
        detection = {"pairs": [], "music": None, "abnormal": [], "embedding": None}
        try:
            from voicemem.utils.audio.environment.scene_classifier import classify_scene
            detector = self._get_env_detector()
            detection = detector.detect_full(_apath)

            if self._enable_scene:
                pairs = detection["pairs"]
                if pairs:
                    parts = ", ".join(f"{l}({s:.2f})" for l, s in pairs)
                    environment = f"background sounds: {parts}"
                    environment_hint = environment
                    scene_result = classify_scene(pairs)
                    scene_tag = scene_result.tag.value
                    scene_raw_labels = [l for l, _ in scene_result.raw_matches]
        except Exception as _e:
            print(f"  [env] detection skipped: {_e}", flush=True)

        # ── 背景音乐/哼唱识别记忆（AST embedding）───────────────────
        if self._enable_music:
            try:
                music = detection.get("music")
                if music is not None:
                    tune_result = self._get_music_store().identify(
                        music["embedding"], labels=[l for l, _ in music["labels"]]
                    )
            except Exception as _e:
                print(f"  [music] detection skipped: {_e}", flush=True)

        # ── 异常环境音记忆（破碎声/警报/尖叫）──────────────────────
        if self._enable_abnormal_sound:
            try:
                abnormal_hits = detection.get("abnormal") or []
            except Exception as _e:
                print(f"  [abnormal] detection skipped: {_e}", flush=True)

        return (environment, environment_hint, scene_tag, scene_raw_labels,
                tune_result, abnormal_hits, detection)

    def _detect_speaker(self, speaker, session_id, text, _apath):
        """声纹识别（3D-Speaker ERes2Net）：算声纹向量、识别 person_id、必要时回收
        候选声纹。返回 (person_id, speaker, stable_voiceprint, vec)。"""
        person_id: str | None = None
        stable_voiceprint = False
        vec = None
        try:
            vec = self._get_speaker_encoder().embed(_apath)
            if vec is not None:
                pinned_pid = (
                    self._session_person_pin.get(session_id)
                    if session_id is not None else None
                )
                id_result = self._get_vp_store().identify(
                    vec, context=text[:80], pinned_person_id=pinned_pid,
                )
                person_id = id_result.person_id
                if id_result.action == "new" and session_id is not None:
                    self._person_origin_session.setdefault(person_id, session_id)
                # candidate 是"可能属于已有声纹"的待确认记录，不能拿来
                # 建姓名映射，否则一次误判会永久污染身份档案。
                stable_voiceprint = id_result.action in {"new", "match"}
                # caller 传入的 speaker 是默认占位符时，用识别结果覆盖
                if speaker == "Speaker 0":
                    speaker = person_id

                # ── 候选自动回收：这次 match 让画像更厚实，借机再核对
                # 一遍，收敛到同一个人的候选就并回去，避免记忆散落两个
                # person_id、speaker_filter 漏掉另一半。
                if id_result.action == "match":
                    person_id, speaker = self._reconcile_speaker_candidates(person_id, speaker)
        except Exception as _e:
            print(f"  [speaker] identification skipped: {_e}", flush=True)
        return person_id, speaker, stable_voiceprint, vec

    def _bind_self_identity(self, person_id, speaker, stable_voiceprint, vec, text, session_id):
        """自我介绍绑定（"我是annie" → 把当前声纹绑定到人名）。绑定后
        voice_input_to_messages() 会自动用真名替换 Speaker N。
        返回 (person_id, speaker, stable_voiceprint)。"""
        try:
            from voicemem.utils.audio.voiceprint.speaker_identity import parse_self_identification
            self_name = parse_self_identification(text)
            if self_name:
                registry = self._get_registry()
                existing_name = registry.display_name(person_id)
                # 自动抽取只补齐空名，不覆盖已确认人名（姓名更正应走显式接口）。
                # "未绑定"不等于"可占用"：声纹阈值对短句不稳，一个已攒了真实
                # 观测的声纹若被别人的自报姓名误匹配，会把之前那人的话张冠李戴。
                # 阈值定在 obs_count<=2（本条 + 最多1条紧邻历史），不用 <=1：
                # 自我介绍常不是第一句，用 <=1 会把"同一人紧接着报姓名"误判为拆分。
                obs_count = self._get_vp_store().get_meta(person_id).get("obs_count", 0)
                fresh_unbound = existing_name == person_id and obs_count <= 2
                if existing_name == self_name or fresh_unbound:
                    registry.bind(person_id, name=self_name)
                    if session_id is not None:
                        self._session_person_pin[session_id] = person_id
                else:
                    # 自报姓名是强证据：当前声纹已绑定成别人名、或虽未绑定
                    # 但已有历史观测（疑似误匹配）时，拆出独立声纹，再把本句
                    # 和后续记忆绑定到自报姓名。
                    if vec is not None:
                        split_result = self._get_vp_store().create_person(
                            vec, context=text
                        )
                        person_id = split_result.person_id
                        if session_id is not None:
                            self._person_origin_session.setdefault(person_id, session_id)
                        stable_voiceprint = True
                        if speaker == "Speaker 0":
                            speaker = person_id
                        registry.bind(person_id, name=self_name)
                        if session_id is not None:
                            self._session_person_pin[session_id] = person_id
                    else:
                        print(
                            f"  [speaker_identity] ignored conflicting auto-name "
                            f"{self_name!r}; {person_id} is already {existing_name!r} "
                            f"(obs_count={obs_count})",
                            flush=True,
                        )
        except Exception as _e:
            print(f"  [speaker_identity] bind failed: {_e}", flush=True)
        return person_id, speaker, stable_voiceprint

    def preprocess(
        self,
        text: str,
        speaker: str = "Speaker 0",
        emotion: str = "",
        session_id: int | str | None = None,
        audio_path: str | None = None,
    ) -> "AudioPerception":
        """流式预处理：Ingest() 推理路径的第 ① 步，一个可单独调用的公开接缝。

        把一轮音频跑过所有声学感知模块——声学场景 / 背景音乐 / 异常环境音 /
        说话人声纹 / 自我介绍绑定 / 韵律情绪，外加纯文本模式下的自我介绍绑定
        与情绪兜底，汇总成一个 AudioPerception。语音层只需给出 text（+ 可选
        audio_path），感知全在这里发生；返回后 Ingest() 负责组装 ctx 写记忆。

        独立公开是为了把"预处理"和"写记忆"解耦：可先拿到场景/说话人/情绪等信号
        用于实时应答，再决定是否 Ingest。不写记忆，但会就地更新声纹注册表等共享状态。
        """
        environment = ""
        environment_hint = ""
        scene_tag: str | None = None
        scene_raw_labels: list[str] = []
        person_id: str | None = None
        stable_voiceprint = False
        vec = None
        tune_result = None   # voicemem.utils.audio.environment.music_memory.TuneIdentifyResult | None
        abnormal_hits: list[tuple[str, float]] = []
        detection = {"pairs": [], "music": None, "abnormal": [], "embedding": None}
        if audio_path is not None:
            _apath = Path(audio_path)

            if self._enable_scene or self._enable_music or self._enable_abnormal_sound:
                (environment, environment_hint, scene_tag, scene_raw_labels,
                 tune_result, abnormal_hits, detection) = self._detect_scene(_apath)

            if self._enable_voiceprint:
                person_id, speaker, stable_voiceprint, vec = self._detect_speaker(
                    speaker, session_id, text, _apath,
                )
                if person_id and stable_voiceprint:
                    person_id, speaker, stable_voiceprint = self._bind_self_identity(
                        person_id, speaker, stable_voiceprint, vec, text, session_id,
                    )

            # ── 情绪识别（韵律 VAD + 负显著轮 Qwen2.5-Omni 归因）───────────
            # 只在调用方没显式传 emotion 时才自动检测填充。
            if self._enable_emotion and not emotion:
                try:
                    emotion = self._get_emotion_detector().detect(_apath)
                except Exception as _e:
                    print(f"  [emotion] detection skipped: {_e}", flush=True)

        # ── 自我介绍绑定，文本模式（无音频/无声纹时）──────────────────────
        # 声纹分支的绑定纯文本调用走不到。文本通道的 speaker id 是调用方给定的
        # 稳定标识，无误匹配风险，只需"只补空名、不覆盖已确认人名"，不用 obs_count 防护。
        if person_id is None and speaker and speaker != "Speaker 0":
            try:
                from voicemem.utils.audio.voiceprint.speaker_identity import parse_self_identification
                self_name = parse_self_identification(text)
                if self_name:
                    registry = self._get_registry()
                    if registry.display_name(speaker) == speaker:  # 只补空名
                        registry.bind(speaker, name=self_name)
            except Exception as _e:
                print(f"  [speaker_identity] text-mode bind failed: {_e}", flush=True)

        # ── 文本情绪兜底（无音频/韵律检测没给出结果时）─────────────────────
        # 情绪检测只挂在音频分支，纯文本调用 emotion 永远为空、右脑写不进。这里
        # 用 anchor_router 的中英情绪关键词表对 text 零成本匹配：有明确情绪词才算，
        # 识别不出返回 None、不写 heartnote。VOICEMEM_TEXT_EMOTION=0 关闭。
        if (
            self._enable_emotion and not emotion and text
            and os.environ.get("VOICEMEM_TEXT_EMOTION", "1") != "0"
        ):
            from voicemem.rightbrain.anchor_router import normalize_emotion_strict
            detected = normalize_emotion_strict(text)
            if detected:
                emotion = detected

        return AudioPerception(
            speaker=speaker,
            emotion=emotion,
            environment=environment,
            environment_hint=environment_hint,
            scene_tag=scene_tag,
            scene_raw_labels=scene_raw_labels,
            person_id=person_id,
            tune_result=tune_result,
            abnormal_hits=abnormal_hits,
            detection=detection,
        )

    def Ingest(
        self,
        text: str,
        speaker: str = "Speaker 0",
        emotion: str = "",
        entities: list[str] | None = None,
        session_id: int | str | None = None,
        audio_path: str | None = None,
        observed_at: str | None = None,
        async_facts: bool = False,
    ) -> dict:
        """将一条语音输入存入记忆库。

        这是 voicemem 与语音层结合的主入口，内部是一条三步流水线::

            ① preprocess()      text (+可选 audio_path) → AudioPerception
                                （流式预处理：场景/声纹/情绪等全部声学分析）
            ② 组装 ctx          把感知结果 + text/时间戳打包
            ③ _finish_ingest()  事实抽取 + 左脑/右脑写入 + audiomem 打标签

        语音那边只需给出 ``text``（多说话人/情绪等结构化输入见
        ``voice_input.ingest_voice_input``），音频感知全部发生在 ①。

        Parameters
        ----------
        observed_at:
            这句话实际发生的时间（如回填历史对话时传真实日期 "2023-05-08" 或 ISO
            字符串）。不传就用当下时刻；回填历史数据必须显式传，否则时序推理和按
            时间排序会失真。

        Returns
        -------
        dict
            ``{facts_count, memory_ids, affect}``
        """
        import time

        ts = observed_at or time.strftime("%H:%M:%S")

        # ① 流式预处理：场景/声纹/情绪等全部声学分析都在这一步（见 preprocess）
        p = self.preprocess(text, speaker, emotion, session_id, audio_path)
        speaker          = p.speaker
        emotion          = p.emotion
        environment      = p.environment
        environment_hint = p.environment_hint
        scene_tag        = p.scene_tag
        scene_raw_labels = p.scene_raw_labels
        person_id        = p.person_id
        tune_result      = p.tune_result
        abnormal_hits    = p.abnormal_hits
        detection        = p.detection
        place_result     = None   # 由 _finish_ingest 的场景聚类阶段填充
        new_routine      = None   # 由 _finish_ingest 的规律检测阶段填充

        ctx = {
            "text": text, "speaker": speaker, "emotion": emotion, "entities": entities,
            "session_id": session_id, "audio_path": audio_path, "observed_at": observed_at,
            "ts": ts,
            # AST remains the immediate hint. When CLAP final-memory mode
            # is enabled, don't put that provisional text in the utterance memory.
            "environment": "" if self._clap_memory_enabled() else environment,
            "environment_hint": environment_hint,
            "scene_tag": scene_tag,
            "scene_raw_labels": scene_raw_labels, "person_id": person_id,
            "tune_result": tune_result, "abnormal_hits": abnormal_hits,
            "place_result": place_result, "new_routine": new_routine, "detection": detection,
        }

        if self._clap_memory_enabled() and audio_path is not None:
            threading.Thread(
                target=self._finish_clap_environment,
                args=(audio_path, text, session_id, environment_hint),
                daemon=True,
            ).start()

        # async_facts=True：事实抽取 + 图谱写入（耗时的部分）扔进后台线程，
        # Ingest() 立刻带着已同步算完的 audiomem 字段返回。默认 False。
        if async_facts:
            threading.Thread(target=self._finish_ingest, args=(ctx,), daemon=True).start()
            return {
                "facts_count":         None,
                "memory_ids":          [],
                "affect":              None,
                "triggered_reminders": [],
                "proactive_memories":  [],
                "current_scene":       scene_tag or "",
                "environment_hint":    environment_hint,
                "speaker_id":          person_id or "",
                "recognized_tune":     (
                    {"tune_id": tune_result.tune_id, "action": tune_result.action,
                     "heard_count": tune_result.heard_count}
                    if tune_result is not None else None
                ),
                "abnormal_sounds":     [l for l, _ in abnormal_hits],
                "recognized_place":    (
                    {"place_id": place_result.place_id, "action": place_result.action,
                     "visit_count": place_result.visit_count,
                     "previous_visit_at": place_result.previous_visit_at}
                    if place_result is not None else None
                ),
                "familiar_place_prompt": None,
                "new_routine":         None,
            }

        return self._finish_ingest(ctx)

    def _tag_memories(self, memory_ids, tags) -> None:
        """给一批记忆写 memory_tags；tags=[(name, conf),...]。cog store 不支持就跳过。"""
        cog_store = self._get_repo()._cognitive_store
        if cog_store and hasattr(cog_store, "upsert_memory_tags"):
            for mid in memory_ids:
                cog_store.upsert_memory_tags(mid, self._user_id, tags)

    def _extract_and_append(self, messages, instructions, ts, extra_metadata):
        """合成消息 → 抽取原子事实 → 追加入库，返回新 memory_ids（抽不出则空）。
        audiomem 里 routine/music/abnormal/环境音 那几处合成记忆共用这条。"""
        extracted = self._get_extractor().extract(
            new_messages=messages, custom_instructions=instructions,
            observation_date=ts, current_date=ts,
        )
        if not extracted:
            return []
        return self._get_repo().append_extracted(
            extracted, user_id=self._user_id, extra_metadata=extra_metadata)

    def _finish_ingest(self, ctx: dict) -> dict:
        """Ingest() 里事实抽取 + 图谱写入（左脑/右脑）那部分，拆出来是为了让
        async_facts=True 时能扔进后台线程跑。"""
        text = ctx["text"]; speaker = ctx["speaker"]; emotion = ctx["emotion"]
        entities = ctx["entities"]; session_id = ctx["session_id"]
        audio_path = ctx["audio_path"]; observed_at = ctx["observed_at"]
        ts = ctx["ts"]; environment = ctx["environment"]; scene_tag = ctx["scene_tag"]
        scene_raw_labels = ctx["scene_raw_labels"]; person_id = ctx["person_id"]
        tune_result = ctx["tune_result"]; abnormal_hits = ctx["abnormal_hits"]
        place_result = ctx["place_result"]; new_routine = ctx["new_routine"]
        detection = ctx["detection"]
        environment_hint = ctx.get("environment_hint", "")

        import uuid
        from voicemem.utils.common.voice_input import VoiceInput, VoiceContent, ingest_voice_input

        vi = VoiceInput(
            id=f"utt_{uuid.uuid4().hex[:8]}",
            time_stamp={"begin": ts, "end": ts},
            slots=[],
            contents=[VoiceContent(
                sub_id="0", time_start=ts, time_end=ts,
                sentence=text, voiceprint_id=speaker, emotion=emotion,
            )],
            environment=environment,
        )

        result = ingest_voice_input(
            vi, self._user_id,
            registry=self._get_registry(),
            repo=self._get_repo(),
            extractor=self._get_extractor(),
            session_id=session_id,
            extra_metadata={"created_at": observed_at} if observed_at else None,
        )

        # ── audiomem：场景/声纹标签写入 + 触发提醒 + 录音归档 + 主动推送 ─────────
        audiomem = self._write_audiomem_tags(
            result, scene_tag, scene_raw_labels, detection, audio_path,
            person_id, tune_result, abnormal_hits, ts, session_id, text,
        )
        triggered_reminders = audiomem["triggered_reminders"]
        proactive_memories = audiomem["proactive_memories"]
        familiar_place_prompt = audiomem["familiar_place_prompt"]
        place_result = audiomem["place_result"]
        new_routine = audiomem["new_routine"]

        self._write_left_brain(result, text)
        self._write_right_brain(emotion, result, text, entities, observed_at)

        # 异步清洁：每多 50 条 heartnote 触发一次
        threading.Thread(target=self._check_and_cleanup, daemon=True).start()
        # 异步清洁：原声定期归档，每天最多跑一次，删除超过 30 天的 WAV 文件本体
        threading.Thread(target=self._check_and_cleanup_audio, daemon=True).start()

        # ── 短期/长期归因触发（session_id 变化 / 每轮）────────────────────────
        # 每次 Ingest() 都跑一次短期归因（run_short_term 对每个 touched entity 是
        # 同步 LLM 调用），Ingest() 的同步延迟按 touched entity 数量线性增加。
        turn_info = self._get_session_tracker().record_turn(self._user_id, session_id)

        # 短期归因每轮跑（长期归因在 session 边界批处理里跑）
        try:
            touched = self._get_session_tracker().pop_touched(self._user_id, "rb_entity_short")
            if touched:
                self._get_attribution_manager().run_short_term(self._user_id, touched)
        except Exception as e:
            print(f"[Attribution] 短期归因失败: {e}")

        if turn_info["session_changed"]:
            self._run_session_boundary_batch()

        return {
            "facts_count":         result.facts_count,
            "memory_ids":          result.memory_ids,
            "affect":              result.affect,
            "triggered_reminders": triggered_reminders,
            "proactive_memories":  proactive_memories,
            "current_scene":       scene_tag or "",
            "environment_hint":    environment_hint,
            "speaker_id":          person_id or "",
            "speaker_name":        (
                self._get_registry().display_name(person_id) if person_id else speaker
            ),
            "recognized_tune":     (
                {"tune_id": tune_result.tune_id, "action": tune_result.action,
                 "heard_count": tune_result.heard_count}
                if tune_result is not None else None
            ),
            "abnormal_sounds":     [l for l, _ in abnormal_hits],
            "recognized_place":    (
                {"place_id": place_result.place_id, "action": place_result.action,
                 "visit_count": place_result.visit_count,
                 "previous_visit_at": place_result.previous_visit_at}
                if place_result is not None else None
            ),
            "familiar_place_prompt": familiar_place_prompt,
            "new_routine":         new_routine,
        }

    def _write_audiomem_tags(
        self, result, scene_tag, scene_raw_labels, detection, audio_path,
        person_id, tune_result, abnormal_hits, ts, session_id, text,
    ) -> dict:
        """audiomem 写入段：场景/声纹/音乐/异常音/地点标签 + 触发提醒 + 录音归档
        + 场景切换主动推送。返回后续 return 需要的字段。"""
        triggered_reminders: list[dict] = []
        proactive_memories: list[dict] = []
        familiar_place_prompt: dict | None = None
        place_result = None
        new_routine = None
        _fire_result = None
        if scene_tag and result.memory_ids:
            try:
                self._tag_memories(result.memory_ids, [(f"scene:{scene_tag}", 0.95)])
            except Exception as _e:
                print(f"  [scene] tag write failed: {_e}", flush=True)

            # 场景变化 → 触发匹配的提醒（先于 update_scene，保留 scene_changed 信息）
            try:
                from voicemem.utils.audio.environment.scene_trigger import check_and_fire
                _fire_result = check_and_fire(
                    self._get_trigger_store(), self._user_id, scene_tag,
                    raw_labels=scene_raw_labels,
                )
                triggered_reminders = [
                    {"id": t.id, "message": t.message, "scene": t.trigger_scene}
                    for t in _fire_result.fired
                ]
            except Exception as _e:
                print(f"  [scene_trigger] check failed: {_e}", flush=True)

            # 生活声音规律：自动建立 routine 记忆——只在真正"进入"场景时记一次
            # 观测（scene_changed），积累够天数后只在刚跨过阈值那次生成记忆。
            if _fire_result is not None and _fire_result.scene_changed:
                try:
                    from datetime import datetime as _datetime
                    from voicemem.utils.audio.environment.routine_memory import bucket_label
                    routine_check = self._get_routine_store().observe(
                        self._user_id, scene_tag, _datetime.now()
                    )
                    if routine_check["is_new_routine"]:
                        blabel = bucket_label(routine_check["bucket"])
                        new_routine = {
                            "scene": scene_tag,
                            "bucket": routine_check["bucket"],
                            "bucket_label": blabel,
                            "distinct_days": routine_check["distinct_days"],
                        }
                        synthetic_message = [{"role": "user", "content": (
                            f"[Routine pattern detected: user is regularly in "
                            f"'{scene_tag}' scene during {blabel}, "
                            f"observed on {routine_check['distinct_days']} different days]"
                        )}]
                        custom_instructions = (
                            "This is an automatically detected behavioral routine based on "
                            "recurring acoustic scene observations over multiple days. "
                            "Extract a concise memory fact describing this habitual pattern "
                            "(e.g. 'User usually commutes around 7-9am'). "
                            "Do NOT invent specifics beyond the scene and time window given."
                        )
                        self._extract_and_append(synthetic_message, custom_instructions, ts, {
                            "source": "routine",
                            "routine_scene": scene_tag,
                            "routine_bucket": routine_check["bucket"],
                            "routine_distinct_days": routine_check["distinct_days"],
                            **({"session_id": session_id} if session_id is not None else {}),
                        })
                except Exception as _e:
                    print(f"  [routine] check failed: {_e}", flush=True)

            # 熟悉地点自动聚类：同样只在"进入"场景时识别一次（scene_changed），
            # 用整段录音的原始 AST embedding 捕捉这个具体地点的声学指纹。
            if (
                _fire_result is not None and _fire_result.scene_changed
                and detection.get("embedding") is not None
            ):
                try:
                    from datetime import datetime as _datetime
                    place_result = self._get_place_store().identify(
                        detection["embedding"], scene=scene_tag, when=_datetime.now()
                    )
                except Exception as _e:
                    print(f"  [place] identification skipped: {_e}", flush=True)

        # WAV 存档：记录 audio_path → memory_id 映射
        if audio_path and result.memory_ids:
            try:
                self._get_audio_archive().record(
                    result.memory_ids, self._user_id, str(audio_path)
                )
            except Exception as _e:
                print(f"  [audio_archive] record failed: {_e}", flush=True)

        # 声纹标签：把 person_id 写入 memory_tags，供后续按说话人检索
        if person_id and result.memory_ids:
            try:
                self._tag_memories(result.memory_ids, [(f"speaker:{person_id}", 1.0)])
            except Exception as _e:
                print(f"  [speaker] tag write failed: {_e}", flush=True)

        # 音乐/哼唱标签：把 tune_id 写入 memory_tags 供按"同一首歌/调子"检索；
        # heard_count>=2 时额外生成一条"又听到熟悉的调子"记忆事实。
        if tune_result is not None and result.memory_ids:
            try:
                self._tag_memories(result.memory_ids, [(f"tune:{tune_result.tune_id}", 0.9)])
            except Exception as _e:
                print(f"  [music] tag write failed: {_e}", flush=True)

            if tune_result.action == "match" and tune_result.heard_count >= 2:
                try:
                    tune_labels = self._get_music_store().get_meta(tune_result.tune_id).get("labels", [])
                    synthetic_message = [{"role": "user", "content": (
                        f"[Recognized recurring background music/humming, heard "
                        f"{tune_result.heard_count} times before: {', '.join(tune_labels) or 'unknown tune'}]"
                    )}]
                    custom_instructions = (
                        "This is a recognition event for a recurring background tune (music or humming) "
                        "that has been heard multiple times before, detected via acoustic similarity. "
                        "Extract a concise memory fact noting that this familiar tune came up again. "
                        "Do NOT invent song titles or lyrics you don't actually know."
                    )
                    self._extract_and_append(synthetic_message, custom_instructions, ts, {
                        "source": "music_recognition",
                        "tune_id": tune_result.tune_id,
                        "heard_count": tune_result.heard_count,
                        **({"session_id": session_id} if session_id is not None else {}),
                    })
                except Exception as _e:
                    print(f"  [music] recognition fact skipped: {_e}", flush=True)

        # 异常环境音记忆：破碎声/警报/尖叫——第一次出现就值得记一笔，每次检测到
        # 都打标签 + 生成一条记忆事实。独立警报事实的写入不依赖 result.memory_ids
        # （异常声音值不值得记，与这句话本身有没有独立事实价值无关）。
        if abnormal_hits:
            alert_memory_ids: list[str] = []
            try:
                labels_str = ", ".join(f"{l}({s:.2f})" for l, s in abnormal_hits)
                synthetic_message = [{"role": "user", "content": (
                    f"[Abnormal environmental sound event detected: {labels_str}]"
                )}]
                custom_instructions = (
                    "This is an unusual/alerting non-speech environmental sound event "
                    "(e.g. breaking glass, alarm, siren, or screaming) captured during the "
                    "conversation. Extract a concise memory fact describing this notable event. "
                    "Do NOT extract facts about the conversation topic itself."
                )
                alert_memory_ids = self._extract_and_append(synthetic_message, custom_instructions, ts, {
                    "source": "abnormal_sound",
                    "abnormal_labels": [l for l, _ in abnormal_hits],
                    **({"session_id": session_id} if session_id is not None else {}),
                })
            except Exception as _e:
                print(f"  [abnormal] alert fact skipped: {_e}", flush=True)

            # 打标签：原有 turn 的记忆 + 上面新建的独立警报事实两边都打，保证至少
            # 有一条可检索记忆挂着 abnormal:<label> 标签。
            try:
                cog_store = self._get_repo()._cognitive_store
                tag_target_ids = list(result.memory_ids) + alert_memory_ids
                if cog_store and hasattr(cog_store, "upsert_memory_tags") and tag_target_ids:
                    tags = [
                        (f"abnormal:{label.lower().replace(' ', '_').replace(',', '')}", score)
                        for label, score in abnormal_hits
                    ]
                    for mid in tag_target_ids:
                        cog_store.upsert_memory_tags(mid, self._user_id, tags)
            except Exception as _e:
                print(f"  [abnormal] tag write failed: {_e}", flush=True)

        # 熟悉地点标签：把 place_id 写入 memory_tags，供后续按"同一个具体地点"检索。
        if place_result is not None and result.memory_ids:
            try:
                self._tag_memories(result.memory_ids, [(f"place:{place_result.place_id}", 0.9)])
            except Exception as _e:
                print(f"  [place] tag write failed: {_e}", flush=True)

        # 熟悉环境主动提示"上次在这里"：只有识别成 match 才有"上次"可提。从这个
        # 地点之前打过 place:<id> 标签的记忆里挑跟当前话题相关的几条，附带到访信息。
        if place_result is not None and place_result.action == "match" and result.memory_ids:
            try:
                cog_store = self._get_repo()._cognitive_store
                place_memories: list[dict] = []
                if cog_store and hasattr(cog_store, "memory_ids_for_slots_v2"):
                    place_mem_ids = set(
                        cog_store.memory_ids_for_slots_v2(
                            self._user_id, [f"place:{place_result.place_id}"]
                        )
                    ) - set(result.memory_ids)
                    if place_mem_ids:
                        _hits = self.Rank(text, place_mem_ids, top_k=3)
                        place_memories = [
                            {"memory_id": h.memory_id, "content": h.text, "score": round(h.score, 3)}
                            for h in _hits
                        ]
                familiar_place_prompt = {
                    "place_id": place_result.place_id,
                    "visit_count": place_result.visit_count,
                    "previous_visit_at": place_result.previous_visit_at,
                    "memories": place_memories,
                }
            except Exception as _e:
                print(f"  [place] proactive recall failed: {_e}", flush=True)

        # 场景切换主动推送：进入新场景时检索场景相关记忆（用 scene_changed 避免重复触发）
        if scene_tag and _fire_result is not None and _fire_result.scene_changed:
            try:
                _query = _SCENE_RECALL_QUERY.get(scene_tag, "")
                if _query:
                    _hits = self.Rank(_query, set(), top_k=3)
                    proactive_memories = [
                        {"memory_id": h.memory_id, "content": h.text, "score": round(h.score, 3)}
                        for h in _hits
                    ]
            except Exception as _e:
                print(f"  [proactive] failed: {_e}", flush=True)

        return {
            "triggered_reminders": triggered_reminders,
            "proactive_memories": proactive_memories,
            "familiar_place_prompt": familiar_place_prompt,
            "place_result": place_result,
            "new_routine": new_routine,
        }

    def _write_left_brain(self, result, text) -> None:
        """左脑写入段：LLM 打 slot 标签 + slot→entity 图层写入。"""
        if not result.memory_ids:
            return
        # LLM 打标签（覆盖 embedding 标签）
        try:
            llm_slots = self._llm_tag_memories(text, result.memory_ids)
            primary_slot = llm_slots[0] if llm_slots else None
        except Exception as e:
            print(f"[v5] LLM 打标签失败: {e}", flush=True)
            primary_slot = None
            llm_slots = []

        # 语义簇宏观连接：这条记忆同时打了 2 个以上 slot 标签，说明这几个 slot
        # 之间存在真实关联，从数据共现自动学，不是人工写死的关系表。
        if len(llm_slots) >= 2:
            try:
                self._get_repo()._cognitive_store.record_slot_cooccurrence(
                    self._user_id, llm_slots
                )
            except Exception as e:
                print(f"[SlotMacro] 共现记录失败: {e}", flush=True)

        # 左脑 slot→entity 图层：把这条记忆挂到对应slot下的entity节点
        # （entity 名字复用 CognitiveAnnotator 已经抽取好的实体，不额外调LLM）
        try:
            cog_store = self._get_repo()._cognitive_store
            graph_store = self._get_graph_entity_store()
            if cog_store is not None and primary_slot:
                for mid in result.memory_ids:
                    for eid in cog_store.entity_ids_for_memory(mid):
                        ent = cog_store.get_entity(eid)
                        if ent is None:
                            continue
                        ent_emb = self._embed_text(ent.name)
                        g_ent, _created = graph_store.get_or_create_entity_semantic(
                            self._user_id, primary_slot, ent.name, ent_emb,
                        )
                        graph_store.link_memory(g_ent.id, self._user_id, mid)
        except Exception as e:
            print(f"[GraphEntity] 左脑图层写入失败: {e}")

    def _write_right_brain(self, emotion, result, text, entities, observed_at) -> None:
        """右脑写入段：每条 utterance 一条 heartnote，挂 emotion + entity anchors +
        关系节点 + 右脑 slot→entity 图层。gate 只看 emotion（不绑 result.memory_ids）——
        纯情绪句左脑可能抽不出事实但情绪仍值得记；mid 为空时不挂证据、不查左脑实体
        链接，但情绪锚点 + 文本实体名锚点仍正常写。"""
        if not emotion:
            return
        try:
            from voicemem.rightbrain.types import MemoryAnchor
            rb_repo = self._get_rb_repo()
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
                cog_store = self._get_repo()._cognitive_store
                if mid and cog_store is not None:
                    rb_graph = self._get_rb_graph_store()
                    tracker  = self._get_session_tracker()
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
                rb_graph = self._get_rb_graph_store()
                tracker = self._get_session_tracker()
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
                    label_emb = self._embed_text(label)
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

    def _run_session_boundary_batch(self) -> None:
        """session 边界批处理：左脑子图判定 + 右脑长期归因。

        左脑这部分把 session 里攒下的检索记账拿出来判断一次
        （RunSubgraphCheckpoint）；纯 ingest（无穿插检索）时天然是空操作。

        由 Ingest() 在检测到 session_id 变化时自动调用。session_changed 靠"看到
        下一个 session 的第一条 ingest"倒推，最后一个 session 没有下一条触发，
        所以 ingest 完之后调用方必须显式调一次 Flush() 补跑最后一个 session。
        """
        try:
            self.RunSubgraphCheckpoint()
        except Exception as e:
            print(f"[Subgraph] session边界判定失败: {e}")

        # schema 描述刷新：给本 session 新增过记忆的 slot 重写一句 ≤40 词的综合
        # 描述，检索时附进 prompt，提供单条事实给不出的跨记忆聚合信息。
        try:
            self._refresh_schema_descriptions()
        except Exception as e:
            print(f"[SchemaDesc] 刷新失败: {e}")

        try:
            touched_slots = self._get_session_tracker().pop_touched(self._user_id, "rb_slot_long")
            if touched_slots:
                self._get_attribution_manager().run_long_term(self._user_id, touched_slots)
        except Exception as e:
            print(f"[Attribution] 长期归因失败: {e}")

    _SCHEMA_DESC_MIN_NEW = 1      # slot 新增 ≥N 条记忆才重写描述
    _SCHEMA_DESC_MAX_FACTS = 80   # 摘要输入上限（最近的在前）

    def _refresh_schema_descriptions(self) -> None:
        """给记忆数有变化的 slot 重写一句综合描述，写入 cognitive store 的 slot_summaries。
        描述语言跟随记忆语言；带该领域最近一条记忆的日期，避免 temporal 题被无日期的
        概括带偏。"""
        repo = self._get_repo()
        cog = repo._cognitive_store
        if cog is None or not hasattr(cog, "memory_ids_for_slots_v2"):
            return
        entries = {}
        try:
            for e in repo._vector_store.list_entries(user_id=self._user_id):
                entries[e["id"]] = e
        except Exception:
            entries = {}
        slots = list(SLOT_RELATIONS.keys())
        try:
            slots += [d.name for d in self._get_dynamic_slot_store().get_dynamic_slots(self._user_id)]
        except Exception:
            pass
        for slot in dict.fromkeys(slots):
            try:
                mids = cog.memory_ids_for_slots_v2(self._user_id, [slot])
                n = len(mids)
                if n < 3:
                    continue
                last = cog.get_slot_summary_mem_count(self._user_id, slot) if hasattr(cog, "get_slot_summary_mem_count") else 0
                if n - last < self._SCHEMA_DESC_MIN_NEW:
                    continue
                facts = []
                for mid in mids:
                    e = entries.get(mid)
                    if e is not None and e["text"]:
                        facts.append((e["date"], e["text"]))
                    else:
                        rec = cog.get_memory_record(mid) if hasattr(cog, "get_memory_record") else None
                        if rec and rec.content:
                            facts.append(("", rec.content))
                if len(facts) < 3:
                    continue
                facts.sort(key=lambda t: t[0], reverse=True)
                facts = facts[: self._SCHEMA_DESC_MAX_FACTS]
                latest = next((d for d, _ in facts if d), "")
                sample = "\n".join(f"- {('[' + d + '] ') if d else ''}{t}" for d, t in facts)
                en = _is_en_text(" ".join(t for _, t in facts[:10]))
                prompt = (
                    f"Below are memory facts about a user, all under the life domain '{slot}'.\n"
                    "Write ONE concise sentence (max 40 words) summarizing the overall picture in this domain: "
                    "the main people, ongoing situations, and how things changed over time. Plain and factual, "
                    "no fluff. Mention the most recent date if relevant. Output only the sentence.\n\n"
                    if en else
                    f"以下是用户在「{slot}」这个维度上的记忆事实。\n"
                    "请用一句话（40字以内）概括这个维度的整体情况：主要人物、正在进行的事、随时间的变化。"
                    "平实、有依据、不抒情；如相关请带上最近的日期。只输出这一句。\n\n"
                ) + sample
                text = (self._llm_text(prompt) or "").strip()
                if not text:
                    continue
                if latest and latest not in text:
                    text = f"{text} (latest: {latest})" if en else f"{text}（最近：{latest}）"
                cog.upsert_slot_summary(self._user_id, slot, text, n)
            except Exception as e:
                print(f"[SchemaDesc] {slot} 失败: {e}")

    def Flush(self) -> None:
        """对话/会话正式结束时调用一次，补跑最后一个 session 漏掉的批处理
        （子图判定 + 右脑长期归因，见 _run_session_boundary_batch 的说明）。
        幂等：没有新 touched refs 时是空操作。
        """
        self._run_session_boundary_batch()
        try:
            touched = self._get_session_tracker().pop_touched(self._user_id, "rb_entity_short")
            if touched:
                self._get_attribution_manager().run_short_term(self._user_id, touched)
        except Exception as e:
            print(f"[Attribution] 短期归因失败: {e}")

    def IngestEnv(
        self,
        audio_path,
        recent_context: list[dict] | None = None,
        session_id: int | str | None = None,
        environment_override: str | None = None,
    ) -> dict:
        """将一段环境音事件存入记忆库。

        Parameters
        ----------
        audio_path:
            环境音 wav 文件路径。
        recent_context:
            最近几轮对话文本，格式 [{"role": "user"/"assistant", "content": "..."}]。
            供 LLM 推断用户当时在做什么。
        session_id:
            当前会话 ID，用于时序排序。

        Returns
        -------
        dict
            ``{facts_count, memory_ids}``
        """
        import time
        from pathlib import Path as _Path

        # ── Step 1: 识别背景音；CLAP 后台复核时直接使用其结果 ────────────────
        if environment_override:
            env_str = environment_override
        else:
            try:
                pairs = self._get_env_detector().detect_full(_Path(audio_path)).get("pairs") or []
                env_str = (
                    "background sounds: " + ", ".join(f"{label}({score:.2f})" for label, score in pairs)
                    if pairs else ""
                )
            except Exception as e:
                print(f"  [IngestEnv] detection failed: {e}", flush=True)
                return {"facts_count": 0, "memory_ids": []}

        if not env_str:
            return {"facts_count": 0, "memory_ids": []}

        # ── Step 2: 构造 custom_instructions（环境音 + 对话上下文） ────────────
        ctx_lines = ""
        if recent_context:
            ctx_lines = "\n".join(
                f"{m['role'].capitalize()}: {m['content']}"
                for m in recent_context[-6:]
            )

        custom_instructions = (
            "This is a non-speech environmental sound event captured during the conversation. "
            "Based on the detected background sounds and the recent conversation context below, "
            "extract a concise memory fact describing what the user was likely doing or experiencing at this moment. "
            "Do NOT extract facts about the conversation topic itself — focus only on the environmental activity.\n"
            + (f"Recent conversation context:\n{ctx_lines}" if ctx_lines else "")
        )

        # ── Step 3: 用 extractor 生成 fact ────────────────────────────────────
        synthetic_message = [{"role": "user", "content": f"[Environmental sound event: {env_str}]"}]
        try:
            extracted = self._get_extractor().extract(
                new_messages=synthetic_message,
                custom_instructions=custom_instructions,
                observation_date=time.strftime("%H:%M:%S"),
                current_date=time.strftime("%H:%M:%S"),
            )
        except Exception as e:
            print(f"  [IngestEnv] extraction failed: {e}", flush=True)
            return {"facts_count": 0, "memory_ids": []}

        if not extracted:
            return {"facts_count": 0, "memory_ids": []}

        # ── Step 4: 存入左脑 ──────────────────────────────────────────────────
        meta = {
            "source":           "environment",
            "background_sounds": env_str,
            **({"session_id": session_id} if session_id is not None else {}),
        }
        memory_ids = self._get_repo().append_extracted(
            extracted, user_id=self._user_id, extra_metadata=meta
        )

        return {"facts_count": len(extracted), "memory_ids": memory_ids or []}

    # ── 右脑清洁模块 ───────────────────────────────────────────────────────────

    def _check_and_cleanup(self) -> None:
        """每增加 50 条 heartnote 触发一次右脑清洁。"""
        try:
            import json as _json
            state_path = self._memory_root / "cleanup_state.json"
            state = _json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"last_count": 0}

            rb_repo = self._get_rb_repo()
            all_mems = rb_repo._store.get_all(self._user_id)
            current_count = sum(1 for m in all_mems if m.memory_class == "heartnote")

            if current_count - state.get("last_count", 0) >= 50:
                state_path.write_text(
                    _json.dumps({"last_count": current_count}), encoding="utf-8"
                )
                self._run_cleanup()
        except Exception as e:
            print(f"[Cleanup] check error: {e}")

    def _check_and_cleanup_audio(self, retention_days: int = 30) -> None:
        """原声定期归档：每天最多跑一次，删除超过保留期的 WAV 文件本体。
        按时间触发，用单独的状态文件节流，避免每次 Ingest 都扫一遍 DB。
        """
        try:
            import json as _json
            from datetime import datetime, timezone
            state_path = self._memory_root / "audio_cleanup_state.json"
            state = _json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"last_run": ""}

            last_run = state.get("last_run", "")
            now = datetime.now(timezone.utc)
            if last_run:
                try:
                    elapsed_hours = (now - datetime.fromisoformat(last_run)).total_seconds() / 3600
                except ValueError:
                    elapsed_hours = 999
            else:
                elapsed_hours = 999

            if elapsed_hours < 24:
                return

            state_path.write_text(
                _json.dumps({"last_run": now.isoformat()}), encoding="utf-8"
            )
            self._get_audio_archive().cleanup_expired(retention_days=retention_days)
        except Exception as e:
            print(f"[Cleanup] audio check error: {e}")

    def _run_cleanup(self) -> None:
        """用 LLM 清洁右脑 heartnote：重复/无意义 → 删除；矛盾 → 标注 supersede。

        矛盾不"删旧留新"（偏好演化题需要新旧两条 + 先后关系）：旧条目打
        superseded_by/superseded_at 标记保留，渲染时标注"旧况"并降权。"""
        try:
            import json as _json
            import sqlite3

            rb_repo = self._get_rb_repo()
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


__all__ = ["VoiceMem", "SearchResult"]
