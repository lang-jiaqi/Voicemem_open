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

from voicemem.leftbrain.cognitive_graph.query_slot_classifier import QueryClassification
from voicemem.leftbrain.local_memory_store import MemorySearchHit
# 左脑那一整块（slot 过滤/实体缩窄/时间扩候选/向量排序/查询分类/LLM 打标签/
# slot→entity 图层/子图记账与 checkpoint/schema 描述刷新/冷记忆归档）搬进了
# LeftBrain 组件；_search_mode 辅助函数随之迁至 voicemem.leftbrain.brain。这里
# 反向 import 回来，维持既有 `from voicemem.engine import _search_mode` 等契约。
from voicemem.leftbrain.brain import LeftBrain, _search_mode
from voicemem.utils.audio.perceiver import AudioPerception, AudioPerceiver
# 右脑那一整块（heartnote 写入/内心OS/图层/检索/清洁）搬进了 RightBrain 组件；
# RightBrainHit 数据类与 _rb_* 辅助函数随之迁至 voicemem.rightbrain.brain。
# 这里反向 import 回来，维持既有 `from voicemem.engine import RightBrainHit /
# _rb_ctx_to_hits / _rb_blended_priority` 等调用点与测试的导入契约不变。
from voicemem.rightbrain.brain import (
    RightBrain,
    RightBrainHit,
    _is_en_text,
    _rb_blended_priority,
    _rb_ctx_to_hits,
    _rb_emotion_trait_hit,
    _rb_graph_hits,
    _rb_lang,
    _rb_mem_date,
    _rb_relation_hits,
    _render_rb_directive,
)


# ── 结果容器 ───────────────────────────────────────────────────────────────────

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


# 左脑那一整块的候选池构造/救回常量（_RESCUE_K / _POOL_MODE_ENV / _pool_mode /
# _STRICT_* 等）与 _search_mode 辅助函数随左脑块迁至 voicemem.leftbrain.brain
# （_search_mode 在本模块顶部导入回来，供 Search() 组装 SearchResult 时调用）。
# RightBrainHit / _rb_* 辅助函数与 _is_en_text 已随右脑块迁至
# voicemem.rightbrain.brain（本模块顶部导入回来，供 Search() 等继续直接调用）；
# AudioPerception 迁至 voicemem.utils.audio.perceiver。


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

        # ── 左脑组件（组合模式：自持左脑零件 + 显式注入跨域/运行时依赖）───────────
        # 左脑那一整块（slot 过滤/实体缩窄/时间扩候选/向量排序/查询分类/LLM 打标签/
        # slot→entity 图层/子图记账与 checkpoint/schema 描述刷新/冷记忆归档）搬进了
        # LeftBrain。它自持 5 个左脑侧懒加载单例（repo/extractor/dynamic_slot_store/
        # graph_entity_store/subgraph_manager，与宿主共享同一 _cache/_lock）；凡是要
        # 用到文本 embedding / LLM(JSON) / LLM(text) / 可注入分类器 / 会话追踪器这些
        # 跨域或运行时能力的地方，一律以 getter/函数引用在此显式注入（懒加载语义不变）。
        # 先于 _audio/_right 构造：engine 的 _get_repo 等转发到 self._left，且 _audio/
        # _right 注入的 repo=self._get_repo 会经转发落到这里同一份左脑单例。
        self._left = LeftBrain(
            memory_root=self._memory_root,
            user_id=self._user_id,
            base_url=self._base_url,
            cognitive_db=self._cognitive_db,
            embedder=self._embedder,
            vector_store=self._vector_store,
            embed=self._embed_text,
            llm_json=self._llm_json,
            llm_text=self._llm_text,
            classifier=self._classifier,
            tracker=self._get_session_tracker,
            cache=self._cache,
            lock=self._lock,
        )

        # ── 音频感知组件（组合模式：自持音频零件 + 显式注入左脑依赖）───────────
        # 音频那一整块（场景/声纹/情绪/环境音/audiomem 标签/回放）搬进了
        # AudioPerceiver。它自持 10 个音频侧懒加载单例（env/clap/speaker/vp/
        # emotion/music/routine/place/trigger/audio_archive）连同各自缓存/锁与
        # 说话人绑定状态（_session_person_pin / _person_origin_session）；凡是要
        # 用到左脑存储 / 抽取器 / 声纹姓名映射 / 打标签 / 事实追加 / 语义排序这些
        # 非音频能力的地方，一律以 getter/函数引用在此显式注入（懒加载语义不变）。
        self._audio = AudioPerceiver(
            memory_root=self._memory_root,
            user_id=self._user_id,
            base_url=self._base_url,
            enable_scene=self._enable_scene,
            enable_music=self._enable_music,
            enable_abnormal_sound=self._enable_abnormal_sound,
            enable_voiceprint=self._enable_voiceprint,
            enable_emotion=self._enable_emotion,
            repo=self._get_repo,
            extractor=self._get_extractor,
            registry=self._get_registry,
            tag=self._tag_memories,
            extract_and_append=self._extract_and_append,
            rank=self.Rank,
            ingest_env=lambda: self.IngestEnv,
            cache=self._cache,
            lock=self._lock,
        )

        # ── 右脑组件（组合模式：自持右脑零件 + 显式注入跨域依赖）─────────────────
        # 右脑那一整块（heartnote 情感写入/内心OS/图层/检索/LLM清洁）搬进了
        # RightBrain。它自持 3 个右脑侧懒加载单例（rb_repo/rb_graph_store/
        # attribution_manager，与宿主共享同一 _cache/_lock）；凡是要用到文本
        # embedding / LLM(JSON) / LLM(text) / 会话追踪器 / 左脑仓库 / 内心OS 生成 /
        # 特质抽取这些非右脑本域能力的地方，一律以 getter/函数引用在此显式注入
        # （懒加载语义不变；generate_inner_os / extract_rb_traits 延迟解析以便测试 patch）。
        self._right = RightBrain(
            memory_root=self._memory_root,
            user_id=self._user_id,
            base_url=self._base_url,
            cognitive_db=self._cognitive_db,
            embed=self._embed_text,
            llm_json=self._llm_json,
            llm_text=self._llm_text,
            tracker=self._get_session_tracker,
            repo=self._get_repo,
            generate_inner_os=lambda text, emotion, entities: self._generate_inner_os(text, emotion, entities),
            extract_rb_traits=lambda text, emotion: self._extract_rb_traits(text, emotion),
            cache=self._cache,
            lock=self._lock,
        )

    # ── 懒加载单例 ──────────────────────────────────────────────────────────────

    def _get_repo(self):
        # 左脑单例已随左脑块搬进 LeftBrain 组件；转发以维持既有调用点/测试对
        # VoiceMem 实例的直接访问，以及 _audio/_right 注入的 repo=self._get_repo
        # （读写的是共享 _cache 里同一个 "repo"）。
        return self._left._get_repo()

    def _get_rb_repo(self):
        # 右脑单例已随右脑块搬进 RightBrain 组件；转发以维持既有调用点/测试对
        # VoiceMem 实例的直接访问（读写的是共享 _cache 里同一个 "rb_repo"）。
        return self._right._rb_repo()

    def _get_extractor(self):
        # 左脑单例已搬进 LeftBrain 组件；转发（共享 _cache 里同一个 "extractor"）。
        return self._left._get_extractor()

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
        return self._audio._env_detector()

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
        return self._audio._clap_env_detector()

    def _finish_clap_environment(self, *a, **k) -> None:
        return self._audio._finish_clap_environment(*a, **k)

    def _get_trigger_store(self):
        return self._audio._trigger_store()

    def _get_audio_archive(self):
        return self._audio._audio_archive()

    def _get_speaker_encoder(self):
        return self._audio._speaker_encoder()

    def _get_vp_store(self):
        return self._audio._vp_store()

    def _get_emotion_detector(self):
        return self._audio._emotion_detector()

    def _get_music_store(self):
        return self._audio._music_store()

    def _get_routine_store(self):
        return self._audio._routine_store()

    def _get_place_store(self):
        return self._audio._place_store()

    # 说话人绑定状态与声纹回收：状态和逻辑都随音频组件走，这里保留转发以维持
    # 既有调用点/测试对 VoiceMem 实例的直接访问（读写的是同一份底层 dict）。
    @property
    def _session_person_pin(self) -> dict[str, str]:
        return self._audio._session_person_pin

    @property
    def _person_origin_session(self) -> dict[str, str]:
        return self._audio._person_origin_session

    def _claimed_by_other_identity(self, *a, **k) -> bool:
        return self._audio._claimed_by_other_identity(*a, **k)

    def _reconcile_speaker_candidates(self, *a, **k) -> tuple[str, str]:
        return self._audio._reconcile_speaker_candidates(*a, **k)

    # ── audiomem：场景触发提醒 ───────────────────────────────────────────────────

    def CreateSceneTrigger(self, *a, **k) -> dict:
        return self._audio.CreateSceneTrigger(*a, **k)

    def GetOriginalAudio(self, *a, **k) -> dict:
        return self._audio.GetOriginalAudio(*a, **k)

    def TryPlayback(self, *a, **k) -> dict | None:
        return self._audio.TryPlayback(*a, **k)

    # ── Dynamic slot（子图机制涌现的新 slot） ──────────────────────────────────

    def _get_dynamic_slot_store(self):
        # 左脑单例已搬进 LeftBrain 组件；转发（共享 _cache 里同一个 "dynamic_slot_store"）。
        return self._left._get_dynamic_slot_store()

    def _get_dynamic_slots(self) -> list[tuple[str, str]]:
        """返回该用户已涌现的动态 slot [(name, description), ...]，转发到 LeftBrain。"""
        return self._left._get_dynamic_slots()

    # ── slot→entity 图层（左脑：挂在 SlotV2 下；右脑：5个感性slot） ─────────────

    def _get_graph_entity_store(self):
        # 左脑单例已搬进 LeftBrain 组件；转发（共享 _cache 里同一个 "graph_entity_store"）。
        return self._left._get_graph_entity_store()

    def _get_rb_graph_store(self):
        # 右脑单例已搬进 RightBrain 组件；转发（共享 _cache 里同一个 "rb_graph_store"）。
        return self._right._rb_graph_store()

    def _get_session_tracker(self):
        with self._lock:
            if "session_tracker" not in self._cache:
                from voicemem.utils.common.session_tracker import SessionTracker
                self._cache["session_tracker"] = SessionTracker(
                    self._memory_root / "session_tracker.sqlite"
                )
        return self._cache["session_tracker"]

    def _get_subgraph_manager(self):
        # 左脑单例已搬进 LeftBrain 组件；转发（共享 _cache 里同一个 "subgraph_manager"）。
        return self._left._get_subgraph_manager()

    def _get_attribution_manager(self):
        # 右脑单例已搬进 RightBrain 组件；转发（共享 _cache 里同一个 "attribution_manager"）。
        return self._right._attribution_manager()

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

    # ── 左脑检索步骤（转发到 LeftBrain：SearchCogGraph/SearchData/Rank 等）─────

    def SearchCogGraph(self, *a, **k) -> tuple[set[str], QueryClassification]:
        """slot 过滤，转发到 LeftBrain.SearchCogGraph。"""
        return self._left.SearchCogGraph(*a, **k)

    def SearchData(self, *a, **k) -> set[str]:
        """实体缩窄，转发到 LeftBrain.SearchData。"""
        return self._left.SearchData(*a, **k)

    def _search_data_impl(self, *a, **k) -> tuple[set[str], list[str]]:
        """SearchData 真正实现（多返 activated_names），转发到 LeftBrain。"""
        return self._left._search_data_impl(*a, **k)

    def _widen_for_time_question(self, *a, **k) -> set[str]:
        """时间类问题扩候选，转发到 LeftBrain。"""
        return self._left._widen_for_time_question(*a, **k)

    def Rank(self, *a, **k) -> list[MemorySearchHit]:
        """向量相似度排序，转发到 LeftBrain.Rank。"""
        return self._left.Rank(*a, **k)

    # ── v5：LLM 打标签（转发到 LeftBrain）─────────────────────────────────────

    def _get_slot_base_embeddings(self, *a, **k) -> dict[str, list[float]]:
        return self._left._get_slot_base_embeddings(*a, **k)

    def _get_slot_dyn_embeddings(self, *a, **k) -> dict[str, list[float]]:
        return self._left._get_slot_dyn_embeddings(*a, **k)

    def _normalize_slot_name(self, *a, **k) -> str:
        return self._left._normalize_slot_name(*a, **k)

    def _llm_tag_memories(self, *a, **k) -> list[str]:
        return self._left._llm_tag_memories(*a, **k)

    # ── 查询分类（含动态 slot） ────────────────────────────────────────────────

    def Classify(self, *a, **k) -> QueryClassification:
        """LLM 分类 query → slots + entities，转发到 LeftBrain.Classify。"""
        return self._left.Classify(*a, **k)

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

    def _record_subgraph_activation(self, *a, **k) -> None:
        """检索结果记账，转发到 LeftBrain._record_subgraph_activation。"""
        return self._left._record_subgraph_activation(*a, **k)

    def RunSubgraphCheckpoint(self, *a, **k) -> dict:
        """子图 checkpoint（建图→判断），转发到 LeftBrain.RunSubgraphCheckpoint。"""
        return self._left.RunSubgraphCheckpoint(*a, **k)

    def ArchiveColdMemories(self, *a, **k) -> dict:
        """冷记忆归档，转发到 LeftBrain.ArchiveColdMemories。"""
        return self._left.ArchiveColdMemories(*a, **k)

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

        # ①②+摘要 左脑检索段（SearchCogGraph→SearchData→时间扩候选→相关槽摘要）
        # 整段抽进 LeftBrain.search；实体缩窄先于右脑跑完（右脑依赖左脑"已激活"的
        # 实体集合），t0/t1/t2 由组件回传，timing 语义不变。
        left = self._left.search(
            query, slots, entities, scene_filter, speaker_filter,
        )
        slot_mem_ids      = left["slot_mem_ids"]
        final_ids         = left["final_ids"]
        activated_names   = left["activated_names"]
        classification    = left["classification"]
        related_summaries = left["related_summaries"]
        t0, t1, t2 = left["t0"], left["t1"], left["t2"]

        # ③ 右脑（依赖 activated_names）与 Rank（向量排序，依赖 final_ids）并发执行——
        # 两者互不依赖对方输出，可以并发。
        rb_hits: list[RightBrainHit] = []
        rb_directive = ""
        rb_duration  = 0.0

        # 右脑检索段已抽进 RightBrain.search、向量排序已抽进 LeftBrain.rank；这里只
        # 保留 Rank ‖ 右脑 并发跑的 ThreadPoolExecutor 结构，两半都换成组件调用。
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            rb_future = pool.submit(
                self._right.search, query, activated_names, emotion, top_k,
            )                                            # 右脑并发开跑

            hits = self._left.rank(query, final_ids, top_k, speaker_filter=speaker_filter)
            t3 = time.time()

            rb_hits, rb_directive = rb_future.result()   # 等右脑完成（通常已经跑完了）
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
        """从左脑记忆中提取用户名字，转发到 LeftBrain（共享 _cache 里 "user_name"）。"""
        return self._left._get_user_name()

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

    def _detect_scene(self, *a, **k):
        return self._audio._detect_scene(*a, **k)

    def _detect_speaker(self, *a, **k):
        return self._audio._detect_speaker(*a, **k)

    def _bind_self_identity(self, *a, **k):
        return self._audio._bind_self_identity(*a, **k)

    def preprocess(self, *a, **k) -> "AudioPerception":
        """流式预处理（音频感知），转发到 AudioPerceiver.preprocess。"""
        return self._audio.preprocess(*a, **k)

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
        from voicemem.utils.common.voice_input import VoiceInput, VoiceContent

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

        # 左脑事实抽取 + 入库（ingest_voice_input）抽进 LeftBrain.ingest_facts；
        # registry 是音频侧声纹姓名映射（跨域），由 engine 编排时注入。
        result = self._left.ingest_facts(
            vi,
            registry=self._get_registry(),
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

    def _write_audiomem_tags(self, *a, **k) -> dict:
        """audiomem 写入段，转发到 AudioPerceiver._write_audiomem_tags。"""
        return self._audio._write_audiomem_tags(*a, **k)

    def _write_left_brain(self, result, text) -> None:
        """左脑写入段（LLM 打 slot 标签 + slot→entity 图层），转发到 LeftBrain.write。"""
        return self._left.write(result, text)

    def _write_right_brain(self, emotion, result, text, entities, observed_at) -> None:
        """右脑写入段，转发到 RightBrain.write。"""
        return self._right.write(emotion, result, text, entities, observed_at)

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

    def _refresh_schema_descriptions(self) -> None:
        """给记忆数有变化的 slot 重写一句综合描述，转发到 LeftBrain。"""
        return self._left._refresh_schema_descriptions()

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

    def IngestEnv(self, *a, **k) -> dict:
        """将一段环境音事件存入记忆库，转发到 AudioPerceiver.IngestEnv。"""
        return self._audio.IngestEnv(*a, **k)

    def _check_and_cleanup(self) -> None:
        """每增加 50 条 heartnote 触发一次右脑清洁，转发到 RightBrain.check_and_cleanup。"""
        return self._right.check_and_cleanup()

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
        """用 LLM 清洁右脑 heartnote，转发到 RightBrain.run_cleanup。"""
        return self._right.run_cleanup()


__all__ = ["VoiceMem", "SearchResult"]
