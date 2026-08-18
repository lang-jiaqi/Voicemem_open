"""Voicemem：左右脑分离的记忆框架，带音频原生感知层。

本文件就是一张「组件目录」——voicemem 的每一个组件都在这里对外暴露，
`from voicemem_core import X` 即可拿到。音频类组件（SpeakerEncoder /
ASTEnvironmentDetector 等）依赖 torch / sherpa-onnx，所以采用惰性加载
（PEP 562 `__getattr__`）：只有真正访问到某个名字时才 import 它对应的模块，
`import voicemem_core` 本身永远不会拉起这些重依赖——纯文本安装照样能用核心。

分组一览：
  · 核心          VoiceMem / SearchResult / RightBrainHit / AudioPerception
  · 语音接入适配   VoiceInput / VoiceContent / VoiceprintRegistry / ingest_voice_input …
  · 音频原生感知   SpeakerEncoder / *EnvironmentDetector / Scene* / Music/Place/Routine …
  · 右脑情绪层     EmotionLayer / EmotionLayerConfig / EmotionLayerResult
  · 便捷记忆 API    Memory / inject / recall / remember
  · 子包           emotion / leftbrain / rightbrain / fusion / persona / prestimulus / voiceprint
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

# ── 名字 → 来源模块 的惰性映射（"模块路径:属性名"）─────────────────────────────
# 每个组件都登记在这里；__getattr__ 在首次访问时按需 import，并缓存进本模块
# 命名空间，之后就是普通属性访问，没有额外开销。
_LAZY: dict[str, str] = {
    # ── 核心 ──
    "VoiceMem":                 "voicemem_core.core:VoiceMem",
    "SearchResult":             "voicemem_core.core:SearchResult",
    "RightBrainHit":            "voicemem_core.core:RightBrainHit",
    "AudioPerception":          "voicemem_core.core:AudioPerception",

    # ── 语音接入适配层（上游语音模块结构化输出 → 左脑注入）──
    "VoiceInput":               "voicemem_core.voice_input:VoiceInput",
    "VoiceContent":             "voicemem_core.voice_input:VoiceContent",
    "VoiceIngestResult":        "voicemem_core.voice_input:VoiceIngestResult",
    "VoiceprintRegistry":       "voicemem_core.voice_input:VoiceprintRegistry",
    "VoiceprintEntry":          "voicemem_core.voice_input:VoiceprintEntry",
    "ingest_voice_input":       "voicemem_core.voice_input:ingest_voice_input",
    "voice_input_to_messages":  "voicemem_core.voice_input:voice_input_to_messages",
    "emotion_to_affect":        "voicemem_core.voice_input:emotion_to_affect",
    "map_voice_slots_to_slotv2":"voicemem_core.voice_input:map_voice_slots_to_slotv2",

    # ── 音频原生感知：说话人声纹 ──
    "SpeakerEncoder":           "voicemem_core.speaker_encoder:SpeakerEncoder",
    "VoiceprintStore":          "voicemem_core.voiceprint_store:VoiceprintStore",
    "IdentifyResult":           "voicemem_core.voiceprint_store:IdentifyResult",
    "parse_self_identification":"voicemem_core.speaker_identity:parse_self_identification",

    # ── 音频原生感知：声学环境 / 场景 ──
    "ASTEnvironmentDetector":   "voicemem_core.environment_detector_ast:ASTEnvironmentDetector",
    "CLAPEnvironmentDetector":  "voicemem_core.environment_detector_clap:CLAPEnvironmentDetector",
    "SceneTag":                 "voicemem_core.scene_classifier:SceneTag",
    "SceneResult":              "voicemem_core.scene_classifier:SceneResult",
    "classify_scene":           "voicemem_core.scene_classifier:classify_scene",
    "scene_to_response_directive":"voicemem_core.scene_classifier:scene_to_response_directive",
    "infer_scene_from_text":    "voicemem_core.scene_classifier:infer_scene_from_text",

    # ── 音频原生感知：场景提醒触发 ──
    "SceneTrigger":             "voicemem_core.scene_trigger:SceneTrigger",
    "SceneTriggerStore":        "voicemem_core.scene_trigger:SceneTriggerStore",
    "TriggerFireResult":        "voicemem_core.scene_trigger:TriggerFireResult",
    "parse_trigger_intent":     "voicemem_core.scene_trigger:parse_trigger_intent",
    "check_and_fire":           "voicemem_core.scene_trigger:check_and_fire",

    # ── 音频原生感知：音乐 / 地点 / 生活规律记忆 ──
    "MusicMemoryStore":         "voicemem_core.music_memory:MusicMemoryStore",
    "TuneIdentifyResult":       "voicemem_core.music_memory:TuneIdentifyResult",
    "PlaceMemoryStore":         "voicemem_core.place_memory:PlaceMemoryStore",
    "PlaceIdentifyResult":      "voicemem_core.place_memory:PlaceIdentifyResult",
    "RoutineStore":             "voicemem_core.routine_memory:RoutineStore",
    "bucket_label":             "voicemem_core.routine_memory:bucket_label",

    # ── 录音归档 / 会话 / 配置 ──
    "AudioArchive":             "voicemem_core.audio_archive:AudioArchive",
    "SessionTracker":           "voicemem_core.session_tracker:SessionTracker",
    "VoiceStoreConfig":         "voicemem_core.voice_config:VoiceStoreConfig",

    # ── 右脑情绪层 ──
    "EmotionLayer":             "voicemem_core.emotion:EmotionLayer",
    "EmotionLayerConfig":       "voicemem_core.emotion:EmotionLayerConfig",
    "EmotionLayerResult":       "voicemem_core.emotion:EmotionLayerResult",

    # ── 启动自检（逐组件测速 + 门控启动）──
    "run_startup_check":        "voicemem_core.startup_check:run_startup_check",
    "check_and_gate":           "voicemem_core.startup_check:check_and_gate",
    "StartupReport":            "voicemem_core.startup_check:StartupReport",

    # ── 便捷记忆 API ──
    "Memory":                   "voicemem_core.memory_api:Memory",
    "build_memory_context":     "voicemem_core.memory_api:build_memory_context",
    "inject":                   "voicemem_core.memory_api:inject",
    "recall":                   "voicemem_core.memory_api:recall",
    "remember":                 "voicemem_core.memory_api:remember",
}

# 子包也作为组件对外暴露：`from voicemem_core import emotion` / `voicemem.leftbrain` …
# 各子包内部的组件由其自身 __init__.__all__ 负责，这里只把包本身列进目录。
_SUBPACKAGES: tuple[str, ...] = (
    "emotion", "leftbrain", "rightbrain", "fusion",
    "persona", "prestimulus", "voiceprint",
)

__all__ = sorted([*_LAZY, *_SUBPACKAGES])


def __getattr__(name: str):
    """PEP 562 惰性属性访问：按需 import 组件，避免 `import voicemem_core` 触发重依赖。"""
    target = _LAZY.get(name)
    if target is not None:
        module_path, _, attr = target.partition(":")
        value = getattr(importlib.import_module(module_path), attr)
        globals()[name] = value          # 缓存，后续走普通属性访问
        return value
    if name in _SUBPACKAGES:
        module = importlib.import_module(f"voicemem_core.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*globals(), *__all__])


# 让静态类型检查器 / IDE 也能看到这些名字（运行期不执行，不触发重依赖）。
if TYPE_CHECKING:  # pragma: no cover
    from voicemem_core.core import (
        AudioPerception, RightBrainHit, SearchResult, VoiceMem,
    )
    from voicemem_core.emotion import (
        EmotionLayer, EmotionLayerConfig, EmotionLayerResult,
    )
