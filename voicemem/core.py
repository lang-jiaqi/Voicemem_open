"""voicemem 顶层门面：一个类 VoiceMem = 左脑 + 右脑 + 一组可换的能力(utils)。

    左脑  事实记忆：实体 + 认知图（slot 分类/检索），底层 mem0 向量库
    右脑  情绪记忆：每轮 valence-arousal、情绪归因、人格画像
    utils 可插拔能力：embedding / schema(分类) / entity / emotion / voiceprint / asr / memory_engine
          每个都有内置默认，传一个函数就换成自己的（本地模型、别的向量库…）

    vm = VoiceMem(api_key="sk-...", mode="text_mode")
    vm.ingest("中午和 Alex 吃了拉面")
    vm.search("我中午吃了什么")                    # 左右脑一起检索
    vm.left_brain.search(...) / vm.right_brain.search(...)
    VoiceMem(embedding=lambda: MyE(), schema=lambda: MyClassifier())   # 换掉某个能力

mode 决定加载哪些能力：left_brain_single / text_mode / multi_modal(带音频)。
重逻辑在 engine.py；这里只做编排 + 依赖注入 + 按 mode 懒加载。
"""
from __future__ import annotations

import os

from voicemem.engine import VoiceMem as _Engine
from voicemem.utils.defaults import default_utils

# 每个 mode 需要哪些 util（只加载这些）
_NEED = {
    "left_brain_single": ["embedding", "schema", "entity", "memory_engine"],
    "text_mode":         ["embedding", "schema", "entity", "emotion", "memory_engine"],
    "multi_modal":       ["embedding", "schema", "entity", "emotion", "voiceprint", "asr", "memory_engine"],
}


class Utils:
    """能力表：内置默认(见 utils/defaults.py) + 用户覆盖，按需懒加载并缓存。"""
    def __init__(self, mode, base_url, memory_root, overrides):
        self._factory = {**default_utils(base_url, memory_root), **overrides}
        self.need = _NEED[mode]
        self._cache = {}
    def get(self, name):
        if name not in self._cache:
            self._cache[name] = self._factory[name]()
        return self._cache[name]


class LeftBrain:
    """左脑=事实记忆。search 返回左脑命中；store 走完整入库。"""
    def __init__(self, engine): self._e = engine
    def search(self, query, **kw): return self._e.Search(query, **kw).hits
    def store(self, text, **kw):   return self._e.Ingest(text, **kw)


class RightBrain:
    """右脑=情绪/经历/画像。search 返回右脑命中；store 走完整入库。"""
    def __init__(self, engine): self._e = engine
    def search(self, query, **kw): return self._e.Search(query, **kw).rb_hits
    def store(self, text, emotion="", **kw): return self._e.Ingest(text, emotion=emotion, **kw)


class VoiceMem:
    """顶层入口 = 左脑 + 右脑 + utils。util 只在用到时加载。"""
    def __init__(self, api_key=None, mode="text_mode", memory_root=None,
                 user_id="voice_user", base_url=None, **util_overrides):
        if mode not in _NEED:
            raise ValueError("mode 必须是 " + " / ".join(_NEED))
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        self.mode = mode
        self.utils = Utils(mode, base_url, memory_root, util_overrides)

        audio = mode == "multi_modal"
        # 只有被用户覆盖的能力才注入引擎（embedding/memory_engine/schema）；否则引擎用自己的默认
        pick = lambda n: self.utils.get(n) if n in util_overrides else None
        self._engine = _Engine(
            memory_root=memory_root, user_id=user_id, base_url=base_url,
            enable_scene=audio, enable_music=audio, enable_abnormal_sound=audio,
            enable_voiceprint=audio, enable_emotion=(mode != "left_brain_single"),
            embedder=pick("embedding"), vector_store=pick("memory_engine"), classifier=pick("schema"),
        )
        self.left_brain = LeftBrain(self._engine)
        self.right_brain = RightBrain(self._engine)

    # 高层便捷方法（委托引擎）
    def ingest(self, text, audio=None, **kw): return self._engine.Ingest(text, audio_path=audio, **kw)
    def search(self, query, **kw):            return self._engine.Search(query, **kw)
    def classify(self, query):                return self._engine.Classify(query)
    def preprocess(self, text, audio=None):   return self._engine.preprocess(text, audio_path=audio)
    def flush(self):                          return self._engine.Flush()

    def test(self):
        """启动自检：只测本 mode 需要的 util，打印 4 档速度表。"""
        from voicemem.startup_check import run_util_report
        return run_util_report(self.utils)
