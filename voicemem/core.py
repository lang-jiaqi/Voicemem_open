"""voicemem 顶层门面：一个类 VoiceMem = 左脑 + 右脑 + 音频感知 + 一组可换的能力(utils)。

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

本文件只是「暴露给别人理解系统」的入口：VoiceMem 是一层薄门面，面向用户的小写
便捷方法（ingest/search/classify/preprocess/flush/test）各自一行委托给内部持有的
一个 Orchestrator 实例（self._o）。真正的整条 pipeline（Search/Ingest 编排、工具
方法、向左右脑/音频三组件的转发、SearchResult / Utils）都藏在 orchestrator.py。

想按老写法直接调大写编排方法（vm.Search / vm.Ingest / vm.Classify / vm.Flush …）
或访问内部转发方法也可以：门面的 __getattr__ 会透明地转到 self._o。
"""

from __future__ import annotations

from pathlib import Path

from voicemem.orchestrator import Orchestrator, SearchResult, Utils

# SearchResult / Utils 经此 re-export，保持 `from voicemem.core import ...` 与
# `from voicemem import SearchResult, Utils` 可用。
__all__ = ["VoiceMem", "SearchResult", "Utils"]


class VoiceMem:
    """顶层门面：左脑 + 右脑 + utils（一眼看懂系统）。实现见 orchestrator.py。

    面向用户的小写便捷方法各自一行委托给内部 ``self._o``（一个 ``Orchestrator``
    实例）；``left_brain`` / ``right_brain`` / ``utils`` 直接指向真组件。构造参数原样
    透传给 ``Orchestrator``：``mode`` 走 mode，能力覆盖（``embedding`` / ``schema`` /
    ``memory_engine`` 等）与 ``enable_*`` / ``embedder`` / ``vector_store`` / ``classifier``
    都经 ``**kw`` 透传，语义与 ``Orchestrator.__init__`` 完全一致。
    """

    def __init__(self, api_key=None, mode="text_mode", memory_root=None,
                 user_id="voice_user", base_url=None, **kw):
        self._o = Orchestrator(api_key=api_key, mode=mode, memory_root=memory_root,
                               user_id=user_id, base_url=base_url, **kw)
        self.mode = self._o.mode
        self.utils = self._o.utils
        self.left_brain = self._o._left      # 真组件
        self.right_brain = self._o._right

    @classmethod
    def from_config(cls, config: dict) -> "VoiceMem":
        """声明式构造：一个统一 config dict 配齐所有本地/api 模型（仿 mem0）。

        每个组件写成 ``{"provider": ..., "config": {...}}``，打开一个 dict 就知道
        每个模型走本地还是 api。这是在现有 ``VoiceMem(embedding=fn, schema=fn, …)``
        注入机制之上的一层糖，现有构造方式照常工作。provider 映射表见
        ``voicemem.config``。::

            vm = VoiceMem.from_config({
                "mode": "multi_modal",
                "embedding": {"provider": "local"},   # 记忆向量走本地 E5
                "slots":     {"provider": "local"},   # slot 分类走本地 E5（0 LLM）
                "llm": {"provider": "openai", "config": {"model": "gpt-4o-mini"}},
            })
        """
        from voicemem.config import build_kwargs
        return cls(**build_kwargs(config))

    # ── 面向用户的便捷方法（各自一行委托给 Orchestrator）─────────────────────────

    def ingest(self, text, audio=None, **kw): return self._o.Ingest(text, audio_path=audio, **kw)
    def search(self, query, **kw):            return self._o.Search(query, **kw)
    def classify(self, query):                return self._o.Classify(query)
    def preprocess(self, text, audio=None):   return self._o.preprocess(text, audio_path=audio)
    def flush(self):                          return self._o.Flush()

    def test(self):
        """启动自检：只测本 mode 需要的 util，打印 4 档速度表。"""
        from voicemem.startup_check import run_util_report
        return run_util_report(self.utils)

    def __getattr__(self, name):
        # 老写法的大写编排方法（Search/Ingest/Classify/Flush…）与内部转发方法
        # 透明落到编排实现上；__getattr__ 只在常规属性查找失败时触发，故 self._o /
        # utils / left_brain / right_brain 这些已设的属性不会走到这里。
        return getattr(self.__dict__["_o"], name)
