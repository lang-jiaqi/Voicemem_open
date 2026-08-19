"""voicemem 核心流式输入会话：边听边投机预取（EOU 0–500ms）。

和「文本」「wav」并列的第三种输入途径。把 web demo（web/run.py）里验证过的
anticipatory 状态机——本地 StreamingASR + silero VAD、partial 一到就后台起投机
Search、200ms 赌说完补发、500ms VAD 确认结束、说到一半停顿又续上(barge-in)取消
投机——原样提升为核心一等能力。

    vm = VoiceMem(mode="multi_modal", ...)
    stream = vm.stream(on_partial=lambda t: print("...", t))
    turn = await stream.feed(pcm_bytes)      # 喂一块 PCM16(24k)；说完时返回 Turn，否则 None
    if turn:
        print(turn.text, turn.result.hits, turn.memory_context)
    turn = await stream.feed_text("我在哪工作")   # 打字轮：直接投机一次返回 Turn

**只到记忆结果**——回复（tts/realtime）由调用方（web demo）在拿到 Turn 后自理，
核心不碰。
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import numpy as np

from voicemem.memory_api import build_memory_context
from voicemem.utils.audio.stream_io import make_vad, resample


@dataclass
class Turn:
    """一轮说完（或打字）时、投机预取早已算好的记忆结果——调用方拿来直接回复，不再搜。"""
    text: str
    result: object

    @property
    def memory_context(self) -> str:
        return build_memory_context(self.result)


class VoiceStream:
    """核心流式输入会话：本地 ASR+VAD 边听边投机，VAD 确认说完时交出 Turn。

    构造 ``vm.stream(on_partial=None, spec_min_chars=6, gamble_s=0.2,
    confirm_s=0.5, src_rate=24000)``。持有 ``vm``（用它的 Classify/Search）、
    ``StreamingASR``（``vm.utils.get("asr")``）、silero VAD。状态机与 web 的
    ``anticipate`` 完全一致，只把"从 sock 收帧"换成"feed 传入的 pcm_bytes"、
    把 Pending 换成 Turn。
    """

    def __init__(self, vm, *, on_partial=None, spec_min_chars=6,
                 gamble_s=0.2, confirm_s=0.5, src_rate=24000):
        self.vm = vm
        self.on_partial = on_partial
        self.spec_min_chars = spec_min_chars
        self.gamble_s = gamble_s
        self.confirm_s = confirm_s
        self.src_rate = src_rate
        # ASR/VAD 懒加载：打字轮（feed_text）不碰音频模型，第一块 PCM 到来时才拉起。
        self._asr = None
        self._vad = None
        # 回合状态（与 anticipate 里的本地变量一一对应）
        self._text = ""
        self._silence = 0.0
        self._spoke = False
        self._spec = None
        self._spec_text = ""

    @property
    def asr(self):
        if self._asr is None:
            self._asr = self.vm.utils.get("asr")
            self._asr.reset()
        return self._asr

    @property
    def vad(self):
        if self._vad is None:
            self._vad = make_vad()
        return self._vad

    # ── 投机预取（原样搬自 web 的 speculate）─────────────────────────────────────
    async def _speculate(self, text) -> Turn:
        """本地投机检索：注入的本地分类器出 slots(+entities) + 本地 E5 向量 Search
        （0 LLM/网络）。放线程里跑，好和继续读麦克风真正并发（这才叫"边听边预取"）。"""
        t0 = time.time()

        def work():
            c = self.vm.classify(text)
            r = self.vm.search(text, slots=c.slots, entities=c.entities)
            return r

        result = await asyncio.to_thread(work)
        print(f"[speculate] {text[:24]!r} -> {len(result.hits)} hits  "
              f"{(time.time()-t0)*1000:.0f}ms", flush=True)
        return Turn(text, result)

    async def _confirm(self) -> Turn:
        try:
            return await (self._spec or self._speculate(self._text))
        except asyncio.CancelledError:
            return await self._speculate(self._text)

    def _reset_turn(self) -> None:
        self.asr.reset()
        self._text, self._silence, self._spoke = "", 0.0, False
        self._spec, self._spec_text = None, ""

    async def feed_text(self, text) -> Turn:
        """打字轮：直接投机一次返回 Turn。"""
        return await self._speculate(text)

    async def feed(self, pcm_bytes) -> Turn | None:
        """喂一块 PCM16（``src_rate``，默认 24k）。

        内部 resample→16k、``asr.feed``、VAD、和 ``anticipate`` 完全一样的状态机
        （partial 起投机 / 200ms gamble / 500ms confirm / barge-in 取消）；有 partial
        时调 ``on_partial(text)``；VAD 确认说完时返回 ``Turn(text, result)``，否则 None。
        """
        frame = resample(np.frombuffer(pcm_bytes, np.int16).astype(np.float32) / 32768.0,
                         src=self.src_rate)
        self._text = self.asr.feed(frame)
        if self.vad.is_speech(frame):
            if self._silence > 0 and self._spec:          # barge-in：又开口了 → 丢弃这次投机
                self._spec.cancel(); self._spec, self._spec_text = None, ""
            self._spoke, self._silence = True, 0.0
        else:
            self._silence += len(frame) / 16000.0
        if self._text.strip() and self.on_partial:
            self.on_partial(self._text)
        # 边说边预取 / 200ms 赌说完补发
        if self._spoke and self._text.strip() and self._text != self._spec_text and \
                (self._silence == 0.0 and len(self._text) >= self.spec_min_chars
                 or self._silence >= self.gamble_s):
            if self._spec:
                self._spec.cancel()
            self._spec_text = self._text
            self._spec = asyncio.create_task(self._speculate(self._text))
        if self._spoke and self._silence >= self.confirm_s and self._text.strip():
            turn = await self._confirm()               # VAD 确认说完 → 交出预算记忆
            self._reset_turn()
            return turn
        return None
