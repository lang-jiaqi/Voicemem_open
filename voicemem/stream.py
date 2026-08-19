"""voicemem 核心流式输入会话：边听边投机预取（EOU 0–500ms）。

和「文本」「wav」并列的第三种输入途径。两种喂法，每块都返回一个 ``StreamState``
（这块 ``<speak>``/``<silence>`` + 当前投机预取的记忆 + 说完一轮时的 ``Turn``）：

    stream = vm.stream(on_partial=lambda t: print(t))

    # ① 喂音频块：voicemem 自带 sherpa ASR + silero VAD
    st = await stream.feed(pcm_bytes)          # PCM16 @ src_rate（默认 24k）

    # ② 喂【外部 ASR】的 partial 文本（FunASR / Whisper / 任意）——换 ASR 只改喂进来的这行
    st = await stream.feed_partial(text, ended=is_final)

    st.state    # "<speak>" | "<silence>"
    st.memory   # 当前投机预取的记忆（SearchResult）；边说边有，没算好时 None
    st.turn     # 一轮说完才有 Turn（否则 None）

**只到记忆结果**——回复（tts/realtime）由调用方拿到 Turn/memory 后自理，核心不碰。
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


@dataclass
class StreamState:
    """每喂一块（音频或外部 ASR 文本）返回：这块静音/说话 + 当前投机记忆 + 说完了没。"""
    state: str                 # "<speak>" | "<silence>"
    text: str                  # 到目前为止的累积转写
    memory: object | None      # 当前投机预取到的记忆（SearchResult）；没算好时 None
    turn: Turn | None          # 一轮说完才有，否则 None

    @property
    def memory_context(self) -> str:
        m = self.turn.result if self.turn else self.memory
        return build_memory_context(m) if m is not None else ""


class VoiceStream:
    """核心流式输入会话：边喂边投机，说完时交出 Turn。

    ``vm.stream(on_partial=None, spec_min_chars=6, gamble_s=0.2, confirm_s=0.5,
    src_rate=24000)``。``feed`` 走内置 sherpa ASR + silero VAD；``feed_partial``
    接外部 ASR 的文本（换 ASR 只改喂进来的一行）。投机预取逻辑两者共用。
    """

    def __init__(self, vm, *, on_partial=None, spec_min_chars=6,
                 gamble_s=0.2, confirm_s=0.5, src_rate=24000):
        self.vm = vm
        self.on_partial = on_partial
        self.spec_min_chars = spec_min_chars
        self.gamble_s = gamble_s
        self.confirm_s = confirm_s
        self.src_rate = src_rate
        # ASR/VAD 懒加载：feed_text / feed_partial（外部 ASR）不碰音频模型。
        self._asr = None
        self._vad = None
        # 回合状态
        self._text = ""
        self._silence = 0.0
        self._spoke = False
        self._spec = None
        self._spec_text = ""
        self._last_memory = None   # 最新算好的投机记忆（SearchResult）

    @property
    def asr(self):
        if self._asr is None:
            self._asr = self.vm.utils.get("asr"); self._asr.reset()
        return self._asr

    @property
    def vad(self):
        if self._vad is None:
            self._vad = make_vad()
        return self._vad

    # ── 投机预取（本地分类器 + 本地向量 Search，0 LLM/网络，放线程里跟读麦克风并发）──
    async def _speculate(self, text) -> Turn:
        t0 = time.time()

        def work():
            c = self.vm.classify(text)
            return self.vm.search(text, slots=c.slots, entities=c.entities)

        result = await asyncio.to_thread(work)
        print(f"[speculate] {text[:24]!r} -> {len(result.hits)} hits  "
              f"{(time.time()-t0)*1000:.0f}ms", flush=True)
        return Turn(text, result)

    def _kick(self, text):
        """文本够长且变化了就（重）起后台投机。"""
        if text and text != self._spec_text and len(text) >= self.spec_min_chars:
            if self._spec:
                self._spec.cancel()
            self._spec_text = text
            self._spec = asyncio.create_task(self._speculate(text))

    def _ready_memory(self):
        """取最新算好的投机记忆（SearchResult）；没算好就保持上一份/None。"""
        if self._spec is not None and self._spec.done() and not self._spec.cancelled():
            try:
                self._last_memory = self._spec.result().result
            except Exception:
                pass
        return self._last_memory

    async def _confirm(self) -> Turn:
        try:
            return await (self._spec or self._speculate(self._text))
        except asyncio.CancelledError:
            return await self._speculate(self._text)

    def _reset_turn(self):
        if self._asr is not None:
            self._asr.reset()
        self._text, self._silence, self._spoke = "", 0.0, False
        self._spec, self._spec_text, self._last_memory = None, "", None

    async def feed_text(self, text) -> Turn:
        """打字轮：直接投机一次返回 Turn。"""
        return await self._speculate(text)

    async def feed_partial(self, text, ended: bool = False) -> StreamState:
        """接【外部 ASR】的 partial 文本（FunASR / Whisper / 任意流式 ASR）。

        换 ASR 只改「喂进来的这行文本」，本方法一字不用改。text 有新内容 = ``<speak>``
        并（重）起投机；``ended=True``（外部 VAD 判一句说完）→ 交出 Turn。
        """
        text = (text or "").strip()
        new = bool(text) and text != self._text
        if text:
            self._text = text
        if new and self.on_partial:
            self.on_partial(self._text)
        self._kick(self._text)
        if ended and self._text:
            turn = await self._confirm()
            self._reset_turn()
            return StreamState("<speak>" if new else "<silence>", turn.text, None, turn)
        return StreamState("<speak>" if new else "<silence>", self._text, self._ready_memory(), None)

    async def feed(self, pcm_bytes) -> StreamState:
        """喂一块 PCM16（``src_rate``，默认 24k）：内置 sherpa ASR + silero VAD + 投机。
        每块返回 ``StreamState``（``<speak>``/``<silence>`` + 当前投机记忆 + 说完时的 Turn）。
        """
        frame = resample(np.frombuffer(pcm_bytes, np.int16).astype(np.float32) / 32768.0,
                         src=self.src_rate)
        self._text = self.asr.feed(frame)
        speaking = self.vad.is_speech(frame)
        if speaking:
            if self._silence > 0 and self._spec:           # barge-in：又开口了 → 丢弃这次投机
                self._spec.cancel(); self._spec, self._spec_text = None, ""
            self._spoke, self._silence = True, 0.0
        else:
            self._silence += len(frame) / 16000.0
        if self._text.strip() and self.on_partial:
            self.on_partial(self._text)
        # 边说边预取 / 200ms 赌说完补发
        if self._spoke and self._text.strip() and \
                (self._silence == 0.0 and len(self._text) >= self.spec_min_chars
                 or self._silence >= self.gamble_s):
            self._kick(self._text)
        if self._spoke and self._silence >= self.confirm_s and self._text.strip():
            turn = await self._confirm()                   # VAD 确认说完 → 交出预算记忆
            self._reset_turn()
            return StreamState("<speak>" if speaking else "<silence>", turn.text, None, turn)
        return StreamState("<speak>" if speaking else "<silence>", self._text, self._ready_memory(), None)
