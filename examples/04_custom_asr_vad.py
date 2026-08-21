#!/usr/bin/env python3
"""换成自己的 ASR / VAD。两条路，按你的 ASR 是什么形态选。

路 1  组件替换：VoiceMem(asr=..., vad=...)
      你的 ASR 能一块块吃 16k 音频、随时给出「到目前为止的文本」时用。
      喂法不变，还是 stream.feed(pcm)。

      asr 协议:  reset() / feed(samples) -> str（累积文本）/ flush() -> str（可选）
      vad 协议:  is_speech(samples) -> bool
      samples 是 np.float32、16kHz、单声道、[-1, 1]。
      两个都传**零参工厂**（VoiceMem 懒加载时才调），不是实例。

路 2  完全绕开音频：stream.feed_partial(text, ended=...)
      你的 ASR/VAD 已经在别处跑了（云端 ASR、系统自带、另一个进程），
      voicemem 只收文本。ended=True 表示你的 VAD 判定这句说完了。

    python examples/04_custom_asr_vad.py       # 两条路各跑一遍
"""
import asyncio
import os

import numpy as np

from voicemem import VoiceMem

# ── 路 1：自己的 ASR / VAD ────────────────────────────────────────────────────


class MyVAD:
    """能量 VAD——真能用，够短到看清协议长什么样。换成 silero/webrtc 同理。"""

    def __init__(self, threshold=0.01):
        self.threshold = threshold

    def is_speech(self, samples) -> bool:
        return float(np.sqrt(np.mean(samples ** 2))) > self.threshold


class MyASR:
    """把你的流式 ASR 包成这个形状即可（whisper / paraformer / 云端都行）。

    这里为了能离线跑通，用一段固定文本按喂进来的时长逐字吐出来，
    真实现里 feed() 就是把 samples 送进你的模型、返回它的累积转写。
    """

    SCRIPT = "我对坚果过敏"

    def reset(self) -> None:
        self._n = 0

    def feed(self, samples) -> str:
        self._n += len(samples)
        chars = min(len(self.SCRIPT), self._n // 4000)      # 每 0.25s 出一个字
        return self.SCRIPT[:chars]

    def flush(self) -> str:                                  # 可选：收尾补字
        return self.SCRIPT


async def route_1_custom_components():
    print("── 路 1：VoiceMem(asr=…, vad=…) ──")
    vm = VoiceMem(mode="leftbrain_only", openai_key=os.environ["OPENAI_API_KEY"],
                  asr=lambda: MyASR(), vad=lambda: MyVAD())   # 零参工厂
    stream = vm.stream(src_rate=16000)

    # 造一段「1.5s 说话 + 1s 静音」的音频喂进去，让 VAD 自己判说完
    speech = (np.random.randn(int(16000 * 1.5)) * 0.1).astype(np.float32)
    pcm = np.concatenate([speech, np.zeros(16000, np.float32)])
    pcm16 = (np.clip(pcm, -1, 1) * 32767).astype(np.int16)

    step = int(16000 * 0.1)
    for i in range(0, len(pcm16), step):
        st = await stream.feed(pcm16[i:i + step].tobytes())
        if st.turn:
            print(f"   一轮说完: {st.turn.text!r}")
            print(f"   记忆条数: {len(st.turn.result.hits)}")
            return
    print("   （没触发 turn_over，调 vad_threshold / confirm_s）")


# ── 路 2：ASR/VAD 都在外面，只喂文本 ──────────────────────────────────────────


async def route_2_feed_text_only():
    print("\n── 路 2：stream.feed_partial(text, ended=…) ──")
    vm = VoiceMem(mode="leftbrain_only", openai_key=os.environ["OPENAI_API_KEY"])
    stream = vm.stream()                       # 不碰音频，ASR/VAD 模型都不会加载

    # 你的外部 ASR 吐出的 partial 序列
    for partial in ["我对", "我对坚果", "我对坚果过敏"]:
        st = await stream.feed_partial(partial)
        print(f"   partial={partial!r:16} state={st.state}")

    st = await stream.feed_partial("我对坚果过敏", ended=True)   # 你的 VAD 说：完了
    print(f"   一轮说完: {st.turn.text!r}")
    print(f"   记忆条数: {len(st.turn.result.hits)}")


async def main():
    await route_1_custom_components()
    await route_2_feed_text_only()


if __name__ == "__main__":
    asyncio.run(main())
