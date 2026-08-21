#!/usr/bin/env python3
"""Plug in your own ASR / VAD. Two ways, pick by what your ASR looks like.

Way 1 — swap the components: VoiceMem(asr=..., vad=...)
    For an ASR that eats audio chunk by chunk and can report the transcript so far.
    You still feed audio with stream.feed(pcm).

    asr: reset() / feed(samples) -> str (cumulative) / flush() -> str (optional)
    vad: is_speech(samples) -> bool
    samples is float32, 16 kHz, mono, in [-1, 1].
    Both are passed as zero-arg factories, not instances.

Way 2 — skip audio entirely: stream.feed_partial(text, ended=...)
    For an ASR/VAD already running elsewhere (cloud, OS, another process).
    No audio model is ever loaded on this path.

    python examples/05_custom_asr_vad.py
"""
import asyncio
import os

import numpy as np

from voicemem import VoiceMem


class MyVAD:
    """Energy VAD — short enough to show the protocol, real enough to work."""

    def __init__(self, threshold=0.01):
        self.threshold = threshold

    def is_speech(self, samples) -> bool:
        return float(np.sqrt(np.mean(samples ** 2))) > self.threshold


class MyASR:
    """Wrap your streaming ASR in this shape (whisper, paraformer, cloud, ...).

    Here feed() just reveals a fixed sentence over time so the example runs
    offline; in a real one it would push samples through your model.
    """

    SCRIPT = "I'm allergic to nuts"

    def reset(self) -> None:
        self._n = 0

    def feed(self, samples) -> str:
        self._n += len(samples)
        return self.SCRIPT[:min(len(self.SCRIPT), self._n // 2000)]

    def flush(self) -> str:
        return self.SCRIPT


async def way_1_swap_components():
    print("-- way 1: VoiceMem(asr=..., vad=...) --")
    vm = VoiceMem(mode="leftbrain_only", openai_key=os.environ["OPENAI_API_KEY"],
                  asr=lambda: MyASR(), vad=lambda: MyVAD())
    stream = vm.stream(src_rate=16000)

    # 1.5s of speech then 1s of silence, so the VAD decides the turn is over
    speech = (np.random.randn(int(16000 * 1.5)) * 0.1).astype(np.float32)
    pcm = np.concatenate([speech, np.zeros(16000, np.float32)])
    pcm16 = (np.clip(pcm, -1, 1) * 32767).astype(np.int16)

    step = 1600
    for i in range(0, len(pcm16), step):
        st = await stream.feed(pcm16[i:i + step].tobytes())
        if st.turn:
            print(f"   turn: {st.turn.text!r}  hits: {len(st.turn.result.hits)}")
            return
    print("   no turn — tune vad_threshold / confirm_s")


async def way_2_text_only():
    print("\n-- way 2: stream.feed_partial(text, ended=...) --")
    vm = VoiceMem(mode="leftbrain_only", openai_key=os.environ["OPENAI_API_KEY"])
    stream = vm.stream()

    for partial in ["I'm", "I'm allergic", "I'm allergic to nuts"]:
        st = await stream.feed_partial(partial)
        print(f"   partial={partial!r:24} state={st.state}")

    st = await stream.feed_partial("I'm allergic to nuts", ended=True)  # your VAD said so
    print(f"   turn: {st.turn.text!r}  hits: {len(st.turn.result.hits)}")


async def main():
    await way_1_swap_components()
    await way_2_text_only()


if __name__ == "__main__":
    asyncio.run(main())
