#!/usr/bin/env python3
"""最简单的语音对话：麦克风 → 说完一轮 → 带着记忆回答 → 存进记忆。

一轮里只有生成是等出来的——记忆在你还在说的时候就预取好了（``vm.stream()``
内部投机检索，本地 E5，0 LLM 0 网络）。

    pip install -e ".[all]" sounddevice
    bash scripts/download_models.sh
    export OPENAI_API_KEY=sk-...
    python examples/03_voice_chat.py
"""
import asyncio
import os
import queue

import numpy as np
import sounddevice as sd

from voicemem import VoiceMem

SR = 16000

vm = VoiceMem(mode="normal", openai_key=os.environ["OPENAI_API_KEY"])
stream = vm.stream(src_rate=SR,
                   on_partial=lambda t: print(f"\r🎙️  {t}", end="", flush=True))


async def main():
    mic = queue.Queue()                        # 录音回调只丢数据，不被生成阻塞
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=320,
                        callback=lambda d, *_: mic.put(
                            (np.clip(d[:, 0], -1, 1) * 32767).astype(np.int16).tobytes())):
        print("🎙️  开始说话…（Ctrl-C 退出）", flush=True)
        loop = asyncio.get_running_loop()
        while True:
            st = await stream.feed(await loop.run_in_executor(None, mic.get))
            if not st.turn:
                continue
            turn = st.turn
            print(f"\n🧑 {turn.text}")
            print(f"🤖 {await vm.reply(turn)}")     # turn 自带预取好的 memory_context
            vm.ingest(turn.text, async_facts=True)  # 存这轮；agent 刚说的话自动一起存
            print(flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 bye")
