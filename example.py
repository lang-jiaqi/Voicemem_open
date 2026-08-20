#!/usr/bin/env python3
"""voicemem 主业务逻辑：实时语音对话一轮。

麦克风逐帧喂进 ``vm.stream()``，说完一轮就拿到 ``Turn``——**记忆早在你说话时就查好了**，
关键路径上只剩回复模型。``stream.feed()`` 里发生的事（``voicemem/stream.py``）：

    说话中、文本 ≥6 字   后台起投机 Search（本地 E5，0 LLM 0 网络）；文本每变一次就重起
    静音 200ms          赌你说完了，补投机一次
    停顿后又开口         barge-in：取消投机，不误判成一轮
    静音 500ms          ASR flush 补尾字 → 交出 Turn（.text / .result / .memory_context）

装：
    pip install -e ".[all]" sounddevice
    bash scripts/download_models.sh models      # silero VAD 必需
    export OPENAI_API_KEY=sk-...                # 回复 + 存记忆时的事实抽取用

跑：
    python example.py                           # 需要麦克风，Ctrl-C 退出

换成自己的回复模型：见 README「回复：两条路」，或 scripts/realtime_funasr_qwen.py。
"""
import asyncio
import queue
import sys

import numpy as np

from voicemem import VoiceMem

SR, FRAME = 16000, 320                          # 20ms 一帧

vm = VoiceMem.from_config({                     # 检索侧全本地 → 投机预取 0 网络
    "embedding": {"provider": "local"},
    "slots":     {"provider": "local"},
})
stream = vm.stream(on_partial=lambda t: print(f"\r🎙️  {t}", end="", flush=True),
                   src_rate=SR)

mic_q: queue.Queue = queue.Queue()              # callback 只丢数据，不被生成阻塞


def on_mic(indata, *_):
    mic_q.put((np.clip(indata[:, 0], -1, 1) * 32767).astype(np.int16).tobytes())


async def main():
    import sounddevice as sd

    loop = asyncio.get_running_loop()
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                        blocksize=FRAME, callback=on_mic):
        print("🎙️  开始说话…（Ctrl-C 退出）", flush=True)
        while True:
            st = await stream.feed(await loop.run_in_executor(None, mic_q.get))
            if st.turn:                                      # VAD 确认说完
                answer = await vm.reply(st.turn)             # 记忆已就绪，直接生成
                print(f"\n🧑 {st.turn.text}\n🤖 {answer}\n", flush=True)
                vm.ingest(st.turn.text, async_facts=True)    # 存这轮，抽事实走后台


if __name__ == "__main__":
    try:
        import sounddevice  # noqa: F401
    except ImportError:
        sys.exit("需要 sounddevice 读麦克风：pip install sounddevice")
    vm.classify("你好")                          # 预热本地 E5，第一轮就快
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 bye")
