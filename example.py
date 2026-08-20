#!/usr/bin/env python3
"""voicemem 主业务逻辑：实时语音对话一轮。

麦克风（或一个 wav）逐块喂进 ``vm.stream()``，说完一轮就拿到 ``Turn``——**记忆早在你
说话时就查好了**，关键路径上只剩回复模型。``stream.feed()`` 里发生的事：

    说话中、文本 ≥6 字   后台起投机 Search（本地 E5，0 LLM 0 网络）；文本每变一次就重起
    静音 200ms          赌你说完了，补投机一次
    停顿后又开口         barge-in：取消投机，不误判成一轮
    静音 500ms          ASR flush 补尾字 → 交出 Turn（.text / .result / .memory_context）

    pip install -e ".[all]"                     # 麦克风模式再加 sounddevice
    bash scripts/download_models.sh models      # silero VAD 必需
    export OPENAI_API_KEY=sk-...                # 没有就只出记忆、不生成回复

    python example.py                           # 麦克风，Ctrl-C 退出
    python example.py speech.wav                # 喂一个 wav，没麦克风也能跑

想调阈值：改下面 vm.stream(...) 那一行。想换回复模型：见 README「回复：两条路」。
"""
import asyncio
import os
import queue
import sys

import numpy as np

from voicemem import VoiceMem

SR = 16000
REPLY = bool(os.environ.get("OPENAI_API_KEY"))       # 没 key 就只出记忆，不生成回复

vm = VoiceMem.from_config({                          # 检索侧全本地 → 投机预取 0 网络
    "embedding": {"provider": "local"},
    "slots":     {"provider": "local"},
})
stream = vm.stream(on_partial=lambda t: print(f"\r🎙️  {t}", end="", flush=True),
                   src_rate=SR, spec_min_chars=6, gamble_s=0.20, confirm_s=0.50)


async def on_turn(turn):
    """一轮说完：记忆早在说话期间预取好了，这里只剩生成 + 存。"""
    print(f"\n🧑 {turn.text}")
    for hit in turn.result.hits:
        print(f"   记忆: {hit.text}")
    if REPLY:
        print(f"🤖 {await vm.reply(turn)}")          # 换自己的模型：VoiceMem(reply=fn)
    vm.ingest(turn.text, async_facts=True)           # 存这轮，抽事实走后台
    print(flush=True)


async def from_wav(path):
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32")
    audio = audio[:, 0] if audio.ndim > 1 else audio
    stream.src_rate = sr                             # 非 16k 会自动重采样
    step = int(sr * 0.6)                             # 600ms 一块
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    # 文件常在话音未落时就结束，后面没静音 → VAD 等不到「说完」，最后一轮交不出来。
    # 补一段静音逼它出来；静音按块累加，所以要 confirm_s 再加两块才够阈值。
    pcm = np.concatenate([pcm, np.zeros(int(sr * 0.5) + 2 * step, np.int16)])
    print(f"▶ {path}  {len(audio)/sr:.1f}s @ {sr}Hz", flush=True)
    for i in range(0, len(pcm), step):
        st = await stream.feed(pcm[i:i + step].tobytes())
        if st.turn:
            await on_turn(st.turn)


async def from_mic():
    import sounddevice as sd

    mic_q: queue.Queue = queue.Queue()               # callback 只丢数据，不被生成阻塞
    def on_mic(indata, *_):
        mic_q.put((np.clip(indata[:, 0], -1, 1) * 32767).astype(np.int16).tobytes())

    loop = asyncio.get_running_loop()
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                        blocksize=320, callback=on_mic):          # 20ms 一帧
        print("🎙️  开始说话…（Ctrl-C 退出）", flush=True)
        while True:
            st = await stream.feed(await loop.run_in_executor(None, mic_q.get))
            if st.turn:
                await on_turn(st.turn)


if __name__ == "__main__":
    wav = sys.argv[1] if len(sys.argv) > 1 else None
    if not wav:
        try:
            import sounddevice  # noqa: F401
        except ImportError:
            sys.exit("读麦克风要 sounddevice：pip install sounddevice\n"
                     "或者喂个文件：python example.py speech.wav")
    vm.classify("你好")                              # 预热本地 E5，第一轮就快
    try:
        asyncio.run(from_wav(wav) if wav else from_mic())
    except KeyboardInterrupt:
        print("\n👋 bye")
