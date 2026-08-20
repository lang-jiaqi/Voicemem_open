#!/usr/bin/env python3
"""voicemem 主业务逻辑：实时语音对话一轮。

麦克风（或一个 wav 文件）逐块喂进 ``vm.stream()``，说完一轮就拿到 ``Turn``——
**记忆早在你说话时就查好了**，关键路径上只剩回复模型。
``stream.feed()`` 里发生的事（``voicemem/stream.py``）：

    说话中、文本 ≥6 字   后台起投机 Search（本地 E5，0 LLM 0 网络）；文本每变一次就重起
    静音 200ms          赌你说完了，补投机一次
    停顿后又开口         barge-in：取消投机，不误判成一轮
    静音 500ms          ASR flush 补尾字 → 交出 Turn（.text / .result / .memory_context）

装：
    pip install -e ".[all]"                     # 麦克风模式再加 sounddevice
    bash scripts/download_models.sh models      # silero VAD 必需
    export OPENAI_API_KEY=sk-...                # 回复 + 存记忆的事实抽取用

跑：
    python example.py                                   # 麦克风，Ctrl-C 退出
    python example.py --audio speech.wav                # 喂一个 wav，没麦克风也能跑
    python example.py --audio speech.wav --no-reply     # 只看记忆检索，不调回复模型

手头没 wav？macOS 上一行造一个：
    say -v Tingting -o t.aiff "我是素食主义者，对坚果过敏。" \\
      && afconvert -f WAVE -d LEI16@16000 -c 1 t.aiff speech.wav

换成自己的回复模型：见 README「回复：两条路」，或 scripts/realtime_funasr_qwen.py。
"""
import argparse
import asyncio
import queue
import sys

import numpy as np

from voicemem import VoiceMem

SR = 16000


def build(args):
    vm = VoiceMem.from_config({                  # 检索侧全本地 → 投机预取 0 网络
        "embedding": {"provider": "local"},
        "slots":     {"provider": "local"},
    })
    stream = vm.stream(
        on_partial=lambda t: print(f"\r🎙️  {t}", end="", flush=True),
        src_rate=SR,
        spec_min_chars=args.spec_min_chars,
        gamble_s=args.gamble_ms / 1000,
        confirm_s=args.confirm_ms / 1000,
    )
    return vm, stream


async def on_turn(vm, turn, args):
    """一轮说完：记忆早在说话期间预取好了，这里只剩生成 + 存。"""
    print(f"\n🧑 {turn.text}")
    for hit in turn.result.hits:
        print(f"   记忆: {hit.text}")
    if not args.no_reply:
        print(f"🤖 {await vm.reply(turn)}")
    vm.ingest(turn.text, async_facts=True)       # 存这轮，抽事实走后台
    print(flush=True)


async def from_wav(vm, stream, args):
    import soundfile as sf

    audio, sr = sf.read(args.audio, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]                      # 取单声道
    stream.src_rate = sr                         # 非 16k 会自动重采样
    pcm16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    step = max(1, int(sr * args.step_ms / 1000))
    # 文件常常在话音未落时就结束了，后面没有静音 → VAD 等不到「说完」，最后一轮永远
    # 交不出来。补一段够长的静音把它逼出来（麦克风模式不需要，你不说话自然就有静音）。
    # 静音是按「块」累加的，所以尾巴要够 confirm_ms 再加两块，否则最后半块凑不满阈值。
    pcm16 = np.concatenate([
        pcm16, np.zeros(int(sr * args.confirm_ms / 1000) + 2 * step, dtype=np.int16)])
    print(f"▶ {args.audio}  {len(audio)/sr:.1f}s @ {sr}Hz，{args.step_ms}ms 一块", flush=True)
    for i in range(0, len(pcm16), step):
        st = await stream.feed(pcm16[i:i + step].tobytes())
        if st.turn:
            await on_turn(vm, st.turn, args)


async def from_mic(vm, stream, args):
    import sounddevice as sd

    mic_q: queue.Queue = queue.Queue()           # callback 只丢数据，不被生成阻塞
    def on_mic(indata, *_):
        mic_q.put((np.clip(indata[:, 0], -1, 1) * 32767).astype(np.int16).tobytes())

    loop = asyncio.get_running_loop()
    frame = max(1, int(SR * args.step_ms / 1000))
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                        blocksize=frame, callback=on_mic):
        print("🎙️  开始说话…（Ctrl-C 退出）", flush=True)
        while True:
            st = await stream.feed(await loop.run_in_executor(None, mic_q.get))
            if st.turn:
                await on_turn(vm, st.turn, args)


def parse():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audio", help="喂一个 wav 文件；不给就读麦克风")
    p.add_argument("--step_ms", type=int, default=20, help="一块多少毫秒（麦克风默认 20）")
    p.add_argument("--spec_min_chars", type=int, default=6, help="转写到几个字起投机预取")
    p.add_argument("--gamble_ms", type=int, default=200, help="静音多久就赌你说完了，补投机一次")
    p.add_argument("--confirm_ms", type=int, default=500, help="静音多久确认一轮结束，交出 Turn")
    p.add_argument("--no-reply", action="store_true", help="只出记忆检索结果，不调回复模型")
    args = p.parse_args()
    if args.audio and args.step_ms == 20:
        args.step_ms = 600                       # 喂文件时按 600ms 一块更省事
    return args


if __name__ == "__main__":
    args = parse()
    if not args.audio:
        try:
            import sounddevice  # noqa: F401
        except ImportError:
            sys.exit("读麦克风要 sounddevice：pip install sounddevice\n"
                     "或者用 --audio speech.wav 喂一个文件。")
    vm, stream = build(args)
    vm.classify("你好")                          # 预热本地 E5，第一轮就快
    runner = from_wav if args.audio else from_mic
    try:
        asyncio.run(runner(vm, stream, args))
    except KeyboardInterrupt:
        print("\n👋 bye")
