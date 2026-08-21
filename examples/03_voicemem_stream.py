#!/usr/bin/env python3
"""Mic -> reply with memory -> store. The smallest voice agent.

Only generation is on the critical path: memory is prefetched speculatively while
you are still speaking, so turn.result is already there when the turn ends.

    pip install -e ".[all]" sounddevice
    bash scripts/download_models.sh
    export OPENAI_API_KEY=sk-...
    python examples/03_voice_agent.py
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
                   on_partial=lambda t: print(f"\r{t}", end="", flush=True))


async def main():
    mic = queue.Queue()          # callback only enqueues, never blocks on generation
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=320,
                        callback=lambda d, *_: mic.put(
                            (np.clip(d[:, 0], -1, 1) * 32767).astype(np.int16).tobytes())):
        print("listening... (Ctrl-C to quit)", flush=True)
        loop = asyncio.get_running_loop()
        while True:
            st = await stream.feed(await loop.run_in_executor(None, mic.get))
            if not st.turn:
                continue
            turn = st.turn
            print(f"\nyou: {turn.text}")
            print(f"bot: {await vm.reply(turn)}")   # turn carries the prefetched context
            vm.ingest(turn.text, async_facts=True)  # the reply is stored alongside it
            print(flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nbye")
