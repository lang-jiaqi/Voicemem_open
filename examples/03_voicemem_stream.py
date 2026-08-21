#!/usr/bin/env python3
"""Mic -> reply with memory -> speak -> store. The smallest voice agent.

Only generation is on the critical path: memory is prefetched speculatively while
you are still speaking, so turn.result is already there when the turn ends. Speech
synthesis runs alongside generation -- the first sentence is already being spoken
while the rest is still being written.

    pip install -e ".[all]" sounddevice
    bash scripts/download_models.sh
    export OPENAI_API_KEY=sk-...
    python examples/03_voice_agent.py

Offline speech instead of the OpenAI voice:

    pip install piper-tts
    export TTS_BACKEND=local VOICEMEM_TTS_MODEL=models/tts/<voice>.onnx
"""
import asyncio
import os
import queue

import numpy as np
import sounddevice as sd

from voicemem import VoiceMem
from voicemem.tts import SAMPLE_RATE, speak_stream

SR = 16000

vm = VoiceMem(mode="normal", openai_key=os.environ["OPENAI_API_KEY"])
stream = vm.stream(src_rate=SR,
                   on_partial=lambda t: print(f"\r{t}", end="", flush=True))


async def main():
    mic = queue.Queue()          # callback only enqueues, never blocks on generation
    loop = asyncio.get_running_loop()
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=320,
                        callback=lambda d, *_: mic.put(
                            (np.clip(d[:, 0], -1, 1) * 32767).astype(np.int16).tobytes())), \
         sd.RawOutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16") as spk:
        print("listening... (Ctrl-C to quit)", flush=True)
        while True:
            st = await stream.feed(await loop.run_in_executor(None, mic.get))
            if not st.turn:
                continue
            turn = st.turn
            print(f"\nyou: {turn.text}\nbot: ", end="", flush=True)
            # turn carries the prefetched context; text is printed as it is written,
            # audio starts as soon as the first sentence is complete.
            async for pcm in speak_stream(vm.reply_stream(turn),
                                          on_delta=lambda d: print(d, end="", flush=True)):
                await loop.run_in_executor(None, spk.write, pcm)
            vm.ingest(turn.text, async_facts=True)  # the reply is stored alongside it
            while not mic.empty():                  # drop what the mic heard us say
                mic.get()
            print(flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nbye")
