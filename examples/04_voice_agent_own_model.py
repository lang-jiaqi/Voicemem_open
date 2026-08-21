#!/usr/bin/env python3
"""Same as 03, but replies with your own fine-tuned Qwen adapter instead of OpenAI.

The memory half is identical. Only the reply function changes: any
``(text, memory_context)`` callable can be the reply provider -- returning a string,
or yielding deltas as this one does so speech can start before generation ends.

    pip install funasr sounddevice transformers peft torch
    export OPENAI_API_KEY=sk-...     # write path only; retrieval is fully local
    python examples/04_voice_agent_own_model.py

Use the sherpa ASR instead:
    VOICEMEM_ASR=sherpa python examples/04_voice_agent_own_model.py

Offline speech instead of the OpenAI voice:
    pip install piper-tts
    export TTS_BACKEND=local VOICEMEM_TTS_MODEL=models/tts/<voice>.onnx
"""
import asyncio
import os
import queue
import sys
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("VOICEMEM_MODELS_DIR", str(ROOT / "models"))

from voicemem import VoiceMem  # noqa: E402
from voicemem.tts import SAMPLE_RATE, speak_stream  # noqa: E402

SR, FRAME = 16000, 320

# 35B in bf16 needs ~70GB. If that is too much, load the base with
# BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4") — which also
# matches how the adapter was trained.
BASE = os.environ.get("VOICEMEM_REPLY_BASE", "Qwen/Qwen3.6-35B-A3B")
ADAPTER = os.environ.get("VOICEMEM_REPLY_ADAPTER", "")
if not ADAPTER:
    local = ROOT / "models" / "reply_adapter"
    ADAPTER = str(local) if (local / "adapter_config.json").exists() \
        else "zhifeixie/VoiceMem_SLM_Qwen25_omni"

tok = AutoTokenizer.from_pretrained(BASE)
base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(base, ADAPTER).eval()


async def our_model_reply(text: str, memory_context: str = ""):
    """Yields text deltas. generate() runs on its own thread and pushes into the
    streamer; we drain it off-loop so the mic thread is never blocked."""
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": memory_context}, {"role": "user", "content": text}],
        tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)

    def generate():
        with torch.inference_mode():
            model.generate(**ids, streamer=streamer, max_new_tokens=200,
                           do_sample=True, temperature=0.7)

    threading.Thread(target=generate, daemon=True).start()
    loop = asyncio.get_running_loop()
    while (delta := await loop.run_in_executor(None, next, streamer, None)) is not None:
        yield delta


vm = VoiceMem.from_config({
    "mode": "text_mode",
    "embedding": {"provider": "local"},   # retrieval must not hit the network:
    "slots": {"provider": "local"},       # it runs inside the speculation budget
    "reply": {"provider": "custom", "config": {"fn": our_model_reply}},
})
stream = vm.stream(src_rate=SR, on_partial=lambda t: print(f"\r{t}", end="", flush=True))

mic_q: queue.Queue = queue.Queue()


def _mic_cb(indata, frames, time_info, status):
    if status:
        print(f"\n[mic] {status}", flush=True)
    mic_q.put((np.clip(indata[:, 0], -1, 1) * 32767).astype(np.int16).tobytes())


async def main():
    loop = asyncio.get_running_loop()
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                        blocksize=FRAME, callback=_mic_cb), \
         sd.RawOutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16") as spk:
        print("listening... (Ctrl-C to quit)", flush=True)
        while True:
            st = await stream.feed(await loop.run_in_executor(None, mic_q.get))
            if not st.turn:
                continue
            turn = st.turn
            print(f"\nyou: {turn.text}\nbot: ", end="", flush=True)
            async for pcm in speak_stream(vm.reply_stream(turn),
                                          on_delta=lambda d: print(d, end="", flush=True)):
                await loop.run_in_executor(None, spk.write, pcm)
            threading.Thread(target=lambda: vm.ingest(turn.text, async_facts=True),
                             daemon=True).start()
            while not mic_q.empty():                # drop what the mic heard us say
                mic_q.get()
            print(flush=True)


if __name__ == "__main__":
    vm.classify("hello")          # warm up the local E5 so the first turn is fast
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nbye")
