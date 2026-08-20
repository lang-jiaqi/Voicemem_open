#!/usr/bin/env python3
"""实时语音对话：麦克风 → voicemem 核心（FunASR 流式 ASR + VAD + 投机预取）→ Voicemem-Qwen 回复。

主业务逻辑全在核心 ``vm.stream()`` 里，这个脚本只做三件事：搬麦克风数据、拿 Turn、
调我们自己的模型生成。核心那条链（``voicemem/stream.py``）：

    FunASR paraformer-zh-streaming  逐块出 partial（内部攒够 600ms 推一次）
      ├─ 说话中且 ≥6 字        → 后台起投机 Search（本地 E5，0 LLM 0 网络）
      ├─ 静音 200ms            → 赌你说完了，再投机一次
      ├─ 停顿后又开口          → barge-in，取消投机、不误判回合
      └─ 静音 500ms            → asr.flush() 补尾字 → 交出 Turn（记忆早已算好）

所以 Search 不在关键路径上：VAD 一确认说完，``turn.result`` 已经是算好的，
唯一还要等的就是回复模型。

装：pip install funasr sounddevice transformers peft torch
用法（仓库根目录）：
    export OPENAI_API_KEY=sk-...        # ingest 的事实抽取用（检索侧已全本地）
    python scripts/realtime_funasr_qwen.py
换回 sherpa 流式 ASR：VOICEMEM_ASR=sherpa python scripts/realtime_funasr_qwen.py
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
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("VOICEMEM_MODELS_DIR", str(ROOT / "models"))

from voicemem import VoiceMem  # noqa: E402

SR    = 16000
FRAME = 320                                    # 20ms/帧，和 web 前端一致（核心自己攒块）

# ── ① 记忆核心：FunASR 流式 ASR + silero VAD + 0–500ms 投机预取，全在 vm.stream() 里 ──
vm = VoiceMem.from_config({
    "mode":      "text_mode",                  # 只存文本；stream 的 ASR 照常加载
    "embedding": {"provider": "local"},        # 记忆向量走本地 E5 —— 投机预算内不能走网络
    "slots":     {"provider": "local"},        # slot 分类走本地 E5（0 LLM）
})
stream = vm.stream(on_partial=lambda t: print(f"\r🎙️  {t}", end="", flush=True),
                   src_rate=SR)

# ── ② 回复模型：Voicemem QLoRA adapter ─────────────────────────────────────────
# 35B bf16 ≈ 70GB 显存；不够就用 BitsAndBytesConfig(load_in_4bit=True,
# bnb_4bit_quant_type="nf4") 加载 base（也和 adapter 的训练配置对齐）。
BASE    = "Qwen/Qwen3.6-35B-A3B"
ADAPTER = "LangJiaqi77/Voicemem-Qwen3_6-35B-A3B-QLoRA-v2"
tok   = AutoTokenizer.from_pretrained(BASE)
base  = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(base, ADAPTER).eval()


def our_model_reply(ctx: str, user_text: str) -> str:
    """ctx = turn.memory_context，说话期间就检索好了，这里直接当 system prompt 用。"""
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": ctx}, {"role": "user", "content": user_text}],
        tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**ids, max_new_tokens=200, do_sample=True, temperature=0.7)
    return tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True).strip()


# ── ③ 麦克风：callback 只往队列丢，永远不被生成/写记忆阻塞 ─────────────────────
mic_q: queue.Queue = queue.Queue()


def _mic_cb(indata, frames, time_info, status):
    if status:                                 # 溢出/欠载在这里报出来，别吞掉
        print(f"\n[mic] {status}", flush=True)
    pcm16 = (np.clip(indata[:, 0], -1, 1) * 32767).astype(np.int16)
    mic_q.put(pcm16.tobytes())


async def on_turn(turn):
    """一轮说完：记忆早在说话期间预取好了（turn.memory_context），直接拿来生成。"""
    answer = await asyncio.to_thread(our_model_reply, turn.memory_context, turn.text)
    print(f"\n🧑 {turn.text}\n🤖 {answer}\n", flush=True)
    threading.Thread(                          # 存这轮：事实抽取扔后台，不挡下一轮
        target=lambda: vm.ingest(turn.text, async_facts=True), daemon=True).start()
    # 要语音回复：把 answer 送 TTS（见 web/utils.py 的 tts_stream，可离线 piper / OpenAI）


async def main():
    loop = asyncio.get_running_loop()
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                        blocksize=FRAME, callback=_mic_cb):
        print("🎙️  开始说话…（Ctrl-C 退出）", flush=True)
        while True:
            pcm = await loop.run_in_executor(None, mic_q.get)
            st = await stream.feed(pcm)        # 核心：ASR + VAD + 投机预取
            if st.turn:                        # VAD 确认说完 → 记忆已备好
                await on_turn(st.turn)


if __name__ == "__main__":
    vm.classify("你好")                        # 预热本地 E5，第一轮就快
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 bye")
