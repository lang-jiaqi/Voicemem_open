"""voicemem web demo —— 对话核心（EOU 0–500ms 投机预取）。管道在 utils.py，渲染在 index.html。

看点：本地 ASR+VAD 边听边算，partial 一到就后台起投机 Search（本地 E5 向量 + 本地 slot
分类，注入的 LocalQueryClassifier，0 LLM 0 网络），VAD 在 500ms 确认说完时记忆早已算好，
交给两条 ~10 行控制流的只剩「发 LLM / 发 Realtime」。说到一半停顿又续上（barge-in）→ 取消投机。

跑：cd web && DEMO_MODE=llm_tts OPENAI_API_KEY=sk-... python run.py   # http://localhost:8787
    （DEMO_MODE=realtime 切 realtime；需 Realtime API 权限）

注意：记忆向量用本地 384 维 E5（投机预算内不能走网络）。换过旧 demo（OpenAI 1536 维）留了
记忆库的，维度不兼容——先清掉记忆目录再跑。
"""
import asyncio
import base64
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import uvicorn

HERE = Path(__file__).resolve().parent
_ROOT = HERE.parent
sys.path.insert(0, str(HERE))                       # 让 `import utils` 找到同目录管道层
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("VOICEMEM_MODELS_DIR", str(_ROOT / "models"))

import utils                                         # noqa: E402  同目录管道层
from voicemem import VoiceMem                        # noqa: E402
from voicemem.memory_api import build_memory_context  # noqa: E402
from voicemem.leftbrain.cognitive_graph.local_query_classifier import LocalQueryClassifier  # noqa: E402

MODE = os.environ.get("DEMO_MODE", "llm_tts")        # llm_tts | realtime
SPEC_MIN_CHARS, GAMBLE_S, CONFIRM_S = 6, 0.20, 0.50  # partial 起投机 / 赌说完 / VAD 确认结束

# embedding + 分类器都注入本地、共享一份 E5 → 整条 search 0 LLM、0 网络（实测 Search 本体 ~10ms）
vm = VoiceMem(mode="multi_modal",
              memory_root=os.environ.get("VOICEMEM_MEMORY_ROOT"),
              embedding=lambda: utils.LocalE5Embedder(),
              schema=lambda: LocalQueryClassifier(model=utils.shared_e5()))


@dataclass
class Pending:
    """一轮说完时、投机预取早已算好的「预算记忆」——控制流拿来直接回复，不再搜。"""
    text: str
    memory_context: str
    result: object
    spoken: bool = True          # True=语音轮（音频已进 realtime 缓冲），False=打字轮


# ══════════════════ 两条控制流（各 ~10 行，只消费预取好的 Pending）══════════════════

async def voicemem_llm_tts(pending, send, send_audio):
    """记忆已在关键路径外预取好：LLM 流式回复 → TTS 流式语音。"""
    await send({"type": "user_transcript", "text": pending.text})
    await send({"type": "memory_hits", **utils.hits_payload(pending.result)})
    await send({"type": "answer_start"})
    reply = ""
    async for d in utils.llm_stream(pending.text, pending.memory_context):
        reply += d
        await send({"type": "answer_delta", "text": d})
    await send({"type": "answer_done"})
    async for pcm in utils.tts_stream(reply):
        await send_audio(pcm)
    vm.ingest(pending.text)


async def voicemem_realtime(pending, conn, send, send_audio):
    """记忆已预取好：注入为 Realtime 指令 → 触发原生语音（音频早已平行进缓冲）。"""
    await send({"type": "user_transcript", "text": pending.text})
    await send({"type": "memory_hits", **utils.hits_payload(pending.result)})
    await conn.session.update(session={"type": "realtime", "turn_detection": None,
                                       "instructions": pending.memory_context or ""})
    if pending.spoken:
        await conn.input_audio_buffer.commit()
    else:
        await conn.conversation.item.create(item={"type": "message", "role": "user",
                                                  "content": [{"type": "input_text", "text": pending.text}]})
    await conn.response.create()
    await send({"type": "answer_start"})
    async for ev in conn:
        t = getattr(ev, "type", "")
        if t.endswith("output_audio.delta"):               await send_audio(base64.b64decode(ev.delta))
        elif t.endswith("output_audio_transcript.delta"):  await send({"type": "answer_delta", "text": ev.delta})
        elif t.endswith("response.done"):                  break
    await send({"type": "answer_done"})
    vm.ingest(pending.text)


# ══════════════════ 投机预取 + anticipatory 状态机（VAD 版 voicemem 逻辑）══════════════

async def speculate(text, spoken=True):
    """本地投机检索：注入的本地分类器出 slots(+entities) + 本地 E5 向量 Search（0 LLM/网络）。
    放线程里跑，好和继续读麦克风真正并发（这才叫"边听边预取"）。"""
    t0 = time.time()

    def work():
        c = vm.classify(text)                                 # 本地 LocalQueryClassifier
        r = vm.search(text, slots=c.slots, entities=c.entities)
        return r, build_memory_context(r)

    result, ctx = await asyncio.to_thread(work)
    print(f"[speculate] {text[:24]!r} -> {len(result.hits)} hits  {(time.time()-t0)*1000:.0f}ms", flush=True)
    return Pending(text, ctx, result, spoken)


async def anticipate(sock, on_frame=None):
    """本地 ASR+VAD 驱动的回合状态机，逐个 yield 确认回合的 Pending。
    on_frame(raw24k)：realtime 用它把原始音频平行喂给 OpenAI（方案 A）。"""
    asr = vm.utils.get("asr"); vad = utils.make_vad(); asr.reset()
    text, silence, spoke = "", 0.0, False
    spec, spec_text = None, ""

    async def confirm(spoken):
        try:
            return await (spec or speculate(text, spoken))
        except asyncio.CancelledError:
            return await speculate(text, spoken)

    while True:
        msg = await sock.receive()
        if msg.get("text"):                                   # 打字轮
            data = json.loads(msg["text"])
            if data.get("type") == "user_text" and data.get("text", "").strip():
                yield await speculate(data["text"], spoken=False)
            continue
        if msg.get("bytes") is None:
            continue
        raw = msg["bytes"]
        if on_frame:
            await on_frame(raw)                               # 方案 A：音频也进 OpenAI 缓冲
        frame = utils.resample(np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0)
        text = asr.feed(frame)
        if vad.is_speech(frame):
            if silence > 0 and spec:                          # barge-in：又开口了 → 丢弃这次投机
                spec.cancel(); spec, spec_text = None, ""
            spoke, silence = True, 0.0
        else:
            silence += len(frame) / 16000.0
        if text.strip():
            await sock.send_json({"type": "partial_transcript", "text": text, "replace": True})
        # 边说边预取 / 200ms 赌说完补发
        if spoke and text.strip() and text != spec_text and \
                (silence == 0.0 and len(text) >= SPEC_MIN_CHARS or silence >= GAMBLE_S):
            if spec:
                spec.cancel()
            spec_text = text
            spec = asyncio.create_task(speculate(text))
        if spoke and silence >= CONFIRM_S and text.strip():   # VAD 确认说完 → 交出预算记忆
            yield await confirm(spoken=True)
            asr.reset(); text, silence, spoke, spec, spec_text = "", 0.0, False, None, ""


# ══════════════════ 每种 mode 的会话循环 ══════════════════

async def llm_tts_session(sock):
    async for pending in anticipate(sock):
        await voicemem_llm_tts(pending, sock.send_json, sock.send_bytes)


async def realtime_session(sock):
    """方案 A：整段麦克风音频平行喂给 OpenAI Realtime；本地 ASR+VAD 只负责投机记忆 +
    用 500ms 判回合（关掉 OpenAI 自带 server_vad）。"""
    async with utils.realtime_connect() as conn:
        await conn.session.update(session={"type": "realtime", "turn_detection": None})

        async def on_frame(raw):
            await conn.input_audio_buffer.append(audio=base64.b64encode(raw).decode())

        async for pending in anticipate(sock, on_frame=on_frame):
            await voicemem_realtime(pending, conn, sock.send_json, sock.send_bytes)


app = utils.build_app(MODE, realtime_session if MODE == "realtime" else llm_tts_session, vm.classify)


if __name__ == "__main__":
    print(f"[web] mode={MODE} -> http://localhost:8787/")
    vm.classify("你好")                                       # 预热本地 E5，第一轮就快
    uvicorn.run(app, host="0.0.0.0", port=8787)
