"""web demo 的管道层（非主流程）——核心对话逻辑在 run.py，页面渲染在 index.html。

这里放：本地 E5（memory embedding + slot 分类共享一份模型）、音频重采样/VAD、
LLM/TTS/Realtime 流、以及 FastAPI + WebSocket 接线。run.py 只管把这些拼成 0–500ms
投机预取的对话流程。
"""
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI
from pydantic import BaseModel

# 本地 E5 embedder（memory embedding + slot 分类共享一份模型）已提进核心，见
# voicemem/leftbrain/local_e5_embedder.py；这里 re-export 保持 `utils.LocalE5Embedder`
# / `utils.shared_e5()` 的既有调用点不变（run.py 用它注入 VoiceMem(embedding=...)）。
from voicemem.leftbrain.local_e5_embedder import LocalE5Embedder, shared_e5  # noqa: F401

HERE = Path(__file__).resolve().parent
CHAT_MODEL = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o")
TTS_MODEL = os.environ.get("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
TTS_BACKEND = os.environ.get("TTS_BACKEND", "openai")   # openai(api) | local(离线小模型)
RT_MODEL = os.environ.get("OPENAI_REALTIME_MODEL", "gpt-realtime")
client = AsyncOpenAI()


# ── 音频小工具 ────────────────────────────────────────────────────────────────
# resample 用核心那份（voicemem/utils/audio/stream_io.py），别再抄一遍。
# 原来这里还有一份 make_vad()——自从 demo 改成复用 vm.stream() 之后就没人调了，
# VAD 现在是核心的可注入能力（VoiceMem(vad=...) / config 的 vad 段），已删。
from voicemem.utils.audio.stream_io import resample  # noqa: E402,F401


# ── 回复模型也在一处配：统一 config 的 reply 段（run.py 从 CONFIG["reply"] 传入）──
# 不传就回落到模块级 env 默认（CHAT_MODEL / TTS_MODEL / TTS_BACKEND / RT_MODEL），
# 现有行为完全不变。reply 结构：{"llm": {"config": {"model": ...}},
# "tts": {"provider": "openai|local", "config": {"model": ...}},
# "realtime": {"config": {"model": ...}}}，每段可省。
def _reply_seg(reply, name):
    seg = (reply or {}).get(name) or {}
    return seg.get("provider"), (seg.get("config") or {})


# ── LLM / TTS / Realtime 流 ───────────────────────────────────────────────────
async def llm_stream(text, ctx, reply=None):
    _, cfg = _reply_seg(reply, "llm")
    stream = await client.chat.completions.create(
        model=cfg.get("model") or CHAT_MODEL, stream=True,
        messages=[{"role": "system", "content": ctx or "你是语音助手，简短自然地回答。"},
                  {"role": "user", "content": text}])
    async for chunk in stream:
        d = chunk.choices[0].delta.content
        if d:
            yield d


async def tts_stream(text, reply=None):
    """可切换 TTS：默认走 OpenAI api；reply.tts.provider==local（或 TTS_BACKEND=local）
    走离线本地小模型。两条都吐 24kHz PCM16 流，前端一视同仁播放。"""
    provider, cfg = _reply_seg(reply, "tts")
    backend_name = provider or TTS_BACKEND          # 对齐现有 TTS_BACKEND 语义
    backend = _local_tts_stream if backend_name == "local" else _openai_tts_stream
    async for chunk in backend(text, cfg.get("model")):
        yield chunk


async def _openai_tts_stream(text, model=None):
    """在线 api：OpenAI TTS（gpt-4o-mini-tts），response_format=pcm 就是 24k PCM16。"""
    async with client.audio.speech.with_streaming_response.create(
            model=model or TTS_MODEL, voice="alloy", input=text, response_format="pcm") as resp:
        async for chunk in resp.iter_bytes():
            yield chunk


@lru_cache(maxsize=1)
def _piper_voice():
    """离线小模型：默认 piper（纯离线 onnx，中英皆可）。装：pip install piper-tts；
    VOICEMEM_TTS_MODEL 指向 voice 的 .onnx。想换 kokoro / edge-tts 等，只改这个函数
    和下面 _local_tts_stream 的取样即可。piper api 随版本，对照其文档。"""
    from piper import PiperVoice
    return PiperVoice.load(os.environ["VOICEMEM_TTS_MODEL"])


async def _local_tts_stream(text, model=None):
    """离线本地 TTS：合成 → 重采样到 24k → 分块 yield，接口和在线版完全一致。
    （离线 voice 由 VOICEMEM_TTS_MODEL 指定；model 形参仅为和在线版对齐签名。）"""
    v = _piper_voice()
    sr = getattr(getattr(v, "config", None), "sample_rate", 22050)
    for raw in v.synthesize_stream_raw(text):          # 同步生成器，int16 bytes @ sr
        f = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
        out = resample(f, src=sr, dst=24000)           # 统一到前端/OpenAI 的 24k
        yield (np.clip(out, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


def realtime_connect(reply=None):
    """方案 A：整段麦克风音频平行喂给它出原生语音。事件名随 SDK 版本可能微调
    （对照 openai_voice_demo/backend/providers/realtime.py）。"""
    _, cfg = _reply_seg(reply, "realtime")
    return client.realtime.connect(model=cfg.get("model") or RT_MODEL)


# ── SearchResult → 脑图 html 认识的 memory_hits 负载 ──────────────────────────
def hits_payload(result):
    return {
        "left_brain": [{"text": h.text, "score": h.score, "attributed_to": h.attributed_to}
                       for h in result.hits],
        "right_brain_hits": [{"content": h.content, "source": h.source, "priority": h.priority}
                             for h in (getattr(result, "rb_hits", None) or [])],
        "current_scene": getattr(result, "current_scene", None) or None,
        "related_summaries": getattr(result, "related_summaries", None) or {},
    }


# ── FastAPI + WS 接线（仅接线，渲染都在 index.html）─────────────────────────────
def build_app(mode, session, classify):
    """session(sock)：run.py 传入的会话循环（llm_tts / realtime）。classify(query)：给脑图生长用。"""
    app = FastAPI()

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        await sock.send_json({"type": "session_ready", "mode": mode})
        await session(sock)

    class Q(BaseModel):
        query: str

    @app.post("/api/classify")                       # 脑图 html 用它把左脑按 slot 生长
    def api_classify(body: Q) -> dict:
        c = classify(body.query)
        return {"slots": list(c.slots), "entities": list(c.entities)}

    (HERE / "images").mkdir(exist_ok=True)
    app.mount("/images", StaticFiles(directory=HERE / "images"), name="images")

    @app.get("/")
    def index():
        return FileResponse(HERE / "index.html")

    return app
