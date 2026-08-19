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

HERE = Path(__file__).resolve().parent
CHAT_MODEL = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o")
TTS_MODEL = os.environ.get("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
RT_MODEL = os.environ.get("OPENAI_REALTIME_MODEL", "gpt-realtime")
client = AsyncOpenAI()


# ── 本地 E5：memory embedding + slot 分类共享同一个模型实例（省一份内存）──────────
_E5_NAME = "intfloat/multilingual-e5-small"


@lru_cache(maxsize=1)
def shared_e5():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(_E5_NAME)


class LocalE5Embedder:
    """注入 VoiceMem(embedding=...)：Rank/存取的向量走本地 E5，0-500ms 预算内不碰网络。"""
    @property
    def model_name(self): return f"{_E5_NAME} (local)"
    @property
    def dimensions(self): return shared_e5().get_sentence_embedding_dimension()
    def embed_texts(self, texts):
        if not texts: return []
        return np.asarray(shared_e5().encode([f"passage: {t}" for t in texts],
                                             normalize_embeddings=True)).tolist()
    def embed_query_text(self, text):
        return np.asarray(shared_e5().encode([f"query: {text}"], normalize_embeddings=True)[0]).tolist()


# ── 音频小工具 ────────────────────────────────────────────────────────────────
def resample(f32, src=24000, dst=16000):        # 脑图 html 发 24k，sherpa ASR 要 16k
    n = int(len(f32) * dst / src)
    return np.interp(np.arange(n) * src / dst, np.arange(len(f32)), f32).astype(np.float32)


def make_vad():
    import sherpa_onnx
    d = os.environ.get("VOICEMEM_MODELS_DIR", "../models")
    v = sherpa_onnx.VoiceActivityDetector(sherpa_onnx.VadModelConfig(
        silero_vad=sherpa_onnx.SileroVadModelConfig(model=f"{d}/silero_vad.onnx", threshold=0.5),
        sample_rate=16000), buffer_size_in_seconds=30)
    class _V:
        def is_speech(self, frame): v.accept_waveform(frame); return v.is_speech_detected()
    return _V()


# ── LLM / TTS / Realtime 流 ───────────────────────────────────────────────────
async def llm_stream(text, ctx):
    stream = await client.chat.completions.create(
        model=CHAT_MODEL, stream=True,
        messages=[{"role": "system", "content": ctx or "你是语音助手，简短自然地回答。"},
                  {"role": "user", "content": text}])
    async for chunk in stream:
        d = chunk.choices[0].delta.content
        if d:
            yield d


async def tts_stream(text):
    async with client.audio.speech.with_streaming_response.create(
            model=TTS_MODEL, voice="alloy", input=text, response_format="pcm") as resp:
        async for chunk in resp.iter_bytes():
            yield chunk


def realtime_connect():
    """方案 A：整段麦克风音频平行喂给它出原生语音。事件名随 SDK 版本可能微调
    （对照 openai_voice_demo/backend/providers/realtime.py）。"""
    return client.realtime.connect(model=RT_MODEL)


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
