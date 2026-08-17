"""FastAPI entry point. Run from inside this directory:

    cd openai_voice_demo/backend && python main.py

so `providers/` resolves as a normal sibling subpackage. Config/MemoryBridge
are process-wide singletons, constructed once at import time; the network
clients inside each Provider are constructed per-connection instead, since
building an OpenAI client is cheap and this keeps connection state isolated.
"""
from __future__ import annotations

from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from config import load_config
from memory_bridge import MemoryBridge
from providers.doubao_realtime import DoubaoRealtimeProvider
from providers.pipeline import PipelineProvider
from providers.qwen_realtime import QwenRealtimeProvider
from providers.realtime import RealtimeProvider

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

config = load_config()
memory_bridge = MemoryBridge(config)
app = FastAPI(title="openai_voice_demo")


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    provider = {"pipeline": PipelineProvider,
                "qwen": QwenRealtimeProvider,
                "doubao": DoubaoRealtimeProvider,
                "realtime": RealtimeProvider}[config.voice_mode](config)
    try:
        await provider.run_session(websocket, memory_bridge)
    except WebSocketDisconnect:
        pass
    finally:
        # Each WebSocket connection is one conversation session for this demo
        # (single-user, one browser tab = one session) -- Flush() here is what
        # makes voicemem's session-boundary batch (Algorithm 1 checkpoint +
        # right-brain attribution) actually run for the session that just
        # ended, instead of silently waiting for a next session that may never
        # come. finally (not just the WebSocketDisconnect branch) so it still
        # runs if the connection drops via some other exception.
        await memory_bridge.flush()


# Frontend files change often during development and browsers kept serving
# stale cached copies through normal reloads (repeated real support pain:
# "the fix doesn't work" reports that were actually a cached index.html).
# For a dev demo, correctness beats cache efficiency: tell the browser to
# never cache, so a plain reload always gets the current page.
@app.middleware("http")
async def no_cache(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    return response


# Mounted after /ws so the WebSocket route isn't shadowed by the static catch-all.
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    print(f"[openai_voice_demo] mode={config.voice_mode} audio_native={config.audio_native} "
          f"user_id={config.user_id} -> http://{config.host}:{config.port}/")
    uvicorn.run(app, host=config.host, port=config.port)
