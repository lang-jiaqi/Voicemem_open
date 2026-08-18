"""server.py — 极简 FastAPI 后端：把前台 `voicemem` 的函数暴露成 HTTP 接口。

本身没有业务逻辑，只是"JSON 请求 → 一次 voicemem.* 调用 → 原样回传"。所有能力都在
根目录的前台 `voicemem.py` 里（引擎是 `voicemem_core` 包）。前端 index.html 由你自己
改，POST 这几个接口即可。

运行::

    cd web && python server.py          # http://localhost:8000

需要 OPENAI_API_KEY。
"""

from __future__ import annotations

import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # 让 import voicemem 找到前台
import voicemem as api

HERE = Path(__file__).resolve().parent
app = FastAPI(title="voicemem web demo")

# 伺服 web/images/（前端 index.html 的脑图 ./images/background.png 从这里加载）
(HERE / "images").mkdir(exist_ok=True)
app.mount("/images", StaticFiles(directory=HERE / "images"), name="images")


# ── 请求体 ───────────────────────────────────────────────────────────────────
class IngestBody(BaseModel):
    text: str
    speaker: str = "Speaker 0"
    audio_path: str | None = None


class QueryBody(BaseModel):
    query: str
    slots: list[str] | None = None
    entities: list[str] | None = None


# ── 接口：每个只是转发到 voicemem.py 的一个能力 ────────────────────────────────
@app.post("/api/ingest")
def api_ingest(body: IngestBody) -> dict:
    return api.ingest(body.text, speaker=body.speaker, audio_path=body.audio_path)


@app.post("/api/preprocess")
def api_preprocess(body: IngestBody) -> dict:
    return api.preprocess(body.text, audio_path=body.audio_path)


@app.post("/api/search")
def api_search(body: QueryBody) -> dict:
    return {"hits": api.search(body.query, slots=body.slots, entities=body.entities)}


@app.post("/api/context")
def api_context(body: QueryBody) -> dict:
    return {"context": api.build_context(body.query)}


@app.post("/api/classify")
def api_classify(body: QueryBody) -> dict:
    return api.classify(body.query)          # → {slots, entities}


@app.get("/api/self-check")
def api_self_check() -> dict:
    return {"report": api.self_check()}


# ── 前端（index.html 由你自己改）──────────────────────────────────────────────
@app.get("/")
def index() -> FileResponse:
    return FileResponse(HERE / "index.html")


if __name__ == "__main__":
    print("[web] voicemem demo -> http://localhost:8000/")
    uvicorn.run(app, host="0.0.0.0", port=8000)
