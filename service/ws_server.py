"""主入口。WebSocket 服务:接前端音频 -> VoiceAssistant -> 推前端。

跟原来 main.py 的差异：不再单独起一个 cognitive_server.py（Flask）进程，
CognitiveService 直接在这个进程里构造好，传给每个 VoiceAssistant 会话共用。

启动: python ws_server.py
前端: index.html 连 ws://<host>:8771
"""
from __future__ import annotations

import json
import os
import sys
import asyncio
import traceback
from pathlib import Path

# backend/ 加进 sys.path，这样 cognitive.py 里 `from voicemem import VoiceMem` 才找得到包
# （service/ 和 voicemem/ 是 backend/ 下的两个平级目录，不是 pip 安装的）。
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import websockets

from session import SharedModels, VoiceAssistant
from cognitive import CognitiveService

_HERE = Path(__file__).resolve().parent
_BACKEND = _HERE.parent

MODELS_DIR = os.environ.get("VOICEMEM_MODELS_DIR", str(_HERE / "models"))
MEMORY_ROOT = Path(os.environ.get("VOICEMEM_MEMORY_ROOT", str(_BACKEND / "memory")))
HISTORY_ROOT = _HERE / "history"
HOST, PORT = "0.0.0.0", int(os.environ.get("VOICEMEM_WS_PORT", "8771"))

_cognitive = CognitiveService(
    memory_root=MEMORY_ROOT,
    history_root=HISTORY_ROOT,
    base_url=os.environ.get("OPENAI_BASE_URL") or None,
)
_shared_models: SharedModels | None = None


class ChatHistory:
    """供 GUI 恢复的聊天记录；与认知管线的逐轮诊断 history 分开保存。"""

    _PERSIST_TYPES = {"result", "append", "answer_start", "answer", "answer_done"}
    _MAX_EVENTS = 1_000

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.events = self._load()

    def _load(self) -> list[dict]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def append(self, message: dict) -> None:
        if message.get("type") not in self._PERSIST_TYPES:
            return
        # partial 是短暂的，result 未锁定时也只是中间态。
        if message.get("type") == "result" and not message.get("locked"):
            return
        # 聊天区不使用记忆检索详情，避免将其重复写进展示历史。
        event = {key: message[key] for key in (
            "type", "ts", "global_id", "text", "speaker", "emotion", "locked"
        ) if key in message}
        self.events.append(event)
        self.events = self.events[-self._MAX_EVENTS:]
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.events, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)


async def handle_client(ws):
    """一个连接 = 一个语音助手会话。"""
    chat_history = ChatHistory(HISTORY_ROOT / "chat.json")

    async def send(msg):
        # bytes -> 二进制帧(回答音频 PCM);dict -> JSON 文本帧
        if isinstance(msg, (bytes, bytearray)):
            await ws.send(msg)
        else:
            chat_history.append(msg)
            await ws.send(json.dumps(msg, ensure_ascii=False))

    va = None
    try:
        await ws.send(json.dumps({"type": "history", "events": chat_history.events}, ensure_ascii=False))
        va = VoiceAssistant(MODELS_DIR, _cognitive, send_to_ui=send,
                            shared_models=_shared_models)
        await va.start()                 # 建立常驻 Live 连接
        async for message in ws:
            await va.on_audio(message)
    except Exception:
        print("[WebSocket] session error:", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise
    finally:
        if va is not None:
            await va.close()


async def http_ok(connection, request):
    if request.headers.get("Upgrade", "").lower() != "websocket":
        return connection.respond(200, "VoiceMem WebSocket endpoint.\n")


async def main():
    global _shared_models
    _shared_models = SharedModels()
    if os.environ.get("VOICEMEM_WARMUP", "1") != "0":
        # 模型首轮推理会阻塞；在监听端口前完成，确保用户首句不承担冷启动。
        await asyncio.to_thread(_shared_models.warmup)
    async with websockets.serve(handle_client, HOST, PORT,
                                max_size=None, process_request=http_ok):
        print(f"语音助手已启动: ws://{HOST}:{PORT}")
        print(f"models dir: {MODELS_DIR}")
        print(f"memory root: {MEMORY_ROOT}")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
