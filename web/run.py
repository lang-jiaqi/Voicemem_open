"""voicemem web demo —— 对话核心（EOU 0–500ms 投机预取）。管道在 utils.py，渲染在 index.html。

看点：本地 ASR+VAD 边听边算，partial 一到就后台起投机 Search（本地 E5 向量 + 本地 slot
分类，注入的 LocalQueryClassifier，0 LLM 0 网络），VAD 在 500ms 确认说完时记忆早已算好，
交给两条 ~10 行控制流的只剩「发 LLM / 发 Realtime」。说到一半停顿又续上（barge-in）→ 取消投机。

跑（参数见 ``--help``；每个都能用同名环境变量给默认值）::

    export OPENAI_API_KEY=sk-...
    python web/run.py \\
      --mode llm_tts \\
      --port 8787 \\
      --spec_min_chars 6 \\
      --gamble_ms 200 \\
      --confirm_ms 500

``--mode realtime`` 切 OpenAI Realtime 原生语音（需 Realtime API 权限）。

注意：记忆向量用本地 384 维 E5（投机预算内不能走网络）。换过旧 demo（OpenAI 1536 维）留了
记忆库的，维度不兼容——先清掉记忆目录再跑。
"""
import argparse
import asyncio
import base64
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import uvicorn

HERE = Path(__file__).resolve().parent
_ROOT = HERE.parent
sys.path.insert(0, str(HERE))                       # 让 `import utils` 找到同目录管道层
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("VOICEMEM_MODELS_DIR", str(_ROOT / "models"))


# ══════════════════ 命令行参数（同名环境变量给默认值，两种都行）══════════════════
# 放在下面那两个重 import 之前：utils / voicemem 会拉起 torch + sentence-transformers，
# 排在它们后面的话 `--help` 得先等模型库加载完。被 import 时不吃 sys.argv（传 []）。

def _parse(argv):
    p = argparse.ArgumentParser(description="voicemem web demo（脑图 + 0–500ms 投机预取）")
    p.add_argument("--mode", choices=["llm_tts", "realtime"],
                   default=os.environ.get("DEMO_MODE", "llm_tts"),
                   help="回复控制流：llm_tts=LLM 流→TTS 流；realtime=OpenAI 原生语音")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=int(os.environ.get("VOICEMEM_PORT", 8787)))
    p.add_argument("--spec_min_chars", type=int, default=6,
                   help="partial 转写到几个字起投机预取")
    p.add_argument("--gamble_ms", type=int, default=200,
                   help="静音多久就赌你说完了，补投机一次")
    p.add_argument("--confirm_ms", type=int, default=500,
                   help="静音多久由 VAD 确认一轮结束，交出 Turn")
    p.add_argument("--config", default=os.environ.get("VOICEMEM_CONFIG"),
                   help="一个 .json，整体覆盖下面的 CONFIG")
    p.add_argument("--memory_root", default=os.environ.get("VOICEMEM_MEMORY_ROOT"),
                   help="记忆库目录")
    return p.parse_args(argv)


ARGS = _parse(None if __name__ == "__main__" else [])

import utils                                         # noqa: E402  同目录管道层
from voicemem import VoiceMem                        # noqa: E402

MODE = ARGS.mode                                     # llm_tts | realtime
SPEC_MIN_CHARS = ARGS.spec_min_chars                 # partial 起投机
GAMBLE_S  = ARGS.gamble_ms / 1000                    # 赌说完
CONFIRM_S = ARGS.confirm_ms / 1000                   # VAD 确认结束

# ══════════════════ 统一配置入口：一个 dict 配齐所有本地/api 模型 ══════════════════
# 打开这个 dict 就知道每个模型走本地还是 api。记忆侧（embedding/slots）走本地 E5
# → 整条 search 0 LLM、0 网络（实测 Search 本体 ~10ms）；reply 段（回复用的
# llm/tts/realtime）也在这一处配，省得分散在各处 env。缺省项走内置默认。
# 想外挂一份自定义 config：--config path.json（或 VOICEMEM_CONFIG）整体覆盖。
CONFIG = {
    "mode": "multi_modal",
    "memory_root": ARGS.memory_root,
    "embedding": {"provider": "local"},              # 记忆向量走本地 E5（0 网络）
    "slots":     {"provider": "local"},              # slot 分类走本地 E5（0 LLM）
    # reply：回复用模型（核心不管，web 读）。默认全走 OpenAI api。
    "reply": {
        "llm":      {"provider": "openai", "config": {"model": utils.CHAT_MODEL}},
        "tts":      {"provider": utils.TTS_BACKEND, "config": {"model": utils.TTS_MODEL}},
        "realtime": {"provider": "openai", "config": {"model": utils.RT_MODEL}},
    },
}

# --config / VOICEMEM_CONFIG 指向的 json 整体覆盖上面的 CONFIG（一个文件配齐）。
if ARGS.config:
    CONFIG = json.loads(Path(ARGS.config).read_text(encoding="utf-8"))

REPLY = CONFIG.get("reply")                           # 传给 utils 的回复函数

# 声明式构造：from_config 是现有注入机制之上的糖（VoiceMem(embedding=fn, schema=fn,…)）。
vm = VoiceMem.from_config(CONFIG)


@dataclass
class Pending:
    """一轮说完时、投机预取早已算好的「预算记忆」——控制流拿来直接回复，不再搜。"""
    text: str
    memory_context: str
    result: object
    spoken: bool = True          # True=语音轮（音频已进 realtime 缓冲），False=打字轮


# ══════════════════ 两条控制流（各 ~10 行，只消费预取好的 Pending）══════════════════

_SENT_END = "。！？!?…\n"          # 断句点：到这儿就够合成一段了
_SENT_MIN = 12                     # 太短的碎片不单独合成——每次 TTS 都要吃一次首帧延迟


async def voicemem_llm_tts(pending, send, send_audio):
    """记忆已在关键路径外预取好：LLM 流式回复 → TTS 流式语音。

    TTS 跟生成**并行**：LLM 吐满一句就丢进队列，另一条协程取出来合成、发音频。
    等全文生成完再开始合成的话，文本早打完了、音频还没起头（实测 TTS 首帧就要
    ~1.2s，加上生成那几秒，用户看着字干等）。
    """
    await send({"type": "user_transcript", "text": pending.text})
    await send({"type": "memory_hits", **utils.hits_payload(pending.result)})
    await send({"type": "answer_start"})

    queue: asyncio.Queue = asyncio.Queue()

    async def speak():
        while (seg := await queue.get()) is not None:
            async for pcm in utils.tts_stream(seg, REPLY):
                await send_audio(pcm)

    speaker = asyncio.create_task(speak())
    reply, buf = "", ""
    try:
        async for d in utils.llm_stream(pending.text, pending.memory_context, REPLY):
            reply += d
            buf += d
            await send({"type": "answer_delta", "text": d})
            if buf.rstrip().endswith(tuple(_SENT_END)) and len(buf.strip()) >= _SENT_MIN:
                await queue.put(buf.strip())
                buf = ""
        if buf.strip():
            await queue.put(buf.strip())
    finally:
        await queue.put(None)                   # 生成出错也要让 speak() 收工

    await send({"type": "answer_done"})
    # 先落记忆再等放完：ingest 排在音频后面的话，用户一听完就关页面（语音场景很
    # 常见），send_audio 抛 WebSocketDisconnect，这一轮就永远存不进去。
    # async_facts=True：抽事实走后台，不堵住读麦克风那条线。
    vm.ingest(pending.text, agent_reply=reply, async_facts=True)   # 两半一起存
    await speaker


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
    reply = ""                                            # 原生语音也有转写，攒起来一起存
    async for ev in conn:
        t = getattr(ev, "type", "")
        if t.endswith("output_audio.delta"):               await send_audio(base64.b64decode(ev.delta))
        elif t.endswith("output_audio_transcript.delta"):
            reply += ev.delta
            await send({"type": "answer_delta", "text": ev.delta})
        elif t.endswith("response.done"):                  break
    await send({"type": "answer_done"})
    vm.ingest(pending.text, agent_reply=reply, async_facts=True)   # 两半一起存，抽事实走后台


# ══════════════════ 驱动 voicemem 核心流式会话（vm.stream()）══════════════════
# ASR + VAD + 0–500ms 投机预取（边说边预取 / 200ms 赌说完 / barge-in / 500ms 确认）
# 全在核心 VoiceStream 里。这里只做 demo 该做的：搬 socket 帧、发 partial、把说完
# 的一轮包成 Pending 交给控制流——demo 就是核心的使用示例，不再平行重写一套。

async def anticipate(sock, on_frame=None):
    """驱动核心流式会话，逐个 yield 确认回合的 Pending。
    on_frame(raw24k)：realtime 用它把原始音频平行喂给 OpenAI（方案 A）。"""
    stream = vm.stream(spec_min_chars=SPEC_MIN_CHARS, gamble_s=GAMBLE_S, confirm_s=CONFIRM_S)
    last_partial = ""
    while True:
        msg = await sock.receive()
        if msg.get("text"):                                   # 打字轮
            data = json.loads(msg["text"])
            if data.get("type") == "user_text" and data.get("text", "").strip():
                turn = await stream.feed_text(data["text"])
                yield Pending(turn.text, turn.memory_context, turn.result, spoken=False)
            continue
        if msg.get("bytes") is None:
            continue
        raw = msg["bytes"]
        if on_frame:
            await on_frame(raw)                               # 方案 A：音频也进 OpenAI 缓冲
        st = await stream.feed(raw)                           # 核心：ASR + VAD + 投机预取
        if st.text.strip() and st.text != last_partial:
            last_partial = st.text
            await sock.send_json({"type": "partial_transcript", "text": st.text, "replace": True})
        if st.turn:                                           # VAD 确认说完 → 记忆早已预取好
            last_partial = ""
            yield Pending(st.turn.text, st.turn.memory_context, st.turn.result, spoken=True)


# ══════════════════ 每种 mode 的会话循环 ══════════════════

async def llm_tts_session(sock):
    async for pending in anticipate(sock):
        await voicemem_llm_tts(pending, sock.send_json, sock.send_bytes)


async def realtime_session(sock):
    """方案 A：整段麦克风音频平行喂给 OpenAI Realtime；本地 ASR+VAD 只负责投机记忆 +
    用 500ms 判回合（关掉 OpenAI 自带 server_vad）。"""
    async with utils.realtime_connect(REPLY) as conn:
        await conn.session.update(session={"type": "realtime", "turn_detection": None})

        async def on_frame(raw):
            await conn.input_audio_buffer.append(audio=base64.b64encode(raw).decode())

        async for pending in anticipate(sock, on_frame=on_frame):
            await voicemem_realtime(pending, conn, sock.send_json, sock.send_bytes)


app = utils.build_app(MODE, realtime_session if MODE == "realtime" else llm_tts_session, vm.classify)


if __name__ == "__main__":
    print(f"[web] mode={MODE} spec≥{SPEC_MIN_CHARS}字 gamble={ARGS.gamble_ms}ms "
          f"confirm={ARGS.confirm_ms}ms -> http://localhost:{ARGS.port}/", flush=True)
    vm.classify("你好")                                       # 预热本地 E5，第一轮就快
    uvicorn.run(app, host=ARGS.host, port=ARGS.port)
