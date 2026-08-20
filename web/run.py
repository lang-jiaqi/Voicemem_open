"""voicemem web demo —— 对话核心（EOU 0–500ms 投机预取）。管道在 utils.py，渲染在 index.html。

看点：本地 ASR+VAD 边听边算，partial 一到就后台起投机 Search（本地 E5 向量 + 本地 slot
分类，注入的 LocalQueryClassifier，0 LLM 0 网络），VAD 在 500ms 确认说完时记忆早已算好，
交给两条 ~10 行控制流的只剩「发 LLM / 发 Realtime」。说到一半停顿又续上（barge-in）→ 取消投机。

跑（参数见 ``--help``；每个都能用同名环境变量给默认值）::

    export OPENAI_API_KEY=sk-...
    python web/run.py                     # 默认 realtime
    python web/run.py \\
      --mode llm_tts \\                    # 没有 Realtime 权限时走这条
      --port 8787 \\
      --spec_min_chars 6 \\
      --gamble_ms 200 \\
      --confirm_ms 500

默认走 ``realtime``（OpenAI 原生语音）：一次往返直接出声，不像 llm_tts 那样要
"LLM 出文本(~1.0s) → 攒够一句 → TTS 合成(~1.2s)" 两段串行，体验差一截。
key 没有 Realtime 权限就用 ``--mode llm_tts``，那条路只要普通 chat + TTS，
TTS 还能换成本地离线模型（``TTS_BACKEND=local``）。

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
                   default=os.environ.get("DEMO_MODE", "realtime"),
                   help="回复控制流：realtime=OpenAI 原生语音（默认，体验最好）；"
                        "llm_tts=LLM 流→TTS 流（不需要 Realtime 权限，可换本地 TTS）")
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

# 怎么切给 TTS 的段，决定了多久能出第一声。实测 gpt-4o-mini-tts 的首帧延迟随文本
# 长度涨：8字 615ms / 25字 902ms / 100字 1318ms——所以**第一段要尽量短**（早出声），
# 后面的段可以长（少调几次、语气连贯）。
_SENT_END  = "。！？!?…\n"          # 句末：正常的切段点
_SOFT_END  = "，,、；;：: "          # 句中停顿：只有第一段用，为了抢出第一声
_FIRST_MIN = 6                      # 第一段攒够这么多字，遇到任何停顿就发
_FIRST_MAX = 20                     # 一个停顿都没有时，第一段也不能再等了
_SENT_MIN  = 12                     # 后续段的最短长度
_SENT_MAX  = 60                     # 后续段的兜底：LLM 一口气不换气也得切


def _cut_point(buf: str, first: bool) -> bool:
    """这段够不够发去合成了。"""
    s = buf.strip()
    if not s:
        return False
    if first:                                  # 抢第一声：逗号也算，实在没有就按长度切
        return (len(s) >= _FIRST_MIN and s[-1] in _SENT_END + _SOFT_END) or len(s) >= _FIRST_MAX
    return (len(s) >= _SENT_MIN and s[-1] in _SENT_END) or len(s) >= _SENT_MAX


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
            try:
                async for pcm in utils.tts_stream(seg, REPLY):
                    await send_audio(pcm)
            except Exception as e:                # 多半是听到一半关了页面，不是错误
                print(f"[web] 语音发送中断：{type(e).__name__}", flush=True)
                break

    speaker = asyncio.create_task(speak())
    reply, buf, sent = "", "", 0
    try:
        async for d in utils.llm_stream(pending.text, pending.memory_context, REPLY):
            reply += d
            buf += d
            await send({"type": "answer_delta", "text": d})
            if _cut_point(buf, first=sent == 0):
                await queue.put(buf.strip())
                buf, sent = "", sent + 1
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


_RT_PERSONA = "你是用户的语音助手，简短自然地回答。"


def _realtime_instructions(memory_context: str) -> str:
    """人设 + 这一轮检索到的记忆。要说清楚这是「你记得的事」，否则模型会把它当成
    背景资料念出来，而不是当成自己对这个用户的记忆自然地用。"""
    if not memory_context:
        return _RT_PERSONA
    return (f"{_RT_PERSONA}\n\n以下是你记得的关于这个用户的事，回答时自然地用上，"
            f"别说「根据记录」这类话：\n{memory_context}")


async def start_realtime_turn(pending, conn, send):
    """把预取好的记忆注入 Realtime session，触发这一轮的原生语音。

    只负责"发起"；收音频/文本和收尾都在常驻的事件泵里（见 realtime_session）——
    OpenAI 的事件流只能有一个消费者，每轮各读各的会串台：上一轮被打断后残留的
    response.done 会被下一轮读到，当成自己说完了。
    """
    await send({"type": "user_transcript", "text": pending.text})
    await send({"type": "memory_hits", **utils.hits_payload(pending.result)})
    if pending.spoken:
        await conn.input_audio_buffer.commit()
    else:
        await conn.conversation.item.create(item={"type": "message", "role": "user",
                                                  "content": [{"type": "input_text", "text": pending.text}]})
    # 记忆走 response.create 的 per-response instructions，不是 session.update。
    # 后者是会话级设置，实测更新完模型这一轮根本读不到（问"我的猫叫什么"，库里
    # 明明检索到了"叫墨墨"，模型还答"你刚提过但我没听清"）。
    await conn.response.create(response={
        "instructions": _realtime_instructions(pending.memory_context),
    })
    await send({"type": "answer_start"})


async def _no_realtime(sock, err):
    """没有 Realtime 权限时别让人对着 traceback 猜——直接说清楚换哪条路。"""
    msg = (f"连不上 OpenAI Realtime（{type(err).__name__}: {err}）。"
           "这个 key 可能没有 Realtime 权限——改用 `python web/run.py --mode llm_tts`，"
           "那条路只要普通 chat + TTS。")
    print(f"[web] {msg}", flush=True)
    try:
        await sock.send_json({"type": "error", "message": msg})
    except Exception:
        pass


# ══════════════════ 驱动 voicemem 核心流式会话（vm.stream()）══════════════════
# ASR + VAD + 0–500ms 投机预取（边说边预取 / 200ms 赌说完 / barge-in / 500ms 确认）
# 全在核心 VoiceStream 里。这里只做 demo 该做的：搬 socket 帧、发 partial、把说完
# 的一轮包成 Pending 交给控制流——demo 就是核心的使用示例，不再平行重写一套。

async def anticipate(sock, on_frame=None, on_speech=None):
    """驱动核心流式会话，逐个 yield 确认回合的 Pending。
    on_frame(raw24k)：realtime 用它把原始音频平行喂给 OpenAI（方案 A）。
    on_speech()：本地 VAD 一听到人声就叫一次——realtime 拿它做打断（barge-in）。"""
    stream = vm.stream(spec_min_chars=SPEC_MIN_CHARS, gamble_s=GAMBLE_S, confirm_s=CONFIRM_S)
    last_partial = ""
    while True:
        msg = await sock.receive()
        if msg.get("type") == "websocket.disconnect":         # 关页面/刷新：收工
            return
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
        if st.state == "<speak>" and on_speech:
            await on_speech()                                 # 助手还在说 → 打断它
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
    connected = False
    try:
        async with utils.realtime_connect(REPLY) as conn:
            # 握手成功不代表能用：没权限/模型名不对时，OpenAI 是**连上之后**再发
            # close 4000（invalid_model / 权限错误）。所以要等第一次交互成功才算数。
            await conn.session.update(session={"type": "realtime", "turn_detection": None})
            connected = True

            # 这一轮的状态：谁在说、说了什么、这轮用户的输入是什么
            turn = {"live": False, "reply": "", "pending": None}

            def close_turn():
                """一轮结束（说完或被打断）：把两半对话一起存。被打断时存的是用户
                真正听到的那部分——跟 llm_tts 那边 capture() 的取舍一致。"""
                p, reply = turn["pending"], turn["reply"]
                turn.update(live=False, reply="", pending=None)
                if p is not None:
                    vm.ingest(p.text, agent_reply=reply, async_facts=True)

            async def pump():
                """常驻事件泵：OpenAI 的事件流只有这一个消费者。

                turn["live"] 为假时一律不往前端转发——被打断后 OpenAI 还会吐一会儿
                残余音频，转过去的话前端刚 stopPlayback 又排上新的，打断就不干净了。
                """
                async for ev in conn:
                    t = getattr(ev, "type", "")
                    if t.endswith("output_audio.delta"):
                        if turn["live"]:
                            await sock.send_bytes(base64.b64decode(ev.delta))
                    elif t.endswith("output_audio_transcript.delta"):
                        if turn["live"]:
                            turn["reply"] += ev.delta
                            await sock.send_json({"type": "answer_delta", "text": ev.delta})
                    elif t.endswith("response.done") or t.endswith("response.cancelled"):
                        if turn["live"]:
                            await sock.send_json({"type": "answer_done"})
                            close_turn()

            async def on_frame(raw):
                await conn.input_audio_buffer.append(audio=base64.b64encode(raw).decode())

            async def on_speech():
                """用户在助手说话时开口 → 打断。幂等：live 一置 False 就不再触发。"""
                if not turn["live"]:
                    return
                turn["live"] = False
                await conn.response.cancel()
                await sock.send_json({"type": "answer_interrupt"})   # 前端停播已排队的音频
                close_turn()

            pump_task = asyncio.create_task(pump())
            try:
                async for pending in anticipate(sock, on_frame=on_frame, on_speech=on_speech):
                    if turn["live"]:                     # 上一轮还没说完就被新的一轮顶掉
                        await on_speech()
                    turn.update(live=True, reply="", pending=pending)
                    await start_realtime_turn(pending, conn, sock.send_json)
            finally:
                pump_task.cancel()
    except Exception as e:
        if connected:
            raise
        await _no_realtime(sock, e)


app = utils.build_app(MODE, realtime_session if MODE == "realtime" else llm_tts_session, vm.classify)


if __name__ == "__main__":
    print(f"[web] mode={MODE} spec≥{SPEC_MIN_CHARS}字 gamble={ARGS.gamble_ms}ms "
          f"confirm={ARGS.confirm_ms}ms -> http://localhost:{ARGS.port}/", flush=True)
    # 全部预热在这儿做完，别让第一句话去等模型加载。ASR(FunASR paraformer)
    # 是懒加载的，等用户开口才拉起来要好几秒——那几秒的音频堆在 socket 缓冲里，
    # 追赶时逐帧喂 VAD，静音会瞬间累计过 confirm_ms，第一句直接被截断（听感就是
    # "第一句又慢又不准"）。
    vm.classify("你好")                                       # 本地 E5
    print("[web] 预热 ASR / VAD …", flush=True)
    import numpy as _np
    vm.utils.get("asr").feed(_np.zeros(9600, dtype=_np.float32))   # 拉起模型并跑一块
    vm.utils.get("asr").reset()
    vm.utils.get("vad").is_speech(_np.zeros(512, dtype=_np.float32))
    print("[web] 就绪", flush=True)
    uvicorn.run(app, host=ARGS.host, port=ARGS.port)
