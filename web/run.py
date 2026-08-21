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
import uuid
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

BARGE_DEBUG = os.environ.get("BARGE_DEBUG", "1") != "0"   # 打断为什么没触发：看这几行日志
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


#: 语音轮的音频落在这儿。归档表存的是路径，文件本身得真的在。
TURN_AUDIO_DIR = _ROOT / "results" / "turn_audio"


def save_turn_audio(pcm16k) -> str:
    """把这一轮的 PCM 存成 wav，返回路径；存不下就返回 ""（不影响这一轮对话）。

    没有这一步，AudioArchive 里一条记录都不会有——它只在 ingest 收到 audio_path
    时才写。之前 demo 全程走 WS 流、从不落盘，所以"把当时那段原声放回来"做不到。
    """
    if pcm16k is None or not len(pcm16k):
        return ""
    try:
        import numpy as np
        import soundfile as sf
        TURN_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        path = TURN_AUDIO_DIR / f"turn_{uuid.uuid4().hex[:12]}.wav"
        sf.write(path, np.asarray(pcm16k, dtype="float32"), 16000)
        return str(path)
    except Exception as e:
        print(f"[web] 存本轮音频失败（不影响对话）：{e}", flush=True)
        return ""


@dataclass
class Pending:
    """一轮说完时、投机预取早已算好的「预算记忆」——控制流拿来直接回复，不再搜。"""
    text: str
    memory_context: str
    result: object
    spoken: bool = True          # True=语音轮（音频已进 realtime 缓冲），False=打字轮
    audio_path: str = ""         # 这一轮落盘的 wav；ingest 拿它做场景/音乐/声纹感知，
                                 # 并在 audio_archive 里跟记忆绑定，之后能原样放回来
    stranger: bool = False       # 声纹认出说话的不是这个记忆库的主人


# ══════════════════ 两条控制流（各 ~10 行，只消费预取好的 Pending）══════════════════

# 分句规则（决定多久出第一声）在 voicemem/tts.py 的 cut_point。
_cut_point = utils.cut_point


async def voicemem_llm_tts(pending, send, send_audio):
    """记忆已在关键路径外预取好：LLM 流式回复 → TTS 流式语音。

    TTS 跟生成**并行**：LLM 吐满一句就丢进队列，另一条协程取出来合成、发音频。
    等全文生成完再开始合成的话，文本早打完了、音频还没起头（实测 TTS 首帧就要
    ~1.2s，加上生成那几秒，用户看着字干等）。
    """
    await send({"type": "user_transcript", "text": pending.text})
    await send({"type": "memory_hits", **utils.hits_payload(pending.result, has_audio=audio_of)})
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
        # 跟 realtime 用同一份指令：两条路必须表现一致，否则换个 --mode
        # 人设和「右脑不许念出来」的约束就悄悄没了。
        async for d in utils.llm_stream(pending.text,
                                        _realtime_instructions(pending.memory_context, pending.stranger),
                                        REPLY):
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
    vm.ingest(pending.text, agent_reply=reply, async_facts=True,
              audio=pending.audio_path or None)   # 两半一起存
    await speaker


_RT_PERSONA = (
    "你是这个用户认识很久的朋友，不是助手。简短、自然、像人说话。\n"
    "记忆分两种，用法完全不同：\n"
    "· MEMORY CONTEXT 里的事实——可以直接提，就像你本来就记得（"
    "「Annie 那事你还好吗」，不是「根据记录，Annie 要转学」）。\n"
    "· HOW TO SPEAK 里的内容——只影响你的语气、先说什么、什么别碰。"
    "一个字都不要说出来。听出他心情不好就先接住情绪再说事；"
    "知道他不好意思开口，就别追问；知道他讨厌什么，就绕开。\n"
    "别复述他刚说的话，别用「我记得你说过」开头，别念清单。"
)


_STRANGER = ("说话的不是你认识的那个人——声纹对不上。你对他没有任何记忆。"
             "别把别人的事讲给他听，也别猜他是谁。就当第一次见面，"
             "友好但如实地说你还不认识他。")


def _realtime_instructions(memory_context: str, stranger: bool = False) -> str:
    """人设 + 这一轮检索到的记忆。要说清楚这是「你记得的事」，否则模型会把它当成
    背景资料念出来，而不是当成自己对这个用户的记忆自然地用。"""
    if stranger:
        return f"{_RT_PERSONA}\n\n{_STRANGER}"
    if not memory_context:
        return _RT_PERSONA
    return f"{_RT_PERSONA}\n\n{memory_context}"


async def start_realtime_turn(pending, conn, send):
    """把预取好的记忆注入 Realtime session，触发这一轮的原生语音。

    只负责"发起"；收音频/文本和收尾都在常驻的事件泵里（见 realtime_session）——
    OpenAI 的事件流只能有一个消费者，每轮各读各的会串台：上一轮被打断后残留的
    response.done 会被下一轮读到，当成自己说完了。
    """
    await send({"type": "user_transcript", "text": pending.text})
    await send({"type": "memory_hits", **utils.hits_payload(pending.result, has_audio=audio_of)})
    if pending.spoken:
        await conn.input_audio_buffer.commit()
    else:
        await conn.conversation.item.create(item={"type": "message", "role": "user",
                                                  "content": [{"type": "input_text", "text": pending.text}]})
    # 记忆走 response.create 的 per-response instructions，不是 session.update。
    # 后者是会话级设置，实测更新完模型这一轮根本读不到（问"我的猫叫什么"，库里
    # 明明检索到了"叫墨墨"，模型还答"你刚提过但我没听清"）。
    await conn.response.create(response={
        "instructions": _realtime_instructions(pending.memory_context, pending.stranger),
    })
    await send({"type": "answer_start"})


async def _no_realtime(sock, err):
    """连不上 Realtime 时别让人对着 traceback 猜。

    要分清是**网络**还是**权限**：DNS/连接失败跟 key 没有关系，之前一律说成
    「key 可能没权限」，把人往错的方向指了。
    """
    name, text = type(err).__name__, str(err)
    network = (isinstance(err, (OSError, TimeoutError, ConnectionError))
               or "gaierror" in name.lower()
               or any(k in text.lower() for k in ("nodename", "temporary failure",
                                                  "name or service", "getaddrinfo",
                                                  "connection refused", "timed out")))
    if network:
        why = ("网络连不上 api.openai.com（DNS/代理/VPN 的问题，跟 key 无关）。"
               "确认能上网后重开；离线环境用 `--mode llm_tts` 也一样连不上，"
               "两条路都要访问 OpenAI。")
    elif any(k in text for k in ("401", "403", "invalid_api_key", "insufficient", "model_not_found")):
        why = ("这个 key 没有 Realtime 权限或模型不可用——改用 "
               "`python web/run.py --mode llm_tts`，那条路只要普通 chat + TTS。")
    else:
        why = ("先看这条报错本身；如果只是 Realtime 用不了，可以改用 "
               "`python web/run.py --mode llm_tts`（普通 chat + TTS）。")
    msg = f"连不上 OpenAI Realtime（{name}: {text}）。{why}"
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
    owner = {"id": ""}                    # 这场对话第一个开口的声纹
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
        if BARGE_DEBUG and st.state == "<speak>":
            print("[barge] 本地VAD 听到人声", flush=True)
        if st.text.strip() and st.text != last_partial:
            last_partial = st.text
            await sock.send_json({"type": "partial_transcript", "text": st.text, "replace": True})
        if st.turn:                                           # VAD 确认说完 → 记忆早已预取好
            last_partial = ""
            # 谁在说话。第一个开口的人算这场对话的主人；之后换了另一个声纹，
            # 就是陌生人——不能把主人的记忆讲给他听（"我是谁？"→"你是Jiaqi"
            # 这个 bug 就是因为检索从不看说话人）。
            sid = getattr(st, "speaker_id", "") or ""
            if sid and not owner["id"]:
                owner["id"] = sid
            stranger = bool(sid and owner["id"] and sid != owner["id"])
            yield Pending(st.turn.text,
                          "" if stranger else st.turn.memory_context,
                          st.turn.result, spoken=True,
                          audio_path=save_turn_audio(getattr(st, "_pcm", None)),
                          stranger=stranger)


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
            # turn_detection 只借 OpenAI 的 VAD 做**打断**，不让它接管回合：
            #   create_response=False    → 什么时候回复仍由我们决定（本地 VAD 判完
            #                              一轮、记忆预取好，才 response.create）
            #   interrupt_response=True  → 用户一开口，服务端直接掐掉正在播的回复
            # 打断判定放在 OpenAI 那侧，是因为它直接对着音频流做；本地 VAD 要等
            # 麦克风帧过完一整条链路（浏览器 AEC → ws → 重采样 → silero），真人
            # 隔着扬声器插话时信号本来就弱，很容易判不出来。
            # turn_detection 在 session.audio.input 下，**不是顶层**——写成顶层
            # 会被静默拒绝（"Unknown parameter: session.turn_detection"，只以 error
            # 事件回来，没人看就以为设上了）。原来那句 {"turn_detection": None} 一直
            # 没生效，于是 server_vad 始终开着、自动抢着回复：它生成的 response 不带
            # 我们注入的记忆，我们自己的 response.create 又撞上"已有 response 在跑"
            # 而失败——语音轮"没用上记忆"就是这么来的。
            #   create_response=False    → 什么时候回复由我们决定（本地 VAD 判完一轮、
            #                              记忆预取好，才 response.create）
            #   interrupt_response=True  → 用户一开口，服务端直接掐掉正在播的回复；
            #                              这条判定在 OpenAI 侧直接对着音频流做，比
            #                              本地 VAD（隔着 AEC + 网络 + 重采样）可靠
            await conn.session.update(session={
                "type": "realtime",
                "audio": {"input": {"turn_detection": {
                    "type": "server_vad", "create_response": False,
                    "interrupt_response": True,
                }}},
            })
            connected = True

            # 这一轮的状态：谁在说、说了什么、这轮用户的输入是什么
            turn = {"live": False, "reply": "", "pending": None}

            def close_turn():
                """一轮结束（说完或被打断）：把两半对话一起存。被打断时存的是用户
                真正听到的那部分——跟 llm_tts 那边 capture() 的取舍一致。"""
                p, reply = turn["pending"], turn["reply"]
                turn.update(live=False, reply="", pending=None)
                if p is None:
                    return
                # 存记忆失败不能连累这条会话：close_turn 是在常驻事件泵里调的，
                # 抛出去会打死那个 Task，而 Task 的异常没人 await 就被静默丢弃——
                # 表现是"整个会话突然不响应了，日志里一个字都没有"。
                try:
                    vm.ingest(p.text, agent_reply=reply, async_facts=True,
                              audio=p.audio_path or None)
                except Exception as e:
                    print(f"[web] 存这一轮失败：{type(e).__name__}: {e}", flush=True)

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
                    elif t == "error" or t.endswith(".error"):
                        err = getattr(ev, "error", None)
                        # server_vad 判完一句会自己 commit 音频缓冲，我们随后那次
                        # commit 就撞上空缓冲。两种情况都得留着手动 commit（说得太短
                        # 时 server_vad 不会自动 commit），所以这条属于预期内，忽略。
                        # response_cancel_not_active：打断有两条路（本地 VAD 的
                        # on_speech + server_vad 的 interrupt_response），互为备份，
                        # 谁先到算谁的，慢的那个扑空是正常的。
                        if getattr(err, "code", "") not in (
                                "input_audio_buffer_commit_empty",
                                "response_cancel_not_active"):
                            print(f"[web] realtime 事件错误：{err or ev}", flush=True)
                    elif t.endswith("input_audio_buffer.speech_started"):
                        # OpenAI 的 VAD 听到人声：它那侧已经掐了回复，我们同步收尾
                        if BARGE_DEBUG:
                            print(f"[barge] OpenAI VAD 听到人声 (live={turn['live']})", flush=True)
                        if turn["live"]:
                            turn["live"] = False
                            await sock.send_json({"type": "answer_interrupt"})
                            close_turn()
                    elif t.endswith("response.done") or t.endswith("response.cancelled"):
                        if turn["live"]:
                            await sock.send_json({"type": "answer_done"})
                            close_turn()

            async def on_frame(raw):
                await conn.input_audio_buffer.append(audio=base64.b64encode(raw).decode())

            async def on_speech():
                """用户在助手说话时开口 → 打断。幂等：live 一置 False 就不再触发。"""
                if not turn["live"]:
                    if BARGE_DEBUG:
                        print("[barge] 有人声但助手没在说，忽略", flush=True)
                    return
                if BARGE_DEBUG:
                    print("[barge] ★ 打断：本地VAD 触发", flush=True)
                turn["live"] = False
                # 先通知前端停播——这是本地操作，立刻生效；而 response.cancel() 要
                # 等一次 OpenAI 往返。之前顺序反了，人插话后还得听完那一个往返的
                # 时间，听感就是"打断没用，他非要说完"。
                await sock.send_json({"type": "answer_interrupt"})
                close_turn()
                await conn.response.cancel()

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


#: 偏好类的词面信号——右脑只有 heartnote / response_experience 两个 class，
#: 光靠 class 分不出脑图上的 emotion / preference / experiences 三簇。
_PREF_WORDS = ("喜欢", "爱好", "偏好", "习惯", "规律", "不吃", "过敏", "素食", "讨厌", "不喜欢", "放松")
_CALM = ("", "平静", "中性")


def _rb_cluster(m) -> str:
    """右脑记忆归到脑图哪个簇。0 LLM，只看已有字段。"""
    if str(getattr(m, "memory_class", "")) == "response_experience":
        return "experiences"                      # 回复经验：怎么跟这个人说话
    if any(w in (m.content or "") for w in _PREF_WORDS):
        return "preference"                       # 口味、习惯、忌口
    emo = (getattr(m, "metadata", None) or {}).get("emotion", "")
    return "emotion" if emo not in _CALM else "experiences"


def audio_of(memory_id: str) -> str:
    """这条记忆当时那段原声在哪；没归档过、或已过保留期被清掉，返回 ""。

    走核心的 GetOriginalAudio——它已经带了"文件还在不在"的检查，不用在这儿重写。
    """
    try:
        r = vm._o._audio.GetOriginalAudio(memory_id)
        return r.get("audio_path") or "" if r.get("found") else ""
    except Exception as e:
        print(f"[web] 查存档音频失败：{e}", flush=True)
        return ""


def memory_snapshot(limit: int = 48) -> dict:
    """库里已有的记忆，供前端在打开页面时把脑图先铺满。

    只读，不碰模型：左脑走 list_entries + 认知图的 slot 标注，右脑走 list_all。
    库是空的（新用户）就返回空列表，前端照旧从空图开始长。
    """
    from voicemem.leftbrain.cognitive_graph.types import SlotV2

    uid = vm._o._user_id
    left, right = [], []
    try:
        repo = vm._o._get_repo()
        entries = repo._vector_store.list_entries(user_id=uid)
        # slot 标在认知图里，不在记忆条目上——先建 id -> slot 的反查表
        cog = repo._cognitive_store
        slot_of = {}
        for slot in SlotV2:
            for mid in cog.memory_ids_for_slots(uid, [slot]):
                slot_of.setdefault(mid, slot.value)
        for e in entries[:limit]:
            # list_entries 的 date 直接截了 time_start 前 10 位，遇到纯时间串会切出
            # "09:20:37" 这种。不像日期就置空，别把垃圾送到前端。
            d = str(e.get("date", ""))
            left.append({"text": e["text"], "date": d if d[:4].isdigit() else "",
                         "slot": slot_of.get(e["id"], "daily_life")})
    except Exception as e:
        print(f"[web] 左脑快照读取失败：{e}", flush=True)
    try:
        for m in vm._o._right._rb_repo().list_all(uid)[:limit]:
            right.append({"text": m.content, "cluster": _rb_cluster(m)})
    except Exception as e:
        print(f"[web] 右脑快照读取失败：{e}", flush=True)
    return {"left": left, "right": right}


app = utils.build_app(MODE, realtime_session if MODE == "realtime" else llm_tts_session, vm.classify, memory_snapshot, audio_of)


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
