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
import re
import sys
import time
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
    p.add_argument("--memory_root",
                   default=os.environ.get("VOICEMEM_MEMORY_ROOT")
                   or str(_ROOT / "results" / "voice_memory"),
                   help="记忆库目录")
    return p.parse_args(argv)


ARGS = _parse(None if __name__ == "__main__" else [])

import utils                                         # noqa: E402  同目录管道层
from voicemem import VoiceMem                        # noqa: E402

BARGE_DEBUG = os.environ.get("BARGE_DEBUG", "1") != "0"
BARGE_THRESHOLD = float(os.environ.get("BARGE_THRESHOLD", "0.45"))  # 越小越容易被打断
#: 转写要比上次多出这么多个字，才算"他真的插话了"。
#: 之前这里是「连续人声 ≥280ms」，纯 VAD 判定太松——咳嗽、关门、AEC 没压干净的
#: 助手回声都算人声，日志里一串"连续人声 → 请求打断 / 助手没在说，忽略"在空转。
#: 换成等 ASR 真的吐出字，代价是多等一次出字（~200-300ms），换来不会自己掐自己。
BARGE_MIN_CHARS = int(os.environ.get("BARGE_MIN_CHARS", "2"))
#: 助手刚开口那一小段不允许被打断——那时候麦克风里几乎只有它自己的声音。
BARGE_GRACE_MS = int(os.environ.get("BARGE_GRACE_MS", "500"))
#: OpenAI 那侧用哪种回合/打断判定。semantic_vad 由模型判"这是不是真的在打断"，
#: 对 backchannel（"嗯""对""哦"）不敏感；server_vad 只看有没有声音，所以助手自己
#: 的回声、环境噪声都能把它掐了。模型或 SDK 不支持时会以 error 事件回来（不抛），
#: 日志里看到就改回 TURN_DETECTION=server_vad。
TURN_DETECTION = os.environ.get("TURN_DETECTION", "semantic_vad")
#: semantic_vad 的抢答倾向：low 更愿意等你说完，high 更爱抢。
VAD_EAGERNESS = os.environ.get("VAD_EAGERNESS", "low")
MIC_RATE = 24000                       # 前端上行的采样率（index.html 的 SAMPLE_RATE）
#: 按声纹拦陌生人。默认开——启动时预热过、又在后台线程算，实测对延迟零影响
#: （memory_hits 仍在 EOU 前 0.63s 到达，跟关掉时一样）。
SPEAKER_GATE = os.environ.get("VOICEMEM_SPEAKER_GATE", "1") != "0"   # 打断为什么没触发：看这几行日志
#: 每轮都打一行说话人判定（默认只在判成陌生人时打）。
SPEAKER_DEBUG = os.environ.get("SPEAKER_DEBUG", "0") != "0"
MODE = ARGS.mode                                     # llm_tts | realtime
SPEC_MIN_CHARS = ARGS.spec_min_chars                 # partial 起投机
GAMBLE_S  = ARGS.gamble_ms / 1000                    # 赌说完
CONFIRM_S = ARGS.confirm_ms / 1000                   # VAD 确认结束

_RT_PERSONA = (
    "你是这个用户认识很久的朋友，不是助手。简短、自然、像人说话。\n"
    "记忆分两种，用法完全不同：\n"
    "· MEMORY CONTEXT 里的事实——可以直接提，就像你本来就记得（"
    "「Annie 那事你还好吗」，不是「根据记录，Annie 要转学」）。\n"
    "· HOW TO SPEAK 里的内容——只影响你的语气、先说什么、什么别碰。"
    "一个字都不要说出来。听出他心情不好就先接住情绪再说事；"
    "知道他不好意思开口，就别追问；知道他讨厌什么，就绕开。\n"
    "别复述他刚说的话，别用「我记得你说过」开头，别念清单。\n"
    "说话方式：像朋友聊天那样有起伏——该笑就笑出来，替他高兴就热一点，"
    "他难过就把语速放慢、声音压低。用「嗯」「哎」「诶」这种口头反应开头，"
    "不要播音腔，不要每句都四平八稳。句子短，一次说一两句就停。"
)


_STRANGER = ("说话的不是你认识的那个人——声纹对不上。你对他没有任何记忆。"
             "别把别人的事讲给他听，也别猜他是谁。就当第一次见面，"
             "友好但如实地说你还不认识他。")


def _turn_detection() -> dict:
    """OpenAI 那侧的回合/打断判定。两种 provider 的参数**不通用**——semantic_vad
    不吃 threshold / prefix_padding_ms / silence_duration_ms，传了整段会被静默拒绝
    （只以 error 事件回来）。所以两套各写各的，别合并。

    create_response=False   什么时候回复由我们决定（本地判完一轮、记忆预取好）
    interrupt_response=True 用户一开口，服务端直接掐掉正在播的回复
    """
    base = {"type": TURN_DETECTION, "create_response": False, "interrupt_response": True}
    if TURN_DETECTION == "semantic_vad":
        return {**base, "eagerness": VAD_EAGERNESS}
    # server_vad：默认 0.5 对插话太钝——人隔着扬声器说话，回声消除处理过之后
    # 信号本来就弱，够不到阈值就等于打不断。
    return {**base, "threshold": BARGE_THRESHOLD,
            "prefix_padding_ms": 200, "silence_duration_ms": 320}


def _realtime_instructions(memory_context: str, stranger: bool = False) -> str:
    """人设 + 这一轮检索到的记忆。要说清楚这是「你记得的事」，否则模型会把它当成
    背景资料念出来，而不是当成自己对这个用户的记忆自然地用。"""
    if stranger:
        return f"{_RT_PERSONA}\n\n{_STRANGER}"
    if not memory_context:
        return _RT_PERSONA
    return f"{_RT_PERSONA}\n\n{memory_context}"


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
        "llm":      {"provider": "openai", "config": {"model": utils.CHAT_MODEL,
                                                      "system": _RT_PERSONA}},
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


async def voicemem_llm_tts(pending, send, send_audio, owner):
    """记忆已在关键路径外预取好：LLM 流式回复 → TTS 流式语音。

    TTS 跟生成**并行**：LLM 吐满一句就丢进队列，另一条协程取出来合成、发音频。
    等全文生成完再开始合成的话，文本早打完了、音频还没起头（实测 TTS 首帧就要
    ~1.2s，加上生成那几秒，用户看着字干等）。
    """
    await send({"type": "user_transcript", "text": pending.text})
    await send({"type": "memory_hits", **utils.hits_payload(pending.result, has_audio=audio_of, cluster_of=hit_cluster)})
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
    interrupted = False
    try:
        # 跟 realtime 用同一份指令：两条路必须表现一致，否则换个 --mode
        # 人设和「右脑不许念出来」的约束就悄悄没了。
        # 走核心回复层（人设在 CONFIG.reply.llm.config.system，见 voicemem/reply.py
        # 的 compose_system：system + memory_context，和 realtime 那条拼出来的一样）。
        async for d in vm.reply_stream(
                pending.text, _STRANGER if pending.stranger else pending.memory_context):
            reply += d
            buf += d
            await send({"type": "answer_delta", "text": d})
            if _cut_point(buf, first=sent == 0):
                await queue.put(buf.strip())
                buf, sent = "", sent + 1
        if buf.strip():
            await queue.put(buf.strip())
    except asyncio.CancelledError:
        interrupted = True                      # 用户插话了，这一轮到此为止
    finally:
        await queue.put(None)                   # 生成出错也要让 speak() 收工

    if interrupted:
        # 别把 speak() 留在后台继续往一条已经停播的连接上发音频
        speaker.cancel()
    else:
        await speaker
        await send({"type": "answer_done"})

    # 存这一轮：被打断时存的是用户真正听到的那半句。
    # 先落记忆再收工——ingest 排在音频后面的话，用户一听完就关页面（语音场景很
    # 常见），这一轮就永远存不进去。async_facts=True：抽事实走后台。
    remember_turn(pending, reply, owner)



async def start_realtime_turn(pending, conn, send):
    """把预取好的记忆注入 Realtime session，触发这一轮的原生语音。

    只负责"发起"；收音频/文本和收尾都在常驻的事件泵里（见 realtime_session）——
    OpenAI 的事件流只能有一个消费者，每轮各读各的会串台：上一轮被打断后残留的
    response.done 会被下一轮读到，当成自己说完了。
    """
    await send({"type": "user_transcript", "text": pending.text})
    await send({"type": "memory_hits", **utils.hits_payload(pending.result, has_audio=audio_of, cluster_of=hit_cluster)})
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

def remember_turn(pending, reply: str, owner: dict) -> None:
    """存这一轮，并顺手记下说话人是谁。

    说话人不用单独算：ingest 内部本来就要跑一次 preprocess（场景/声纹/情绪），
    返回值里直接带 speaker_id。之前我在热路径上又单独触发了一次完整感知——
    那套一次 424ms（AST 占 361ms），纯属重复劳动，而且挡在读 socket 前面。
    """
    try:
        r = vm.ingest(pending.text, agent_reply=reply, async_facts=True,
                      audio=pending.audio_path or None) or {}
    except Exception as e:
        print(f"[web] 存这一轮失败：{type(e).__name__}: {e}", flush=True)
        return
    sid = r.get("speaker_id") or ""
    if not sid:
        return
    if not owner["id"]:
        owner["id"] = sid                  # 第一个开口的算这场对话的主人
    owner["last"] = sid


async def anticipate(sock, on_frame=None, on_speech=None, owner=None):
    """驱动核心流式会话，逐个 yield 确认回合的 Pending。
    on_frame(raw24k)：realtime 用它把原始音频平行喂给 OpenAI（方案 A）。
    on_speech()：本地 VAD 一听到人声就叫一次——realtime 拿它做打断（barge-in）。"""
    stream = vm.stream(spec_min_chars=SPEC_MIN_CHARS, gamble_s=GAMBLE_S, confirm_s=CONFIRM_S)
    last_partial = ""
    if owner is None:
        owner = {"id": "", "last": ""}    # 主人的声纹 / 上一轮是谁
    barge_base = 0                        # 上次触发打断时的转写长度
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
        # 打断走「ASR 确认制」：不是听到人声就掐，而是等转写真的多出几个字。
        # 助手的回声进了 ASR 也转不出连贯的新字，咳嗽和关门声更不会——这一条
        # 比任何 VAD 阈值都好使，见 BARGE_MIN_CHARS 上面那段。
        grown = len(st.text.strip()) - barge_base
        if grown >= BARGE_MIN_CHARS and on_speech:
            barge_base = len(st.text.strip())
            if BARGE_DEBUG:
                print(f"[barge] 转写多出 {grown} 个字 → 请求打断：{st.text[-12:]!r}", flush=True)
            await on_speech()
        if st.text.strip() and st.text != last_partial:
            last_partial = st.text
            await sock.send_json({"type": "partial_transcript", "text": st.text, "replace": True})
        if st.turn:                                           # VAD 确认说完 → 记忆早已预取好
            last_partial = ""
            barge_base = 0                                    # 新一轮，转写从头开始涨
            # 谁在说话。第一个开口的人算这场对话的主人；之后换了另一个声纹，
            # 就是陌生人——不能把主人的记忆讲给他听（"我是谁？"→"你是Jiaqi"
            # 这个 bug 就是因为检索从不看说话人）。
            # 用上一轮算出来的说话人做判断，这一轮的放后台算。
            # 直接读 st.speaker_id 会同步跑声纹+情绪+场景一整套模型——实测 2.1 秒，
            # 而且是在事件循环里，这期间连 socket 都不读，麦克风帧全堆着，
            # 表现就是"ASR 很卡"。代价是换人之后第一句仍按上一个人算。
            stranger = bool(SPEAKER_GATE and owner["id"] and owner["last"]
                            and owner["last"] != owner["id"])
            if stranger or SPEAKER_DEBUG:
                # "他怎么突然不认识我了"——看这一行。声纹把同一个人认成两个
                # person_* 时就会这样：记忆被清空，指令换成"就当第一次见面"。
                print(f"[speaker] owner={owner['id'] or '-'} last={owner['last'] or '-'}"
                      f" stranger={stranger}", flush=True)
            yield Pending(st.turn.text,
                          "" if stranger else st.turn.memory_context,
                          st.turn.result, spoken=True,
                          audio_path=await asyncio.to_thread(
                              save_turn_audio, getattr(st, "_pcm", None)),
                          stranger=stranger)



# ══════════════════ 每种 mode 的会话循环 ══════════════════

async def llm_tts_session(sock):
    """llm_tts 这条路的打断。

    原来是 `async for pending in anticipate(sock): await voicemem_llm_tts(...)`——
    两个问题叠在一起，打断在结构上就不可能：
      · on_speech 没传进 anticipate，本地 VAD 听到人声也没人管；
      · voicemem_llm_tts 最后 `await speaker`，要等全部音频发完才返回。async for
        在这期间不会去拉下一个，anticipate 就停在那儿不读 socket 了，麦克风帧全
        堆在缓冲区里。听感就是"说什么都没用，他非要说完"。

    现在回复丢进后台任务，读 socket 的循环一刻不停；听到人声就取消那个任务。
    """
    turn = {"task": None, "t0": 0.0}
    owner = {"id": "", "last": ""}

    async def stop_reply():
        task = turn["task"]
        if task is None or task.done():
            return
        since = (time.monotonic() - turn["t0"]) * 1000
        if since < BARGE_GRACE_MS:
            if BARGE_DEBUG:
                print(f"[barge] 才说了 {since:.0f}ms，还在宽限期内，不打断", flush=True)
            return
        task.cancel()
        turn["task"] = None
        try:
            await sock.send_json({"type": "answer_interrupt"})   # 前端停播已排队的音频
        except Exception:
            pass

    async for pending in anticipate(sock, on_speech=stop_reply, owner=owner):
        await stop_reply()                    # 上一轮还没说完就被新的一轮顶掉
        turn["t0"] = time.monotonic()
        turn["task"] = asyncio.create_task(
            voicemem_llm_tts(pending, sock.send_json, sock.send_bytes, owner))


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
                "audio": {
                    "input": {"turn_detection": _turn_detection()},
                    "output": {"voice": utils.RT_VOICE},
                },
            })
            connected = True

            # 这一轮的状态：谁在说、说了什么、这轮用户的输入是什么。
            # until：前端预计几点才把已发出去的音频播完（见 hearing()）。
            turn = {"live": False, "reply": "", "pending": None,
                    "t0": 0.0, "until": 0.0}
            owner = {"id": "", "last": ""}

            def hearing() -> bool:
                """用户此刻还听不听得见助手。

                不能用 turn["live"] 代替：realtime 推音频比实时播放快得多，一段
                十几秒的回复两三秒就推完了，response.done 一到 live 就变 False，
                可前端那边还在播剩下的十几秒。这时候用户插话，on_speech 看到
                live=False 就"忽略"，前端从没收到 answer_interrupt——表现正是
                "打断没反应，它非要念完"。
                所以按**已发出去的音频时长**算：24k PCM16，一个样本 2 字节。
                """
                return turn["live"] or time.monotonic() < turn["until"]

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
                remember_turn(p, reply, owner)

            async def pump():
                """常驻事件泵：OpenAI 的事件流只有这一个消费者。

                turn["live"] 为假时一律不往前端转发——被打断后 OpenAI 还会吐一会儿
                残余音频，转过去的话前端刚 stopPlayback 又排上新的，打断就不干净了。
                """
                async for ev in conn:
                    t = getattr(ev, "type", "")
                    if t.endswith("output_audio.delta"):
                        if turn["live"]:
                            pcm = base64.b64decode(ev.delta)
                            # 前端是排队播的（index.html 的 nextPlay），这里跟着算
                            # 同一条时间线：上一块播完之后再接这一块。
                            turn["until"] = (max(turn["until"], time.monotonic())
                                             + len(pcm) / 2 / 24000)
                            await sock.send_bytes(pcm)
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
                        since = (time.monotonic() - turn["t0"]) * 1000
                        if BARGE_DEBUG:
                            print(f"[barge] OpenAI VAD 听到人声 (live={turn['live']}, "
                                  f"还在播={hearing()}, {since:.0f}ms)", flush=True)
                        # 宽限期，跟本地那条路一样。本地 VAD 判完一轮（静音 500ms）就
                        # 发 response.create，而 OpenAI 的 server_vad 只要 320ms 静音就
                        # 认为下一句开始了——用户的话尾、呼吸声、环境噪声都够触发。
                        # 没有这道门的话，speech_started 会在助手出声之前就到，
                        # 回复被掐在第一个音频块之前：一个字都听不见，还不报错。
                        if hearing() and since >= BARGE_GRACE_MS:
                            turn["live"], turn["until"] = False, 0.0
                            await sock.send_json({"type": "answer_interrupt"})
                            close_turn()
                    elif t.endswith("response.done") or t.endswith("response.cancelled"):
                        if turn["live"]:
                            await sock.send_json({"type": "answer_done"})
                            close_turn()

            async def on_frame(raw):
                await conn.input_audio_buffer.append(audio=base64.b64encode(raw).decode())

            async def on_speech():
                """用户在助手说话时开口 → 打断。幂等：hearing() 一变假就不再触发。"""
                if not hearing():
                    if BARGE_DEBUG:
                        print("[barge] 有人声但助手没在说，忽略", flush=True)
                    return
                # 刚开口那一小段不许打断：那时候麦克风里几乎只有助手自己的声音，
                # 回声消除还没跟上，很容易一出声就把自己掐了。
                since = (time.monotonic() - turn["t0"]) * 1000
                if since < BARGE_GRACE_MS:
                    if BARGE_DEBUG:
                        print(f"[barge] 才说了 {since:.0f}ms，还在宽限期内，不打断", flush=True)
                    return
                if BARGE_DEBUG:
                    left = max(0.0, turn["until"] - time.monotonic()) * 1000
                    print(f"[barge] ★ 打断：转写触发（前端还剩 {left:.0f}ms 没播完）",
                          flush=True)
                turn["live"], turn["until"] = False, 0.0
                # 先通知前端停播——这是本地操作，立刻生效；而 response.cancel() 要
                # 等一次 OpenAI 往返。之前顺序反了，人插话后还得听完那一个往返的
                # 时间，听感就是"打断没用，他非要说完"。
                await sock.send_json({"type": "answer_interrupt"})
                close_turn()
                await conn.response.cancel()

            pump_task = asyncio.create_task(pump())
            try:
                async for pending in anticipate(sock, on_frame=on_frame,
                                                on_speech=on_speech, owner=owner):
                    if hearing():                        # 上一轮还没播完就被新的一轮顶掉
                        await on_speech()
                    turn.update(live=True, reply="", pending=pending,
                                t0=time.monotonic(), until=0.0)
                    await start_realtime_turn(pending, conn, sock.send_json)
            finally:
                pump_task.cancel()
    except Exception as e:
        if connected:
            raise
        await _no_realtime(sock, e)


#: 右脑 slot → 脑图三个簇。
#:
#: 右脑真正的分类单位是 rb_slots 里那 6 个 slot（情绪 / 喜好与厌恶 / 应对方式 /
#: 表达风格 / 思维模式 / 人物地点态度），不是 memory_class——那只有 heartnote /
#: response_experience 两种，分不出东西。脑图上只有三个簇，所以这里把 6 个 slot
#: 收敛成 3 个。
#:
#: 检索命中的内容里，slot 名就写在开头（"情绪：…""喜好与厌恶：…"）；
#: heartnote 是一条条的情绪时刻（"情感记录：…（内心OS：【难过】…）"），归 emotion。
SLOT_TO_CLUSTER = {
    "情绪":         "emotion",
    "情感记录":     "emotion",
    "内心OS":       "emotion",
    "喜好与厌恶":   "preference",
    "思维模式":     "preference",
    "应对方式":     "experiences",
    "表达风格":     "experiences",
    "人物地点态度": "experiences",
    "避免重复":     "experiences",
}
_CALM = ("", "平静", "中性")


def rb_cluster(content: str, memory_class: str = "", emotion: str = "") -> str:
    """一条右脑记忆归到脑图哪个簇。0 LLM，只看 slot 名。"""
    # 先剥掉 "[2026-06-20] " 这种日期前缀，否则它占满取来比对的那一小段，
    # heartnote 的"情感记录"就落到窗口外了。
    text = re.sub(r"^\s*\[[0-9-]{6,12}\]\s*", "", content or "")
    head = text[:14]
    for slot, cluster in SLOT_TO_CLUSTER.items():
        if slot in head:
            return cluster
    if str(memory_class) == "response_experience":
        return "experiences"
    if emotion not in _CALM:
        return "emotion"
    return "experiences"


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


def hit_cluster(content: str, source: str) -> str:
    """给检索命中用：只有 content 和 source，没有 metadata。"""
    return rb_cluster(content, source, "")


def _rb_cluster(m) -> str:
    """给快照用：从 RightBrainMemory 对象取字段。"""
    meta = getattr(m, "metadata", None) or {}
    return rb_cluster(getattr(m, "content", ""),
                      str(getattr(m, "memory_class", "")),
                      meta.get("emotion", ""))


def fact_index(uid: str) -> dict:
    """左脑记忆 id → 事实原文。

    原文在向量库里，认知图的 memories 表只有 id/slot/热度这些，取不到文本——
    一开始用 get_memory_record 取，结果每条 heartnote 的起因都是空的。
    """
    try:
        entries = vm._o._get_repo()._vector_store.list_entries(user_id=uid)
        return {e["id"]: e["text"] for e in entries}
    except Exception as e:
        print(f"[web] 读左脑事实失败：{e}", flush=True)
        return {}


def right_brain_tree(uid: str, facts: dict) -> list:
    """右脑真实的三层结构：slot → entity → 挂在下面的 heartnote。

    脑图上的节点是 **entity**（"委屈""讨厌坚果和过敏""选择沉默忍耐"），不是
    一条条 heartnote —— entity 才是右脑归纳出来的那个"点"，heartnote 是支撑它
    的证据。每条 heartnote 再带上引发它的左脑事实，这样"为什么委屈"点两下就能看到。

    只读，全部走 graph_store 的公开方法。
    """
    graph = vm._o._right._rb_graph_store()
    repo = vm._o._right._rb_repo()
    notes = {}
    for m in repo.list_all(uid):
        meta = getattr(m, "metadata", None) or {}
        notes[m.id] = {
            "text": m.content,
            "emotion": meta.get("emotion", ""),
            "cause": facts.get(meta.get("left_memory_id", ""), ""),
        }

    out = []
    for slot in graph.list_slots(uid):
        cluster = SLOT_TO_CLUSTER.get(slot.name, "experiences")
        for ent in graph.get_entities_for_slot(uid, slot.id):
            mids = graph.get_memories_for_entity(ent.id)
            out.append({
                "cluster": cluster,
                "slot": slot.name,
                "text": ent.name,                      # 脑图上显示的就是这个
                "desc": getattr(ent, "description", "") or "",
                "notes": [notes[i] for i in mids if i in notes],
            })
    return out


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
            # 带上这条记忆挂了哪些实体：前端据此在讲同一个人/同一件事的两条记忆
            # 之间连线——脑图上的连线才对应真实关系，而不是随便连。
            try:
                ents = cog.entity_ids_for_memory(e["id"])
            except Exception:
                ents = []
            left.append({"text": e["text"], "date": d if d[:4].isdigit() else "",
                         "slot": slot_of.get(e["id"], "daily_life"),
                         "entities": list(ents)[:6]})
    except Exception as e:
        print(f"[web] 左脑快照读取失败：{e}", flush=True)
    try:
        right = right_brain_tree(uid, fact_index(uid))
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
    # 感知那套（场景 AST / 声纹 3D-Speaker / 情绪 SenseVoice）也要预热。
    # 实测第一次 preprocess 要 2120ms（全在加载模型），之后稳定在 340–410ms。
    # 不预热的话那 2 秒会落在用户说完第一句的时候，正是最不该卡的位置。
    print("[web] 预热 感知（场景 / 声纹 / 情绪）…", flush=True)
    import tempfile
    import soundfile as _sf
    _warm = Path(tempfile.gettempdir()) / "voicemem_warmup.wav"
    _sf.write(_warm, _np.zeros(16000, dtype=_np.float32), 16000)
    try:
        vm.preprocess("预热", audio=str(_warm))
    except Exception as _e:
        print(f"[web] 感知预热跳过：{_e}", flush=True)
    finally:
        _warm.unlink(missing_ok=True)
    print("[web] 就绪", flush=True)
    uvicorn.run(app, host=ARGS.host, port=ARGS.port)
