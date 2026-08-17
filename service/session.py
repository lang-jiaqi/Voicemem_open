"""语音助手主类。组合所有小对象,实现完整时序逻辑。

时序(每个新 turn 一个 +1 的 global_id):
  说话中            -> 推 partial(浅灰实时文本)
  停顿满 200ms      -> prepare: 分析(文本/说话人/情感/实体/簇),调 CognitiveService.prepare()
                       起搜索,推前端(未锁定)
  200-500ms 再开口  -> 继续累积同一 turn
  停顿满 500ms      -> ready: 调 CognitiveService.ready() 拿 info,推前端(已锁定),
                       并把(语音/转录/情感/info)交给 _answer()
  不足时长          -> 不做说话人/情感/路由
  碎字(不足N字)     -> 并入上一句,不新开 turn

跟原来 assistant.py 的差异：
  - 不再走 HTTP 调 cognitive_server.py，直接进程内调 CognitiveService（见 cognitive.py）。
  - 去掉了 ECAPA 跨 session 声纹匹配（_spk_audio/_spk_person/EcapaEncoder）——
    这块在当前协议里其实已经是死代码：服务端本来就不读 voiceprint_vec/
    confirmed_person_id，也从没往回传过 person_id，这个分支从没真正触发过。
    跨 session 认人现在完全交给 voicemem 核心的 CAM++（Ingest() 返回的
    info_detail.audiomem_info.speaker_id 就是 CAM++ 认出来的 person_id）。
"""
from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import numpy as np

from asr import pick_device, StreamingASR, Transcriber
from live_signals import EmotionRecognizer, SpeakerId, Vad
from router import Encoder, EntityExtractor, MemoryRouter
from cognitive import CognitiveService
from tts import LiveTTS
from turn import (
    Turn, now_str, char_count, pad_audio,
    FRAME_SAMPLES, SILENCE_PREVIEW, SILENCE_PREFETCH, SILENCE_LOCK,
    BARGE_MIN_VOICE, MIN_VOICE_SEC, MIN_TURN_CHARS,
)
from asr import SAMPLE_RATE

CATEGORIES = {
    "工作": "上班、职场、老板同事、任务项目、会议、职业发展",
    "经济": "金钱、工资收入、开销花费、存钱预算、买东西、理财",
    "人际关系": "朋友、家人、同事、恋人之间的相处、交往、矛盾",
    "健康": "身体、生病吃药、看医生、运动锻炼、饮食睡眠作息",
    "目标": "计划、想要达成的事、减肥升职、未来打算、待办",
    "日常": "everyday life、吃饭散步、天气、看电影、做家务等琐事",
    "事实": "客观信息、时间日期、地点、常识、不带个人色彩的陈述",
}
FORCE_NOUNS = ["鸡", "鸭", "鱼", "肉", "蛋", "奶", "茶", "酒", "药", "水", "饭"]
EMOTION_BACKEND = os.environ.get("EMOTION_BACKEND", "sensevoice").lower()


class SharedModels:
    """服务进程级的只读模型，所有浏览器会话复用。

    SenseVoice、emotion2vec 和 model2vec 的权重加载/GPU 首次执行很慢，但它们不保存
    单个会话状态；将其放在这里可把首句冷启动移到 ws_server 启动阶段。流式 ASR、VAD、
    SpeakerId 仍按会话新建，因为它们各自保存流状态。
    """

    def __init__(self, device: str | None = None) -> None:
        self.device = device or pick_device()
        print(f"[warmup] 加载共享模型到 {self.device} ...", flush=True)
        self.transcriber = Transcriber(self.device)
        # SenseVoice 已同时输出情感 token；仅在显式要求旧模型时再加载 emotion2vec。
        self.emotion = EmotionRecognizer(self.device) if EMOTION_BACKEND == "emotion2vec" else None
        self.router = MemoryRouter(Encoder(), EntityExtractor(force_nouns=FORCE_NOUNS))
        self.router.set_categories(CATEGORIES)

    def warmup(self) -> None:
        """跑一次最小推理，触发 CUDA 初始化与模型首轮 lazy 初始化。"""
        silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
        self.transcriber.run(silence)
        if self.emotion is not None:
            self.emotion.run(silence)
        self.router.route("你好")
        print("[warmup] 共享模型预热完成", flush=True)


# ====== 文字情绪(看内容,补声学情绪之不足)======
# 声学模型只听语气;这里从转写文本里直接命中情绪词,平淡语气说出的情绪词也能抓到。
_EMO_LEXICON = [
    ("开心", ["开心", "高兴", "快乐", "兴奋", "太好了", "好棒", "爽", "开森", "哈哈",
            "嘻嘻", "happy", "excited", "glad", "great", "awesome"]),
    ("愤怒", ["生气", "愤怒", "火大", "气死", "气炸", "恼火", "烦死", "气死了",
            "angry", "mad", "furious", "pissed"]),
    ("悲伤", ["难过", "伤心", "失落", "沮丧", "低落", "郁闷", "想哭", "委屈", "失望",
            "心碎", "难受", "不开心", "不高兴", "没意思", "emo",
            "sad", "down", "upset", "unhappy", "depressed"]),
    ("恐惧", ["害怕", "恐惧", "紧张", "担心", "焦虑", "慌", "吓", "scared",
            "afraid", "nervous", "worried", "anxious"]),
    ("惊讶", ["惊讶", "震惊", "没想到", "居然", "竟然", "天哪", "天呐", "我的天",
            "wow", "surprised", "shocked"]),
    ("厌恶", ["恶心", "厌恶", "嫌弃", "反感", "disgusting", "gross"]),
]
_EMO_NEG = ("不", "没", "别", "无", "莫")


def text_emotion(text: str) -> str | None:
    """从文本命中情绪词,返回中文情绪标签;命中前若紧跟否定词则跳过。无命中返回 None。"""
    if not text:
        return None
    low = text.lower()
    best, best_hits = None, 0
    for label, kws in _EMO_LEXICON:
        hits = 0
        for kw in kws:
            start = 0
            while True:
                i = low.find(kw, start)
                if i < 0:
                    break
                pre = text[max(0, i - 2):i]
                if kw not in ("不开心", "不高兴", "没意思") and any(n in pre for n in _EMO_NEG):
                    pass
                else:
                    hits += 1
                start = i + len(kw)
        if hits > best_hits:
            best, best_hits = label, hits
    return best


def reply_lang(transcript: str) -> str:
    """判断用户这句话主要是中文还是英文,用于锁定回复语言。"""
    import re
    cjk = len(re.findall(r"[一-鿿]", transcript or ""))
    lat = len(re.findall(r"[a-zA-Z]", transcript or ""))
    if lat > 0 and cjk == 0:
        return "en"
    if cjk > lat:
        return "zh"
    return "en" if lat >= cjk else "zh"


def fuse_emotion(acoustic: str, textual: str | None) -> str:
    """融合声学情绪与文字情绪。
    - 文字命中情绪词是强信号(内容),优先采用;
    - emotion2vec 对平常语音"过度自信地报悲伤",误报最重,所以声学的"悲伤"在没有
      文字佐证时不单独采信,降级为中性;
    - 其它声学情绪(愤怒/开心等)照常采用。"""
    if textual:
        return textual
    if acoustic == "悲伤":
        return "中性"
    return acoustic


class VoiceAssistant:
    """最大类:组合所有模块,处理音频流,驱动整个时序。"""

    def __init__(self, models_dir: str, cognitive: CognitiveService, send_to_ui,
                 shared_models: SharedModels | None = None):
        device = shared_models.device if shared_models is not None else pick_device()
        print("使用设备:", device)
        asr_dir = f"{models_dir}/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"

        # 小对象
        self.asr = StreamingASR(asr_dir)
        self.transcriber = (shared_models.transcriber if shared_models is not None
                            else Transcriber(device))
        self.emotion = (shared_models.emotion if shared_models is not None
                        else (EmotionRecognizer(device) if EMOTION_BACKEND == "emotion2vec" else None))
        self.speaker = SpeakerId(
            f"{models_dir}/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx")
        self.vad = Vad(f"{models_dir}/silero_vad.onnx")
        self.router = (shared_models.router if shared_models is not None
                       else MemoryRouter(Encoder(), EntityExtractor(force_nouns=FORCE_NOUNS)))
        if shared_models is None:
            self.router.set_categories(CATEGORIES)
        self.cognitive = cognitive

        self.send_to_ui = send_to_ui      # async fn(dict|bytes) -> None,推给前端
        self.last_voice_ts = 0.0
        self.turn: Turn | None = None
        self.turn_counter = 0
        # 上一个已锁定 turn(供碎字并入)
        self.last_text = ""
        self.last_spk = "-"
        self.last_emotion = "-"
        self.has_last = False

        # 回答(Gemini Live)
        self.live: LiveTTS | None = None
        self._answering = False           # 是否正在播一段回答(用于 barge-in)
        self._barge_voice = 0.0           # 回答进行中累计的"持续说话"时长,够了才打断

    # ---------- 生命周期 ----------
    async def start(self):
        """建立常驻 Live 连接。回调把音频/文字推给前端。"""
        self.live = LiveTTS(
            on_audio=self._on_answer_audio,
            on_text=self._on_answer_text,
            on_turn_complete=self._on_answer_done,
            on_interrupted=self._on_answer_interrupt,
        )
        await self.live.start()

    async def close(self):
        if self.live is not None:
            await self.live.close()
        # 补跑 voicemem 的 session-boundary 批处理（子图判定+右脑归因）——
        # 不然只有"下一个 session 的 Ingest() 检测到 session_id 变化"才会
        # 触发，这个连接如果是最后一个 session，永远等不到那个信号。
        # flush_all() 内部是真实同步 LLM 调用，跟文件里其它调用 cognitive
        # 的地方一样丢进线程池，不阻塞事件循环。
        try:
            await asyncio.to_thread(self.cognitive.flush_all)
        except Exception as e:
            print(f"[session] flush_all failed: {e}")

    # ---------- 回答音频/文字回调(Live -> 前端) ----------
    async def _on_answer_audio(self, pcm_bytes):
        await self.send_to_ui(bytes(pcm_bytes))   # 二进制帧 = 24kHz PCM
    async def _on_answer_text(self, text):
        await self.send_to_ui({"type": "answer", "ts": now_str(), "text": text})
    async def _on_answer_done(self):
        self._answering = False
        await self.send_to_ui({"type": "answer_done", "ts": now_str()})
    async def _on_answer_interrupt(self):
        await self.send_to_ui({"type": "answer_interrupt"})

    # ---------- 对外主入口 ----------
    async def on_audio(self, pcm_bytes):
        """接收一段 PCM(int16 bytes),切帧逐帧处理。唯一对外入口。"""
        pcm = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if not hasattr(self, "_buf"):
            self._buf = np.zeros(0, dtype=np.float32)
        self._buf = np.concatenate([self._buf, pcm])
        while len(self._buf) >= FRAME_SAMPLES:
            await self._on_frame(self._buf[:FRAME_SAMPLES], time.time())
            self._buf = self._buf[FRAME_SAMPLES:]

    # ---------- 每帧逻辑 ----------
    async def _on_frame(self, samples, t):
        is_voice = self.vad.is_voice(samples)
        partial = self.asr.feed(samples)

        # barge-in:回答进行中,只有"持续说话"超过阈值才打断,避免回答前摇期间
        # 一两帧噪声/回声把整段回复误打断丢掉(原生音频首字慢,这个窗口尤其大)。
        if self._answering:
            if is_voice:
                self._barge_voice += FRAME_SAMPLES / SAMPLE_RATE
            else:
                self._barge_voice = 0.0
            if self._barge_voice >= BARGE_MIN_VOICE:
                self._answering = False
                self._barge_voice = 0.0
                if self.live is not None:
                    self.live.cancel()
                await self.send_to_ui({"type": "answer_interrupt"})

        if is_voice:
            await self._on_voice(samples, partial, t)
        else:
            await self._on_silence(t)

    async def _on_voice(self, samples, partial, t):
        if self.turn is None:
            self.turn_counter += 1
            self.turn = Turn(str(self.turn_counter))
        # 一旦继续说话，取消本次静音的句末计时；避免分析线程慢时仍把一句话提前染蓝。
        endpoint_task = self.turn.endpoint_task
        if endpoint_task is not None and not endpoint_task.done():
            endpoint_task.cancel()
        self.turn.endpoint_task = None
        if self.turn.previewed and not self.turn.locked:
            self.turn.previewed = False
            self.turn.analysis = None
            self.turn.prefetched = False
            self.turn.ready_result = None
        self.last_voice_ts = t
        self.turn.add(samples)
        if partial and partial != self.turn.cur_text:
            self.turn.cur_text = partial
            await self.send_to_ui({"type": "partial", "ts": now_str(),
                                   "global_id": self.turn.global_id, "text": partial})

    async def _emit_ui_lock_after_silence(self, turn: Turn):
        """独立于分析/检索的句末计时器，保证 UI 不被慢模型阻塞。"""
        try:
            await asyncio.sleep(SILENCE_LOCK)
            if (self.turn is turn and not turn.locked and not turn.ui_locked
                    and time.time() - self.last_voice_ts >= SILENCE_LOCK):
                turn.ui_locked = True
                await self.send_to_ui({
                    "type": "turn_locked", "ts": now_str(),
                    "global_id": turn.global_id,
                })
        except asyncio.CancelledError:
            pass

    async def _emit_ui_lock(self, turn: Turn):
        if turn.ui_locked:
            return
        turn.ui_locked = True
        task = turn.endpoint_task
        if task is not None and not task.done():
            task.cancel()
        await self.send_to_ui({
            "type": "turn_locked", "ts": now_str(), "global_id": turn.global_id,
        })

    async def _on_silence(self, t):
        if self.turn is None or self.turn.locked:
            return
        if self.turn.endpoint_task is None and not self.turn.ui_locked:
            self.turn.endpoint_task = asyncio.create_task(
                self._emit_ui_lock_after_silence(self.turn))
        silence = t - self.last_voice_ts
        if not self.turn.previewed and silence >= SILENCE_PREVIEW:
            await self._on_preview()
            self.turn.previewed = True
        if (self.turn.previewed and not self.turn.prefetched
                and not self.turn.locked and silence >= SILENCE_PREFETCH):
            await self._on_prefetch()
            self.turn.prefetched = True
        if self.turn.previewed and not self.turn.locked and silence >= SILENCE_LOCK:
            # 此路径通常已由独立计时器先染蓝；仅作为计时器尚未来得及执行的兜底。
            await self._emit_ui_lock(self.turn)
            await self._on_lock()
            self.turn.locked = True
            self.asr.reset()
            self.turn = None

    # ---------- 分析 ----------
    def _analyze(self, audio):
        """出 (文本, 说话人str, 情感, 实体list, 簇list)。"""
        text, sensevoice_emotion = self.transcriber.run_with_emotion(pad_audio(audio))
        temo = text_emotion(text or "")
        if audio is None or len(audio) / SAMPLE_RATE < MIN_VOICE_SEC:
            return text, "-", (temo or "-"), [], []
        spk = "Speaker %d" % self.speaker.run(audio)
        acoustic_emotion = (
            self.emotion.run(audio) if self.emotion is not None else sensevoice_emotion
        )
        emo = fuse_emotion(acoustic_emotion, temo)
        entities, cats = self.router.route(text or "", top_k=3)
        slots = [c for c, _ in cats]
        return text, spk, emo, entities, slots

    # ---------- 200ms: prepare ----------
    async def _on_preview(self):
        audio = self.turn.audio()
        text, spk, emo, entities, slots = await asyncio.to_thread(self._analyze, audio)
        if not text:
            text = self.turn.cur_text
        self.turn.analysis = (text, spk, emo, entities, slots)

        self.turn.prepare_t = time.time()
        reply = await asyncio.to_thread(
            self.cognitive.prepare,
            global_id=self.turn.global_id, transcript=text,
            entities=entities, slots=slots, emotion=emo,
        )
        await self.send_to_ui({
            "type": "result", "locked": False, "ts": now_str(),
            "global_id": self.turn.global_id, "text": text,
            "speaker": spk, "emotion": emo,
            "entities": entities, "slots": slots,
            "status": reply.get("status"),
        })

    # ---------- 0.4s: 提前发 ready 预取 info(藏在静音等待里)----------
    async def _on_prefetch(self):
        if self.turn is None or self.turn.analysis is None:
            return
        text, spk, emo, entities, slots = self.turn.analysis
        result = await asyncio.to_thread(
            self.cognitive.ready,
            global_id=self.turn.global_id, transcript=text, speaker_id=spk,
            entities=entities, slots=slots, emotion=emo,
            audio=self.turn.audio(), sample_rate=SAMPLE_RATE,
        )
        if self.turn is not None:
            self.turn.ready_result = result

    # ---------- 500ms: ready(锁定) ----------
    async def _on_lock(self):
        audio = self.turn.audio()
        if self.turn.analysis is not None:
            text, spk, emo, entities, slots = self.turn.analysis
        else:
            text, spk, emo, entities, slots = await asyncio.to_thread(self._analyze, audio)
        if not text:
            text = self.turn.cur_text

        # 碎字:并入上一句,不新开 turn,不送 answer
        if char_count(text) < MIN_TURN_CHARS and self.has_last:
            self.last_text = (self.last_text + " " + text).strip()
            await self.send_to_ui({
                "type": "append", "ts": now_str(),
                "text": self.last_text, "speaker": self.last_spk,
                "emotion": self.last_emotion})
            return

        if self.turn.ready_result is not None:
            print("[lock] prefetch 命中:voicemem 不在关键路径上 ✅")
            reply = self.turn.ready_result
        else:
            print("[lock] prefetch 未命中:本轮在关键路径上等 voicemem ⏳(慢的来源)")
            reply = await asyncio.to_thread(
                self.cognitive.ready,
                global_id=self.turn.global_id, transcript=text, speaker_id=spk,
                entities=entities, slots=slots, emotion=emo,
                audio=audio, sample_rate=SAMPLE_RATE,
            )

        elapsed_ms = None
        if self.turn.prepare_t is not None:
            elapsed_ms = round((time.time() - self.turn.prepare_t) * 1000, 1)
        ok = reply.get("status") == "search_end_success"
        info = reply.get("info", "") if ok else ""
        info_detail = reply.get("info_detail", {}) if ok else {}
        audiomem_info = info_detail.get("audiomem_info") or {}
        print("[ready] status=", reply.get("status"), " prepare->memory 耗时:", elapsed_ms, "ms")

        ui_msg = {
            "type": "result", "locked": True, "ts": now_str(),
            "global_id": self.turn.global_id, "text": text,
            "speaker": spk, "emotion": emo,
            "entities": entities, "slots": slots,
            "info": info,
            "left_brain": info_detail.get("left_brain_info", ""),
            "right_brain": info_detail.get("right_brain_info", ""),
            "audiomem": audiomem_info,
            "elapsed_ms": elapsed_ms,
            "status": reply.get("status"),
        }
        await self.send_to_ui(ui_msg)

        # 原声回放("回放一下当时讨论...的原声")
        playback = audiomem_info.get("playback")
        if playback and playback.get("audio_path"):
            try:
                import base64
                wav_bytes = open(playback["audio_path"], "rb").read()
                await self.send_to_ui({
                    "type": "playback", "ts": now_str(),
                    "memory_text": playback.get("memory_text", ""),
                    "audio_b64": base64.b64encode(wav_bytes).decode("ascii"),
                })
                print(f"[playback] 回放原声: {playback['memory_id']}  {playback.get('memory_text', '')!r}")
            except Exception as e:
                print(f"[playback] 读取/发送原声失败: {e}")

        self.last_text, self.last_spk, self.last_emotion = text, spk, emo
        self.has_last = True

        await self._answer(text, emo, info,
                           info_detail.get("left_brain_info", ""),
                           info_detail.get("right_brain_info", ""),
                           audiomem_info)

    # 每轮按识别到的情绪,给 Live 一条"怎么说"的即时指令(比静态 system prompt 更管用)
    _DELIVERY = {
        "悲伤": "【必须】以一声哀叹开头(用用户的语言,中『唉……』/英 'Oh no…' / 'Aw…'),"
                "声音放得很慢很轻很柔,满满的心疼和同情,像叹着气安慰 ta。",
        "开心": "【必须】以一个兴奋的欢呼开头(中『哇!太棒啦!』/英 'Wow!' / 'Yay!' / "
                "'That's awesome!'),声音明显上扬、雀跃、带笑,语速快,比 ta 还激动地替 ta 高兴。",
        "愤怒": "【必须】以一句帮 ta 出气的话开头(中『太过分了吧!』/英 'That's so unfair!'),"
                "语气有力、站 ta 这边。",
        "惊讶": "【必须】以一声惊呼开头(中『啊?!真的假的!』/英 'Wait, really?!'),语调猛地上扬。",
        "恐惧": "声音放低、放稳、放慢,以安抚的话(中『别怕,有我在』/英 \"It's okay, I'm here\")"
                "给足安全感。",
        "厌恶": "【必须】以一句共鸣的吐槽开头(中『确实够膈应的!』/英 'Ugh, that's gross!'),"
                "站在 ta 这边。",
    }

    # ---------- audiomem 信号 -> 自然语言提示行 ----------
    @staticmethod
    def _audiomem_lines(audiomem_info):
        """从 audiomem_info 里挑出这一轮值得让回答提一句的信号,生成自然语言提示行。"""
        if not audiomem_info:
            return []
        lines = []
        scene = audiomem_info.get("current_scene")
        if scene:
            lines.append(f"ta 现在所在的环境:{scene}")
        tune = audiomem_info.get("recognized_tune") or {}
        if tune.get("action") == "match" and tune.get("heard_count", 0) >= 2:
            lines.append(f"ta 又哼起/放起了之前出现过的那段熟悉调子(已经第 {tune['heard_count']} 次了)")
        abnormal = audiomem_info.get("abnormal_sounds")
        if abnormal:
            lines.append(f"背景里出现了不寻常的声音({'/'.join(abnormal)}),如果相关,关心一下 ta 是否安全")
        place = audiomem_info.get("recognized_place") or {}
        place_prompt = audiomem_info.get("familiar_place_prompt") or {}
        if place.get("action") == "match" and place_prompt.get("memories"):
            mem_lines = "; ".join(m["content"] for m in place_prompt["memories"])
            lines.append(f"ta 回到了一个待过 {place.get('visit_count')} 次的熟悉地方,上次在这里聊到:{mem_lines}")
        routine = audiomem_info.get("new_routine")
        if routine:
            lines.append(f"刚发现 ta 有个生活规律:经常在 {routine.get('bucket_label')} 处于'{routine.get('scene')}'场景")
        for reminder in audiomem_info.get("triggered_reminders") or []:
            lines.append(f"ta 之前设置的提醒被触发了:{reminder.get('message')}")
        trig = audiomem_info.get("scene_trigger_created")
        if trig:
            lines.append(f"ta 刚设置了一条新提醒(到{trig.get('scene')}场景时触发):{trig.get('message')},确认一下已经记下了")
        if audiomem_info.get("playback"):
            lines.append(
                "ta 要求回放当时的原声,系统马上就会自动播放那段原始录音了——"
                "你只需要简短说一句『好,你听』这类话确认一下,不要复述或转述录音内容,"
                "也不要描述录音里具体说了什么。"
            )
        return lines

    # ---------- 最终回答 ----------
    @staticmethod
    def _answer_prompt(transcript, emotion, info, left_brain, right_brain, audiomem_info=None):
        cap = lambda s, n=180: (s[:n] + "…") if s and len(s) > n else (s or "")
        parts, mem = [], []
        if left_brain:
            mem.append("左脑(ta 经历的事):" + cap(left_brain))
        if right_brain:
            mem.append("右脑(ta 的情绪):" + cap(right_brain))
        if info and not (left_brain or right_brain):
            mem.append("背景:" + cap(info))
        mem.extend(VoiceAssistant._audiomem_lines(audiomem_info))

        if mem:
            parts.append("关于 ta 的记忆:")
            parts.extend(mem)
            parts.append("结合上面记得的【那件具体的事】回应,让 ta 觉得你懂 ta,别泛泛反问。")
        if emotion and emotion != "-":
            parts.append("情绪:" + emotion)
        parts.append("ta 刚说:" + transcript)
        parts.append("简短,最多 2-3 句。")
        delivery = VoiceAssistant._DELIVERY.get(emotion)
        if delivery:
            parts.append("怎么说:" + delivery + "情绪要外放、有起伏,别平淡。")
        if reply_lang(transcript) == "en":
            parts.append("CRITICAL — LANGUAGE: The user is speaking English. Reply ONLY in "
                         "natural spoken English. Do NOT use any Chinese, even one word.")
        else:
            parts.append("【语言】用户在说中文,只用中文回答,不要夹英文。")
        return "\n".join(parts)

    async def _answer(self, transcript, emotion, info, left_brain="", right_brain="",
                      audiomem_info=None):
        text = (transcript or "").strip()
        if not text or self.live is None:
            return
        self._answering = True
        self._barge_voice = 0.0
        await self.send_to_ui({"type": "answer_start", "ts": now_str()})
        await self.live.say(
            self._answer_prompt(text, emotion, info, left_brain, right_brain, audiomem_info))
