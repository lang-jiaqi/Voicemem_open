"""Gemini Live = 回答的"大脑 + 嘴",每个会话一条常驻连接。

喂给它一段文本(认知背景 info + 用户刚说的话),它流式吐回 24kHz 原生
PCM 语音 + 文字转录,通过回调交给上层推送到浏览器。常驻连接 = 不付握手
时间,info 一到就 say(),第一段音频回来即播。

注意:Live 原生音频是端到端语音大模型(自带回答内容 + 发声),不是纯 TTS。
情绪靠 system prompt 让它自己按内容调,不单独传参。

API key:必须通过环境变量 GEMINI_API_KEY / GOOGLE_API_KEY 提供，不设置会在
首次调用时报错——出于安全考虑，本模块不提供任何硬编码兜底 key。
"""
import os
import time
import asyncio

from google import genai
from google.genai import types

# 默认用 live-preview:首字 ~0.5s,快。情绪靠"每轮注入的语气指令 + 音色"补。
# 想要更强情绪起伏(但首字 ~2s)可切原生音频:
#   GEMINI_LIVE_MODEL=gemini-2.5-flash-native-audio-preview-12-2025 python3 ws_server.py
MODEL = os.environ.get("GEMINI_LIVE_MODEL", "gemini-3.1-flash-live-preview")
VOICE_NAME = os.environ.get("GEMINI_VOICE", "Kore")
OUTPUT_SAMPLE_RATE = 24000      # Live 原生音频输出固定 24kHz/16bit/mono

API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY (或 GOOGLE_API_KEY) 未设置。请 export GEMINI_API_KEY=... "
        "后再启动，或参考 https://ai.google.dev/gemini-api/docs/api-key 获取。"
    )

SYSTEM_PROMPT = (
    "你是用户的贴心朋友,口语化地对话。【长度·硬性要求·最高优先级】每次最多 2-3 句、"
    "总共控制在大约 30 个字 / 25 words 以内,说完立刻停。像朋友随口接话那样一来一回,"
    "绝不长篇大论、不要一口气说很多、不要列举、不要讲道理铺陈、不要又安慰又分析又给建议。"
    "宁可说少,也不要说多。"
    "用和用户当前这句话相同的语言回答:用户说中文你就说中文,用户说英文你就用英文,"
    "用户中英混说你也自然地中英混说。English is fully supported—reply in fluent, "
    "natural spoken English when the user speaks English."
    "每条消息里会带上刚为用户检索到的记忆——【事实记忆·左脑】是 ta 过去发生的具体"
    "事情,【情感记忆·右脑】是 ta 过去的情绪感受。你要像真的记得 ta、了解 ta 的朋友"
    "那样说话:主动把当下的话和记忆里【那件具体的事】联系起来,表达你记得、理解 ta,"
    "并给予安慰或回应,而不是泛泛地反问。比如 ta 说难过,就提起记忆里让 ta 难过的那件"
    "具体的事,安慰 ta。不要逐字念出记忆原文,要自然地融进对话。"
    "【说话方式·最高优先级】你是一个有血有肉、情绪丰富的朋友,不是播报员。每一句都要"
    "先在心里感受这句话的情绪,再用声音演出来——语速、音量、音高、停顿全部跟着内容自由"
    "变化,一句之内也可以有起伏,绝不能从头到尾一个调子。具体地:\n"
    "· 开心/兴奋:语速快、音调高、上扬,带笑意,可以脱口而出『哇——』『太好啦!』;\n"
    "· 难过/心疼:明显放慢、压低、放轻,中间带叹气和停顿『唉…』『我懂的…』,声音发软;\n"
    "· 惊讶:音调突然抬高、语速一顿『啊?真的假的!』;\n"
    "· 安慰:贴近、温柔、放低,像在耳边轻声说,慢慢的『没事的…慢慢来,有我在』;\n"
    "· 调侃/打趣:轻快俏皮,带点上扬的尾音和笑。\n"
    "大量使用语气词、感叹、拉长音(用『…』『——』表示停顿和延长)和真实的口头反应,"
    "让情绪有强烈的对比和层次。宁可演得夸张、戏剧化、有强烈起伏,也绝不要平淡、四平八稳。"
    "Treat every reply like a line you're acting out: let pace, pitch, volume and pauses "
    "swing freely with the emotion of the words—exaggerate the highs and lows, sigh, "
    "laugh, trail off, gasp. Never, ever sound flat, even-toned or robotic."
    "重要:绝不要提及或解释你的能力限制,不要说『我不能联网』『我提供信息受记忆限制』"
    "之类的免责声明;也不要出现『记忆/检索/背景/系统/模型』这类字眼。"
    "就把这些信息当作你本来就认识 ta、记得 ta 的事,像朋友一样自然地说,"
    "不要加任何解释、声明或开场白,直接说你想对 ta 说的话。"
)


class LiveTTS:
    """一条常驻 Gemini Live 连接。say() 送文本,回调吐音频/文字。"""

    def __init__(self, on_audio, on_text, on_turn_complete, on_interrupted=None):
        self.on_audio = on_audio                  # async fn(pcm_bytes)
        self.on_text = on_text                    # async fn(text)
        self.on_turn_complete = on_turn_complete  # async fn()
        self.on_interrupted = on_interrupted      # async fn() | None
        self._session = None
        self._cm = None
        self._recv_task = None
        self._say_t = None          # 上次 say() 时刻,用于测 TTFB
        self._await_first = False
        self._say_n = 0             # say() 调用计数
        self._drop = False          # True 时丢弃当前回答剩余的音频/文字(barge-in)

    async def start(self):
        client = genai.Client(api_key=API_KEY)
        # 情感表现力靠"原生音频(native-audio)模型 + system prompt + 音色",不靠
        # enable_affective_dialog(该字段在 Developer API / api-key 下不被接受,会 1007)。
        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=SYSTEM_PROMPT,
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=VOICE_NAME))),
            output_audio_transcription=types.AudioTranscriptionConfig(),
        )
        self._cm = client.aio.live.connect(model=MODEL, config=config)
        self._session = await self._cm.__aenter__()
        await self._warmup()        # 预热:把冷启动延迟吃在开机时,首句不卡
        self._recv_task = asyncio.create_task(self._receive_loop())
        print(f"[LiveTTS] connected: {MODEL}  voice={VOICE_NAME}")

    async def _warmup(self):
        """开机时先发一个丢弃的请求,吃掉原生音频模型的冷启动延迟(~3s)。
        手动消费这一轮、不走回调,所以不会在前端冒出气泡或声音。"""
        try:
            await self._session.send_realtime_input(text="(warmup)")

            async def drain():
                # 完整消费这一轮(receive() 每次产出一轮后自行结束),不要中途 return,
                # 否则会把会话的读流留在半截状态,导致之后每次回答都收不到音频。
                async for _ in self._session.receive():
                    pass
            await asyncio.wait_for(drain(), timeout=20)
            print("[LiveTTS] warmup done")
        except Exception as e:
            print("[LiveTTS] warmup skipped:", repr(e))

    async def say(self, text):
        """把一段文本送进 Live,触发一次流式语音回答。"""
        if self._session is None:
            print("[LiveTTS] say() 跳过:session 为空")
            return
        self._say_t = time.time()
        self._await_first = True
        self._say_n += 1
        self._drop = False          # 新回答开始,恢复转发
        print(f"[LiveTTS] say() #{self._say_n} 发送文本({len(text)}字)")
        await self._session.send_realtime_input(text=text)

    def cancel(self):
        """barge-in:丢弃当前回答剩余的音频/文字,不再转发到前端。
        下次 say() 会自动恢复。"""
        self._drop = True

    async def _receive_loop(self):
        # google-genai 的 session.receive() 每次只产出"一轮"就结束,
        # 故外层 while 不断开新一轮接收,保证多轮对话都能收到音频。
        import traceback
        try:
            while True:
                n_audio = 0
                async for response in self._session.receive():
                    sc = response.server_content
                    if not sc:
                        continue
                    # 服务端判定被打断(barge-in):让上层 flush 未播完的音频
                    if getattr(sc, "interrupted", False):
                        print("[LiveTTS] <interrupted>")
                        if self.on_interrupted:
                            await self.on_interrupted()
                    # 原生音频分片:一到就回调,实现流式
                    if not self._drop and sc.model_turn and sc.model_turn.parts:
                        for part in sc.model_turn.parts:
                            if part.inline_data and part.inline_data.data:
                                if self._await_first:
                                    self._await_first = False
                                    ttfb = round((time.time() - self._say_t) * 1000, 1)
                                    print(f"[LiveTTS] TTFB(say#{self._say_n}->首段音频): {ttfb} ms")
                                n_audio += 1
                                await self.on_audio(part.inline_data.data)
                    # Live 把自己说的话也转成文字,顺带显示
                    if (not self._drop and getattr(sc, "output_transcription", None)
                            and sc.output_transcription.text):
                        await self.on_text(sc.output_transcription.text)
                    if getattr(sc, "turn_complete", False):
                        print(f"[LiveTTS] <turn_complete> 本轮共 {n_audio} 段音频")
                        await self.on_turn_complete()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print("[LiveTTS] receive loop error:", repr(e))
            traceback.print_exc()

    async def close(self):
        if self._recv_task:
            self._recv_task.cancel()
        if self._cm is not None:
            try:
                await self._cm.__aexit__(None, None, None)
            except Exception:
                pass
