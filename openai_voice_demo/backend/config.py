"""Env -> Config. Loaded once at process start, before `voicemem` is imported
anywhere, so OPENAI_API_KEY/OPENAI_MODEL/OPENAI_EMBEDDING_MODEL/OPENAI_BASE_URL
land in os.environ in time for voicemem's own internal client to pick them up.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_DEMO_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        # Strip a trailing "  # comment" -- found for real: VOICE_MODE=realtime
        # followed by an inline comment on the same line silently became part
        # of the value (config.py had no inline-comment handling at all), and
        # load_config()'s exact-match check on voice_mode then rejected it.
        if " #" in value:
            value = value.split(" #", 1)[0].strip()
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(_DEMO_ROOT / ".env")


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


# Live-test gap #1: replies were flat/monotone -- nothing ever told the model
# its VOICE delivery (tone, pace, warmth) should match the emotional content
# of what it just retrieved. Only audible in realtime mode (native
# speech-to-speech); pipeline mode's separate TTS call doesn't carry emotion
# cues through from the chat text.
# Live-test gap #2: with only "use the MEMORY CONTEXT" as guidance the model
# RECITED memories like a dossier ("你是研二的同学,导师是老周") instead of
# talking like someone who knows you. The fix is style rules about HOW to
# use memories, not more memories.
DEFAULT_SYSTEM_PROMPT = (
    "你是一个有长期记忆的语音伙伴,像认识用户很久、真正懂他的老朋友。\n"
    "说话方式:\n"
    "- 口语化,简短。一般一两句话,别长篇大论,别分点罗列。\n"
    "- 先接住情绪,再给信息:对方开心你就跟着开心,低落就先安抚再说事。\n"
    "- 记忆里的事要像老朋友随口聊起(\"欸,你那个混双比赛练得怎么样了?\"),"
    "绝不要像念档案(\"你是研二的学生,导师是老周\")。\n"
    "- 被问\"你知道我是谁吗\"这类问题,热情自然地接(\"当然啦,我怎么会不知道\"),"
    "顺手带上名字和一两个最有温度的细节就好,不用全背出来。\n"
    "- 不太确定的记忆宁可不提,别硬套;永远不要提\"记忆库\"\"MEMORY\"这些字眼。\n"
    "- 严格区分时间范围:用户问\"下周\"时,只能回答明确属于下周的安排;\"这周\"、\"最近\"或没有日期的旧计划不能当作下周计划。没有明确匹配的下周安排就坦诚说暂时没有确认记录,不要自行推断。\n"
    "- 声音的语气随内容走:温暖的事说得温暖,兴奋的事说得兴奋,沉重的事放轻放柔,"
    "不要平铺直叙地念字。"
)


@dataclass
class Config:
    voice_mode: str  # "pipeline" | "realtime" | "qwen" | "doubao"
    audio_native: bool
    user_id: str
    host: str
    port: int
    openai_chat_model: str
    openai_tts_model: str
    openai_tts_voice: str
    openai_realtime_model: str
    qwen_realtime_model: str
    qwen_voice: str
    system_prompt: str
    memory_root: Path
    raw_audio_dir: Path


def load_config() -> Config:
    voice_mode = os.environ.get("VOICE_MODE", "pipeline").strip().lower()
    if voice_mode not in ("pipeline", "realtime", "qwen", "doubao"):
        raise ValueError(
            f"VOICE_MODE must be one of 'pipeline', 'realtime', 'qwen', 'doubao', got {voice_mode!r}")

    return Config(
        voice_mode=voice_mode,
        audio_native=_env_bool("VOICEMEM_AUDIO_NATIVE", False),
        user_id=os.environ.get("VOICE_DEMO_USER_ID", "voice_demo_user"),
        host=os.environ.get("VOICE_DEMO_HOST", "0.0.0.0"),
        port=int(os.environ.get("VOICE_DEMO_PORT", "8787")),
        openai_chat_model=os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o"),
        openai_tts_model=os.environ.get("OPENAI_TTS_MODEL", "gpt-4o-mini-tts"),
        openai_tts_voice=os.environ.get("OPENAI_TTS_VOICE", "alloy"),
        # No STT/transcription model knobs: speech-to-text is fully local in
        # both modes now (backend/local_asr.py), no cloud ASR involved.
        openai_realtime_model=os.environ.get("OPENAI_REALTIME_MODEL", "gpt-realtime"),
        # DashScope Omni-Realtime (VOICE_MODE=qwen). flash is the cheap GA
        # tier; qwen3.5-omni-plus/flash-realtime exist as newer snapshots.
        # "Cherry" is the stock Chinese female voice.
        qwen_realtime_model=os.environ.get("QWEN_REALTIME_MODEL", "qwen3-omni-flash-realtime"),
        qwen_voice=os.environ.get("QWEN_VOICE", "Cherry"),
        system_prompt=os.environ.get("VOICE_DEMO_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT),
        memory_root=_DEMO_ROOT / "data" / "voicemem_memory",
        raw_audio_dir=_DEMO_ROOT / "data" / "raw_audio",
    )
