"""一轮对话的状态容器 + 计时常量。谁来"判断这轮说完了没"（Silero VAD + 这些
阈值）在 session.py 里，这个文件只放数据结构和纯函数，没有回调依赖。
"""
from __future__ import annotations

import os
import re
import time

from asr import SAMPLE_RATE

FRAME_SAMPLES = SAMPLE_RATE * 32 // 1000   # 32ms 一帧
SILENCE_PREVIEW = 0.2      # 200ms -> prepare
SILENCE_PREFETCH = 0.25    # 尽早发 ready 预取 info,让 voicemem 往返藏进静音等待、
                           # 别落到回答关键路径上(不出声,零抢答风险)
# 停顿多久算说完 -> 锁定并回答。越小回复越快,但太小会把句中停顿当结束(抢答)。
SILENCE_LOCK = float(os.environ.get("SILENCE_LOCK", "0.5"))
BARGE_MIN_VOICE = float(os.environ.get("BARGE_MIN_VOICE", "0.16"))
# 默认等待 500ms 静音再确认句末，换来更快的响应，代价是误判抢答的风险比
# 900ms 时更高（未经真实麦克风环境验证）；记忆预取仍在 250ms 静音时启动，
# 但检索预取窗口只剩 300ms（500-200ms），小于实测检索算力(约450-650ms)，
# 所以大概率无法完全藏进静音等待里，锁定时可能还要再等一段。两者仍可通过
# 环境变量按实际环境调节。
MIN_VOICE_SEC = 1.5        # 短于此不做说话人/情感/路由
MIN_TURN_CHARS = 3         # 不足此字数视为碎字,并入上一句
PAD_SEC = 0.3              # 头尾补静音,缓解吃字


def now_str() -> str:
    return time.strftime("%H:%M:%S")


def char_count(text: str) -> int:
    """中文字数 + 英文单词数。"""
    cn = len(re.findall(r"[一-鿿]", text))
    en = len(re.findall(r"[a-zA-Z]+", text))
    return cn + en


def pad_audio(audio):
    import numpy as np
    pad = np.zeros(int(SAMPLE_RATE * PAD_SEC), dtype=np.float32)
    return np.concatenate([pad, audio, pad])


class Turn:
    """一个 turn 的累积状态。"""

    def __init__(self, global_id: str):
        self.global_id = global_id
        self.buf = []               # 音频帧
        self.cur_text = ""          # 实时 partial 文本
        self.start_ts = now_str()
        self.previewed = False
        self.locked = False
        self.ui_locked = False       # 句末已确认并通知 UI；不等同于记忆流程完成。
        self.endpoint_task = None    # 后台静音计时器（由 VoiceAssistant 管理）。
        self.prepare_t = None       # prepare 发出时刻(time.time()),用于测到memory耗时
        self.analysis = None        # 预览时算好的 (text,spk,emo,entities,slots),锁定时复用
        self.prefetched = False     # 是否已提前发过 ready
        self.ready_result = None    # 提前 ready 拿回的 dict,锁定时直接用

    def add(self, samples) -> None:
        self.buf.append(samples.copy())

    def audio(self):
        import numpy as np
        return np.concatenate(self.buf).astype(np.float32) if self.buf else \
            np.zeros(0, dtype=np.float32)

    def duration(self) -> float:
        return sum(len(b) for b in self.buf) / SAMPLE_RATE
