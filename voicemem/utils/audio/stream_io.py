"""流式输入的音频小工具：重采样 + silero VAD。

原样搬自 web/utils.py（脑图 html 发 24k、流式 ASR 要 16k；silero VAD 判说完），
提升为核心能力，供 voicemem/stream.py 的 VoiceStream 与 web demo 共用（消除重复）。

VAD 现在是可注入能力（``VoiceMem(vad=...)`` / config 的 ``vad`` 段），``make_vad`` 只是
内置的那个 silero 实现；换成自己的只要给个有 ``is_speech(frame) -> bool`` 的对象。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from voicemem.utils.common.paths import model_path, require


def resample(f32, src=24000, dst=16000):        # 脑图 html 发 24k，流式 ASR 要 16k
    n = int(len(f32) * dst / src)
    return np.interp(np.arange(n) * src / dst, np.arange(len(f32)), f32).astype(np.float32)


def make_vad(model: str | None = None, threshold: float = 0.5):
    """内置 VAD：silero（sherpa-onnx 包的）。返回一个只有 ``is_speech(frame)`` 的小对象。

    ``model`` 不给就走 ``VOICEMEM_SILERO_VAD`` / ``VOICEMEM_MODELS_DIR/silero_vad.onnx``。
    这个 .onnx 没有自动下载兜底，缺了就明确报出来（而不是让 sherpa 抛个看不懂的错）。
    """
    import sherpa_onnx
    path = require(
        Path(model) if model else model_path("silero_vad.onnx", "VOICEMEM_SILERO_VAD"),
        "silero VAD 模型 silero_vad.onnx",
    )
    v = sherpa_onnx.VoiceActivityDetector(sherpa_onnx.VadModelConfig(
        silero_vad=sherpa_onnx.SileroVadModelConfig(model=str(path), threshold=threshold),
        sample_rate=16000), buffer_size_in_seconds=30)

    class _V:
        def is_speech(self, frame): v.accept_waveform(frame); return v.is_speech_detected()
    return _V()
