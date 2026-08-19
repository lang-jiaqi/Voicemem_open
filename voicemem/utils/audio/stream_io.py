"""流式输入的音频小工具：重采样 + silero VAD。

原样搬自 web/utils.py（脑图 html 发 24k、sherpa ASR 要 16k；silero VAD 判说完），
提升为核心能力，供 voicemem/stream.py 的 VoiceStream 与 web demo 共用（消除重复）。
"""
from __future__ import annotations

import os

import numpy as np


def resample(f32, src=24000, dst=16000):        # 脑图 html 发 24k，sherpa ASR 要 16k
    n = int(len(f32) * dst / src)
    return np.interp(np.arange(n) * src / dst, np.arange(len(f32)), f32).astype(np.float32)


def make_vad():
    import sherpa_onnx
    d = os.environ.get("VOICEMEM_MODELS_DIR", "../models")
    v = sherpa_onnx.VoiceActivityDetector(sherpa_onnx.VadModelConfig(
        silero_vad=sherpa_onnx.SileroVadModelConfig(model=f"{d}/silero_vad.onnx", threshold=0.5),
        sample_rate=16000), buffer_size_in_seconds=30)

    class _V:
        def is_speech(self, frame): v.accept_waveform(frame); return v.is_speech_detected()
    return _V()
