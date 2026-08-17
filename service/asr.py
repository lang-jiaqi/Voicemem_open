"""语音转文字：流式识别（实时 partial）+ 非流式精转写（最终文本）。

来自 demo/live/models.py，原样搬过来，没改逻辑。
"""
from __future__ import annotations

import re

import sherpa_onnx
from funasr import AutoModel

SAMPLE_RATE = 16000

SENSEVOICE_EMOTION_MAP = {
    "NEUTRAL": "中性",
    "HAPPY": "开心",
    "ANGRY": "愤怒",
    "SAD": "悲伤",
    "FEARFUL": "恐惧",
    "FEAR": "恐惧",
    "DISGUSTED": "厌恶",
    "SURPRISED": "惊讶",
}


def pick_device() -> str:
    """自动选最佳设备: cuda > mps(Apple M) > cpu。"""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class StreamingASR:
    """sherpa-onnx 流式 zipformer，出实时 partial 文本。"""

    def __init__(self, asr_dir: str) -> None:
        self.rec = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=f"{asr_dir}/tokens.txt",
            encoder=f"{asr_dir}/encoder-epoch-99-avg-1.onnx",
            decoder=f"{asr_dir}/decoder-epoch-99-avg-1.onnx",
            joiner=f"{asr_dir}/joiner-epoch-99-avg-1.onnx",
            num_threads=2, sample_rate=SAMPLE_RATE, feature_dim=80,
            decoding_method="greedy_search",
        )
        self.stream = self.rec.create_stream()

    def feed(self, samples):
        self.stream.accept_waveform(SAMPLE_RATE, samples)
        while self.rec.is_ready(self.stream):
            self.rec.decode_stream(self.stream)
        return self.rec.get_result(self.stream)

    def reset(self) -> None:
        self.stream = self.rec.create_stream()


class Transcriber:
    """SenseVoiceSmall 出最终文本（中英），比流式 ASR 更准，锁定一轮时用这个。"""

    def __init__(self, device: str) -> None:
        self.model = AutoModel(model="FunAudioLLM/SenseVoiceSmall", hub="hf",
                               device=device, disable_update=True,
                               trust_remote_code=False)

    def _generate(self, audio) -> str:
        res = self.model.generate(input=audio, cache={}, language="zh",
                                  use_itn=True, ban_emo_unk=True)
        if not res:
            return ""
        return res[0].get("text", "") or ""

    def run(self, audio) -> str:
        return re.sub(r"<\|[^|]*\|>", "", self._generate(audio)).strip()

    def run_with_emotion(self, audio) -> tuple[str, str]:
        """一次 SenseVoice 推理同时取得文本和声学情绪 token。"""
        raw = self._generate(audio)
        tags = re.findall(r"<\|([^|]+)\|>", raw.upper())
        emotion = next((SENSEVOICE_EMOTION_MAP[tag] for tag in tags
                        if tag in SENSEVOICE_EMOTION_MAP), "中性")
        return re.sub(r"<\|[^|]*\|>", "", raw).strip(), emotion
