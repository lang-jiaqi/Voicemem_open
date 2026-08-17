"""实时"快速版"识别：给界面先看个大概，最终结论由 backend/voicemem 核心的
CAM++ 声纹（speaker_encoder.py）/ Qwen-Omni 归因算，这两套不冲突。

来自 demo/live/models.py，去掉了 EcapaEncoder ——那是旧的跨 session 声纹匹配
方案，已经被 voicemem 核心的 CAM++ 覆盖（Ingest() 返回的 speaker_id 就是
CAM++ 认出来的 person_id），不用在这层重复维护一套。
"""
from __future__ import annotations

import os

import numpy as np
import sherpa_onnx
from funasr import AutoModel

from asr import SAMPLE_RATE


class EmotionRecognizer:
    """emotion2vec+ base 出 9 类情感(中文)。

    emotion2vec+ 对中性/平常语音有明显的 sad 偏置,直接 argmax 会把很多正常说话
    判成"悲伤"。这里做两道闸:最高分要够自信(MIN_CONF),且要明显高过中性分
    (MARGIN),否则一律归"中性"。两个阈值可用环境变量现场调:
        EMO_MIN_CONF (默认 0.55)  最高分至少要达到
        EMO_MARGIN   (默认 0.20)  且要比中性分高出这么多
        EMO_SAD_EXTRA(默认 0.10)  sad 偏置大,额外再抬高它的门槛
        EMO_FEAR_EXTRA(默认 0.20) fearful 对平静短句误报更多,额外收紧
    """

    MAP = {"angry": "愤怒", "happy": "开心", "neutral": "中性", "sad": "悲伤",
           "surprised": "惊讶", "fearful": "恐惧", "disgusted": "厌恶",
           "other": "其他", "unknown": "未知"}

    MIN_CONF = float(os.environ.get("EMO_MIN_CONF", "0.55"))
    MARGIN = float(os.environ.get("EMO_MARGIN", "0.20"))
    SAD_EXTRA = float(os.environ.get("EMO_SAD_EXTRA", "0.10"))
    FEAR_EXTRA = float(os.environ.get("EMO_FEAR_EXTRA", "0.20"))
    CAUTIOUS_EXTRA = float(os.environ.get("EMO_CAUTIOUS_EXTRA", "0.10"))

    def __init__(self, device: str) -> None:
        self.model = AutoModel(model="emotion2vec/emotion2vec_plus_base", hub="hf",
                               device=device, disable_update=True)

    def run(self, audio) -> str:
        res = self.model.generate(input=audio, granularity="utterance",
                                  extract_embedding=False)
        if not res:
            return "未知"
        labels, scores = res[0]["labels"], res[0]["scores"]
        sd = {}
        for lab, sc in zip(labels, scores):
            sd[lab.split("/")[-1].strip().lower()] = float(sc)
        if os.environ.get("EMO_DEBUG") == "1":
            top3 = sorted(sd.items(), key=lambda x: -x[1])[:3]
            print("[emo] " + "  ".join(f"{k}={v:.2f}" for k, v in top3))
        i = int(np.argmax(scores))
        top_key = labels[i].split("/")[-1].strip().lower()
        top_score = float(scores[i])
        neutral_score = sd.get("neutral", 0.0)

        if top_key in ("neutral", "unknown"):
            return self.MAP.get(top_key, "中性")

        # 声学 fearful/disgusted/surprised 容易把平静的短句听错；除非有很强
        # 的声学证据，否则交给文本情绪词（session.py）或回落为中性。
        extra = 0.0
        if top_key == "sad":
            extra = self.SAD_EXTRA
        elif top_key == "fearful":
            extra = self.FEAR_EXTRA
        elif top_key in ("disgusted", "surprised"):
            extra = self.CAUTIOUS_EXTRA
        if (top_score < self.MIN_CONF + extra
                or (top_score - neutral_score) < self.MARGIN + extra):
            return "中性"
        return self.MAP.get(top_key, labels[i])


class SpeakerId:
    """sherpa-onnx 声纹，返回本次通话里第几个说话人（从0开始）。

    只做"这通话里是不是同一个人在说话"的会话内区分（轻量、实时），不负责
    "认出这是跨 session 见过的谁"——那是 voicemem 核心 CAM++ 的事。
    """

    def __init__(self, model_path: str, threshold: float = 0.35) -> None:
        self.ext = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=model_path, num_threads=2))
        self.threshold = threshold
        self.embs = []

    def run(self, audio) -> int:
        s = self.ext.create_stream()
        s.accept_waveform(SAMPLE_RATE, audio)
        s.input_finished()
        if not self.ext.is_ready(s):
            return 0
        emb = np.array(self.ext.compute(s), dtype=np.float32)
        if not self.embs:
            self.embs.append(emb)
            return 0
        sims = [float(emb @ e / (np.linalg.norm(emb) * np.linalg.norm(e) + 1e-9))
                for e in self.embs]
        best = int(np.argmax(sims))
        if sims[best] >= self.threshold:
            return best
        self.embs.append(emb)
        return len(self.embs) - 1


class Vad:
    """sherpa-onnx 内置 Silero VAD，判断当前帧是否有人声（"有没有人在说话"，
    跟 voicemem 核心里算情绪正负/激动程度的 VAD 是同名不同义,不要混）。
    """

    def __init__(self, model_path: str) -> None:
        self.vad = sherpa_onnx.VoiceActivityDetector(
            sherpa_onnx.VadModelConfig(
                silero_vad=sherpa_onnx.SileroVadModelConfig(
                    model=model_path, threshold=0.5,
                    min_silence_duration=0.1, min_speech_duration=0.1),
                sample_rate=SAMPLE_RATE),
            buffer_size_in_seconds=30)

    def is_voice(self, samples) -> bool:
        self.vad.accept_waveform(samples)
        return self.vad.is_speech_detected()
