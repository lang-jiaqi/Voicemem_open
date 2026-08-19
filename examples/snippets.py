"""三个 voicemem 调用示例：① 文字 ② 语音 ③ VAD + 说完 300ms 返回全部 info。

都用新的顶层 API：from voicemem import VoiceMem。
"""
import numpy as np
from voicemem import VoiceMem

API_KEY = "sk-..."          # 换成你的 OpenAI key


# ── ① 文字：直接存 / 搜 ────────────────────────────────────────────────────────
def demo_text():
    vm = VoiceMem(api_key=API_KEY, mode="text_mode")
    vm.ingest("中午和 Alex 在公司附近吃了拉面")          # 存
    for hit in vm.left_brain.search("我中午吃了什么"):    # 搜（左脑事实）
        print(hit.text, round(hit.score, 3))


# ── ② 语音：ASR 转写 → 带音频存(跑场景/声纹/情绪感知) → 搜 ──────────────────────
def demo_voice(wav_path="turn.wav"):
    vm = VoiceMem(api_key=API_KEY, mode="multi_modal")
    samples = _read_wav_16k(wav_path)                     # 16kHz float 采样
    asr = vm.utils.get("asr")
    asr.reset()
    text = asr.feed(samples)                              # 语音 → 文字
    vm.ingest(text, audio=wav_path)                       # 带音频存（跑感知层）
    print(vm.search("刚才聊了什么").hits)                 # 搜


# ── ③ VAD + 说完 300ms → 返回全部 info ────────────────────────────────────────
def demo_turn(mic_frames):
    """mic_frames: 生成器，每次给一小段 16kHz float 采样（如 20ms）。
    检测到说完、静音满 300ms，就转写 + 感知 + 检索 + 存，返回一个 info dict。"""
    from voicemem.utils.audio.asr import SAMPLE_RATE
    import sherpa_onnx
    vm = VoiceMem(api_key=API_KEY, mode="multi_modal")
    vad = _silero_vad("models/silero_vad.onnx")
    asr = vm.utils.get("asr"); asr.reset()

    text, silence, spoke = "", 0.0, False
    for frame in mic_frames:
        text = asr.feed(frame)                            # 实时累积转写
        if vad.is_speech(frame): spoke, silence = True, 0.0
        else: silence += len(frame) / SAMPLE_RATE
        if spoke and silence >= 0.30:                     # 说完 300ms
            p = vm.preprocess(text)                       # 场景/说话人/情绪
            hits = vm.search(text).hits                   # 检索
            vm.ingest(text)                               # 存
            return {"text": text, "scene": p.scene_tag or "", "speaker": p.person_id or "",
                    "emotion": p.emotion or "", "hits": [h.text for h in hits]}
    return None


# ── 小工具 ─────────────────────────────────────────────────────────────────────
def _read_wav_16k(path):
    import wave
    with wave.open(path, "rb") as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def _silero_vad(model_path):
    import sherpa_onnx
    class _V:
        def __init__(s):
            s.v = sherpa_onnx.VoiceActivityDetector(sherpa_onnx.VadModelConfig(
                silero_vad=sherpa_onnx.SileroVadModelConfig(model=model_path, threshold=0.5),
                sample_rate=16000), buffer_size_in_seconds=30)
        def is_speech(s, frame):
            s.v.accept_waveform(frame); return s.v.is_speech_detected()
    return _V()


if __name__ == "__main__":
    demo_text()
