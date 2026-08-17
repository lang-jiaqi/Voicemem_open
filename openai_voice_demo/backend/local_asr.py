"""Local streaming ASR + VAD for pipeline mode's input side -- replaces the
OpenAI Realtime transcription session (cloud ASR) with the same sherpa-onnx
stack `service/` already uses (streaming zipformer bilingual zh-en + Silero
VAD, model files already in `service/models/`, fetched by
`scripts/download_models.sh`).

Why: measured with the cloud transcription session, "user stops speaking ->
caption appears" was ~1.2-1.5s at best (VAD silence wait + the API's own
finalization, the latter not tunable) and partial captions for continuous
speech only ever arrived AFTER the utterance ended (API delta-release
behavior, established by isolated A/B testing this session). The local
streaming recognizer decodes word-by-word WHILE the user is talking: real
live captions, near-zero finalization wait at end of turn, and barge-in
confirmation mid-utterance instead of after it. Honest trade-offs, not
glossed over: transcription accuracy is below gpt-4o-mini-transcribe
(greedy-search zipformer; English comes out UPPERCASE, no punctuation), and
ASR now costs local CPU instead of API calls. GPT-4o handles unpunctuated
transcripts fine; stored memory text inherits the rougher transcript.

All DSP here is synchronous and cheap (~1-3ms per 32ms chunk on this
machine); the caller runs it inline on the event loop.
"""
from __future__ import annotations

from pathlib import Path

SAMPLE_RATE_IN = 24000   # what the browser sends (see audio_utils.SAMPLE_RATE)
SAMPLE_RATE_ASR = 16000  # what the sherpa models were trained on

# How long a silence closes the turn. Same trade-off knob as the cloud
# session's silence_duration_ms had (500 chosen there too, after real user
# testing weighed caption latency against mid-sentence cutoffs).
TURN_SILENCE_S = 0.5

# How long a silence emits the early ("silence_hint", text) event -- the
# caller uses it to START generating a reply speculatively while the
# remaining 300ms of turn-close confirmation is still pending (see
# pipeline.py). Well under TURN_SILENCE_S on purpose: if the user resumes
# speaking, the caller cancels the speculative work, and nothing audible has
# happened yet by then (the reply pipeline's first audio is ~1.5s+ away).
EARLY_HINT_S = 0.2

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_ASR_DIR = _REPO_ROOT / "service" / "models" / "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"
_DEFAULT_VAD_MODEL = _REPO_ROOT / "service" / "models" / "silero_vad.onnx"

# The recognizer holds the model weights (~100MB, 1-3s load); share one per
# process across connections (each connection gets its own stream/VAD state).
_RECOGNIZER = None


def _get_recognizer():
    global _RECOGNIZER
    if _RECOGNIZER is None:
        import sherpa_onnx
        d = _DEFAULT_ASR_DIR
        if not d.is_dir():
            raise RuntimeError(
                f"local ASR model dir not found: {d} -- run "
                "`bash scripts/download_models.sh service/models` from the repo root"
            )
        _RECOGNIZER = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(d / "tokens.txt"),
            encoder=str(d / "encoder-epoch-99-avg-1.onnx"),
            decoder=str(d / "decoder-epoch-99-avg-1.onnx"),
            joiner=str(d / "joiner-epoch-99-avg-1.onnx"),
            num_threads=2, sample_rate=SAMPLE_RATE_ASR, feature_dim=80,
            decoding_method="greedy_search",
        )
    return _RECOGNIZER


class LocalASR:
    """Per-connection VAD + streaming-ASR state machine.

    Call ``process(pcm24k_bytes)`` for every binary frame from the browser;
    it returns a list of events, each one of:
      ("speech_start",)          -- VAD flipped to speech
      ("partial", full_text)     -- live transcript-so-far (FULL text, not a
                                    delta: greedy decoding can revise earlier
                                    words, so appending deltas would garble)
      ("silence_hint", text)     -- EARLY_HINT_S of silence: probably done
                                    talking, transcript-so-far attached; at
                                    most once per utterance, and the
                                    utterance may still continue after it
      ("turn_end", final_text)   -- TURN_SILENCE_S of silence closed the
                                    turn; final_text may be "" (noise-only)
    """

    def __init__(self, turn_silence_s: float = TURN_SILENCE_S) -> None:
        import numpy as np
        import sherpa_onnx
        self._np = np
        # 每个连接可覆盖断句静音窗口：realtime 模式传 0.2——它没有独立 TTS
        # 环节可以用投机生成掩盖等待，只能真的缩短窗口来换首音延迟；代价是
        # 句中停顿（喘气/思考）更容易被切成两轮，由两阶段打断机制兜底。
        # pipeline 模式保持默认 0.5s + 0.2s 投机（见 EARLY_HINT_S）。
        self._turn_silence_s = turn_silence_s
        self._rec = _get_recognizer()
        self._stream = self._rec.create_stream()
        if not Path(_DEFAULT_VAD_MODEL).is_file():
            raise RuntimeError(
                f"silero VAD model not found: {_DEFAULT_VAD_MODEL} -- run "
                "`bash scripts/download_models.sh service/models` from the repo root"
            )
        self._vad = sherpa_onnx.VoiceActivityDetector(
            sherpa_onnx.VadModelConfig(
                silero_vad=sherpa_onnx.SileroVadModelConfig(
                    model=str(_DEFAULT_VAD_MODEL),
                    # First version shipped threshold=0.3/min_speech=0.1
                    # (tuned for this user's then-very-quiet mic, 1-2% of
                    # full scale). Real follow-up feedback: with the user now
                    # speaking closer to the mic (measured 20-65% peaks),
                    # that tuning triggered on NON-speech noise constantly
                    # ("明明没有人声音也被打断"). Back to silero's default
                    # probability threshold, plus require a sustained 250ms
                    # of speech before flipping -- keyboard thumps and chair
                    # creaks are short, real utterances aren't. The pause
                    # this gates (speech_tentative) fires ~150ms later than
                    # before as the price.
                    threshold=0.4,
                    min_silence_duration=0.1, min_speech_duration=0.2),
                sample_rate=SAMPLE_RATE_ASR),
            buffer_size_in_seconds=30)
        # Streaming 24k->16k resampler state (linear interpolation, numpy --
        # NOT stdlib audioop, which was removed in Python 3.13; same
        # interpolation scheme the browser side already uses for its own
        # mic downsampling, see frontend/index.html's downsample()).
        # _resample_carry holds the tail samples needed to interpolate
        # across chunk boundaries; _resample_pos is the fractional read
        # position into carry+chunk.
        self._resample_carry = None
        self._resample_pos = 0.0
        self._in_speech = False
        self._silence_s = 0.0
        self._last_partial = ""         # only emit ("partial", ...) when the text changed
        self._hint_sent = False         # ("silence_hint", ...) at most once per utterance
        # The ONE recognizer stream lives for the whole connection; turns are
        # carved out of its ever-growing result text by this offset instead
        # of resetting the stream at turn boundaries. Real-audio testing
        # showed a freshly-created stream cold-starting on a quiet utterance
        # can decode it to NOTHING (the same audio decodes fine when the
        # stream has been running -- feature-normalization warm-up), so
        # per-turn stream resets were losing quiet turns entirely.
        self._consumed = 0

    def _to_16k_floats(self, pcm24k: bytes):
        np = self._np
        x = np.frombuffer(pcm24k, dtype=np.int16).astype(np.float32) / 32768.0
        if self._resample_carry is None:
            self._resample_carry = np.zeros(0, dtype=np.float32)
        buf = np.concatenate((self._resample_carry, x))
        ratio = SAMPLE_RATE_IN / SAMPLE_RATE_ASR  # 1.5
        if len(buf) < 2 or self._resample_pos > len(buf) - 1:
            self._resample_carry = buf
            return np.zeros(0, dtype=np.float32)
        n_out = int((len(buf) - 1 - self._resample_pos) / ratio) + 1
        idx = self._resample_pos + np.arange(n_out) * ratio
        i0 = idx.astype(np.int64)
        frac = (idx - i0).astype(np.float32)
        i1 = np.minimum(i0 + 1, len(buf) - 1)
        out = buf[i0] * (1 - frac) + buf[i1] * frac
        next_pos = self._resample_pos + n_out * ratio
        keep_from = min(int(next_pos), len(buf) - 1)
        self._resample_carry = buf[keep_from:]
        self._resample_pos = next_pos - keep_from
        return out

    def process(self, pcm24k: bytes) -> list[tuple]:
        samples = self._to_16k_floats(pcm24k)
        if len(samples) == 0:
            return []
        events: list[tuple] = []
        chunk_s = len(samples) / SAMPLE_RATE_ASR

        # The recognizer hears EVERYTHING, continuously -- VAD only decides
        # turn boundaries and the "someone started talking" signal, never
        # what audio the recognizer gets. The first version gated recognizer
        # input on the VAD (with an onset-replay buffer patching over the
        # VAD's trigger delay); real-audio testing showed quiet utterance
        # onsets still getting clipped into empty transcripts whenever the
        # VAD flipped late, and every noise-robustness retune of the VAD
        # made that clipping worse -- the two tunings were fighting each
        # other by construction. Decoupling them ends that: VAD thresholds
        # can now be tuned purely for trigger robustness with zero effect on
        # transcription completeness. Cost: the zipformer also decodes
        # silence, which is cheap (measured ~12x faster than real-time on
        # one core here) and produces no output.
        self._stream.accept_waveform(SAMPLE_RATE_ASR, samples)
        while self._rec.is_ready(self._stream):
            self._rec.decode_stream(self._stream)
        turn_text = self._rec.get_result(self._stream)[self._consumed:]
        if turn_text.strip() and turn_text != self._last_partial:
            self._last_partial = turn_text
            events.append(("partial", turn_text.strip()))

        self._vad.accept_waveform(samples)
        is_voice = self._vad.is_speech_detected()

        if not self._in_speech:
            if is_voice:
                self._in_speech = True
                self._silence_s = 0.0
                self._hint_sent = False
                events.append(("speech_start",))
        else:
            if is_voice:
                self._silence_s = 0.0
                self._hint_sent = False  # utterance continues; a new pause gets a new hint
            else:
                self._silence_s += chunk_s
                full = self._rec.get_result(self._stream)
                # 窗口本身已 ≤ 提示阈值时（realtime 的 0.2s），hint 与
                # turn_end 同时到，没有提前量可言，直接不发
                if (not self._hint_sent and self._silence_s >= EARLY_HINT_S
                        and EARLY_HINT_S < self._turn_silence_s):
                    self._hint_sent = True
                    events.append(("silence_hint", full[self._consumed:].strip()))
                if self._silence_s >= self._turn_silence_s:
                    events.append(("turn_end", full[self._consumed:].strip()))
                    self._consumed = len(full)
                    self._last_partial = ""
                    self._in_speech = False
                    self._silence_s = 0.0
                    self._hint_sent = False
        return events
