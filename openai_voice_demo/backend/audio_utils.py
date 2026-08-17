"""Raw PCM16 mono -> WAV helper. Stdlib only (`wave`), no numpy — the browser
already sends signed 16-bit little-endian PCM frames per the wire protocol.
"""
from __future__ import annotations

import wave
from pathlib import Path

SAMPLE_RATE = 24000  # fixed: matches OpenAI Realtime's only supported PCM rate
SAMPLE_WIDTH = 2  # bytes, 16-bit
CHANNELS = 1


def write_wav_file(path: Path, pcm: bytes, sample_rate: int = SAMPLE_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
