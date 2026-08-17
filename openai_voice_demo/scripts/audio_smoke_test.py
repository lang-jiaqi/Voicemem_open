"""Scripted WebSocket client that feeds REAL AUDIO into the backend --
no browser, no microphone, no frontend involved at all. This is the audio
counterpart to smoke_test.py (which only proves the text path): it proves
the backend handles voice on its own, independent of whatever sends the
audio in. The frontend (frontend/index.html) is just a thin UI that captures
mic PCM and streams it over the exact same WebSocket wire format this
script uses directly -- the backend doesn't know or care whether the PCM
bytes came from a browser tab or a script like this one.

Works against either VOICE_MODE (pipeline or realtime) -- both providers
accept binary PCM16/24kHz/mono frames on the same /ws route and use
OpenAI's own server-side VAD to detect turn boundaries, so this script
doesn't send any explicit turn_start/turn_end marker, just raw audio
followed by a bit of real silence (see audio_utils.SAMPLE_RATE / frontend/
index.html's SAMPLE_RATE -- 24kHz is not optional, it's what the Realtime
API's PCM sessions require).

Usage:
    # synthesizes its own test speech via OpenAI TTS if no file is given
    python scripts/audio_smoke_test.py
    # or feed a real recording (any format ffmpeg/soundfile can read is
    # NOT assumed here -- must already be PCM16 mono; use a .wav at any
    # sample rate, it gets resampled to 24kHz)
    python scripts/audio_smoke_test.py path/to/recording.wav [ws://localhost:8787/ws]
"""
from __future__ import annotations

import asyncio
import json
import sys
import wave

import numpy as np
import websockets
from openai import OpenAI

DEFAULT_URL = "ws://localhost:8787/ws"
TARGET_RATE = 24000  # fixed -- see audio_utils.py / frontend's SAMPLE_RATE


def _resample_pcm16(pcm: bytes, from_rate: int, to_rate: int) -> bytes:
    """Whole-file linear-interpolation resample (numpy, not stdlib audioop --
    audioop was removed in Python 3.13)."""
    if from_rate == to_rate:
        return pcm
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
    n_out = int(len(x) * to_rate / from_rate)
    idx = np.arange(n_out) * (from_rate / to_rate)
    i0 = np.minimum(idx.astype(np.int64), len(x) - 1)
    i1 = np.minimum(i0 + 1, len(x) - 1)
    frac = idx - i0
    return (x[i0] * (1 - frac) + x[i1] * frac).astype(np.int16).tobytes()


def synth_test_audio(text: str) -> bytes:
    """No input file given -- synthesize one via OpenAI TTS instead of
    requiring a real microphone recording on hand."""
    client = OpenAI()
    resp = client.audio.speech.create(
        model="tts-1", voice="alloy", input=text, response_format="pcm",
    )
    return resp.read()  # already 24kHz PCM16 mono, OpenAI TTS's own fixed rate


def load_wav_as_24k_pcm(path: str) -> bytes:
    with wave.open(path, "rb") as wf:
        assert wf.getsampwidth() == 2, "expected 16-bit PCM WAV"
        pcm = wf.readframes(wf.getnframes())
        rate = wf.getframerate()
        channels = wf.getnchannels()
    if channels == 2:
        stereo = np.frombuffer(pcm, dtype=np.int16).reshape(-1, 2)
        pcm = stereo.mean(axis=1).astype(np.int16).tobytes()
    return _resample_pcm16(pcm, rate, TARGET_RATE)


async def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("ws://")]
    urls = [a for a in sys.argv[1:] if a.startswith("ws://")]
    url = urls[0] if urls else DEFAULT_URL

    if args:
        print(f"loading {args[0]} ...")
        pcm = load_wav_as_24k_pcm(args[0])
        spoken_text_hint = None  # unknown -- real recording
    else:
        text = "My favorite food is ramen."
        print(f"no audio file given -- synthesizing TTS for: {text!r}")
        pcm = synth_test_audio(text)
        spoken_text_hint = text
    print(f"got {len(pcm)} bytes of {TARGET_RATE}Hz PCM16 mono audio "
          f"(~{len(pcm) / (TARGET_RATE * 2):.2f}s)")

    print(f"connecting to {url} ...")
    async with websockets.connect(url, max_size=None) as ws:
        audio_bytes_total = 0
        transcript_seen = ""
        events_seen: list[str] = []

        async def receiver():
            nonlocal audio_bytes_total, transcript_seen
            async for msg in ws:
                if isinstance(msg, (bytes, bytearray)):
                    audio_bytes_total += len(msg)
                    continue
                payload = json.loads(msg)
                events_seen.append(payload.get("type"))
                print("  <-", payload.get("type"),
                      {k: v for k, v in payload.items() if k not in ("type", "left_brain")} or "")
                if payload.get("type") == "user_transcript":
                    transcript_seen = payload.get("text", "")
                if payload.get("type") == "error":
                    raise RuntimeError(f"server error: {payload.get('message')}")

        recv_task = asyncio.create_task(receiver())

        # Stream like a real mic: ~20ms chunks, real-time paced, matching
        # frontend/index.html's own chunking so the server's VAD sees the
        # same shape of input it always sees.
        chunk_ms = 20
        chunk_bytes = int(TARGET_RATE * 2 * chunk_ms / 1000)
        print(f"-> streaming {len(pcm) // chunk_bytes} audio chunks over the wire ...")
        for i in range(0, len(pcm), chunk_bytes):
            await ws.send(pcm[i:i + chunk_bytes])
            await asyncio.sleep(chunk_ms / 1000)

        # Real trailing silence (not a magic marker) -- this IS how the
        # server's own VAD (server_vad, silence_duration_ms=700, see
        # providers/pipeline.py) decides the turn is over. A real mic
        # streams silence too; stopping the send loop entirely wouldn't
        # match how a live connection actually behaves.
        print("-> streaming trailing silence so server-side VAD can close the turn ...")
        silence_chunk = b"\x00" * chunk_bytes
        for _ in range(int(1.5 * 1000 / chunk_ms)):
            await ws.send(silence_chunk)
            await asyncio.sleep(chunk_ms / 1000)

        print("-> done sending; waiting up to 30s for transcription + answer ...")
        try:
            await asyncio.wait_for(recv_task, timeout=30.0)
        except asyncio.TimeoutError:
            recv_task.cancel()

        print()
        print(f"transcript recognized: {transcript_seen!r}")
        print(f"total binary (TTS/answer) audio bytes received: {audio_bytes_total}")
        print(f"event types seen: {events_seen}")

        assert transcript_seen, "expected a real transcript from the audio-native path"
        if spoken_text_hint:
            key_word = spoken_text_hint.split()[-1].strip(".").lower()
            assert key_word in transcript_seen.lower(), (
                f"expected transcript to contain {key_word!r} from the synthesized speech, "
                f"got {transcript_seen!r}"
            )
        assert "answer_done" in events_seen, "expected the turn to reach answer_done"
        assert audio_bytes_total > 0, "expected some TTS/answer audio bytes back"
        print("\nAUDIO SMOKE TEST OK -- fed a real audio file directly into the backend, "
              "no browser/frontend involved at any point.")


if __name__ == "__main__":
    asyncio.run(main())
