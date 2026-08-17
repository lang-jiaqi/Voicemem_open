"""Measures real end-to-end latency: from the moment real speech audio stops
(end of utterance, before trailing silence) to each milestone in the
server's response -- transcript locked in, memory retrieved, answer text
starts, and (most importantly) the first playable reply audio byte arrives.
That last number is "说完话到听到回复要多久" -- what a real user actually
experiences as the pause before the assistant starts talking back.

Same wire protocol as audio_smoke_test.py (real PCM16/24kHz/mono streamed
over the WebSocket, no browser involved) -- see that script for the
underlying mechanics; this one adds timing instrumentation instead of just
proving the round trip works.

Usage:
    python scripts/latency_test.py [path/to/recording.wav] [ws://host:port/ws] [--runs N]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
import wave

import numpy as np
import websockets
from openai import OpenAI

DEFAULT_URL = "ws://localhost:8787/ws"
TARGET_RATE = 24000


def synth_test_audio(text: str) -> bytes:
    client = OpenAI()
    resp = client.audio.speech.create(
        model="tts-1", voice="alloy", input=text, response_format="pcm",
    )
    return resp.read()


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


async def run_once(url: str, pcm: bytes, run_label: str) -> dict:
    """One turn, timed from end-of-speech. Returns a dict of milestone -> seconds."""
    milestones: dict[str, float] = {}
    t_end_of_speech: float | None = None
    got_first_answer_delta = False
    got_first_audio_byte = False

    async with websockets.connect(url, max_size=None) as ws:
        async def receiver():
            nonlocal got_first_answer_delta, got_first_audio_byte
            async for msg in ws:
                now = time.monotonic()
                if isinstance(msg, (bytes, bytearray)):
                    if t_end_of_speech is not None and not got_first_audio_byte:
                        got_first_audio_byte = True
                        milestones["first_reply_audio_byte"] = now - t_end_of_speech
                    continue
                payload = json.loads(msg)
                et = payload.get("type")
                if t_end_of_speech is None:
                    continue  # ignore session_ready / anything before we start timing
                if et == "user_transcript" and "transcript_locked" not in milestones:
                    milestones["transcript_locked"] = now - t_end_of_speech
                elif et == "memory_hits" and "memory_hits" not in milestones:
                    milestones["memory_hits"] = now - t_end_of_speech
                elif et == "answer_start" and "answer_start" not in milestones:
                    milestones["answer_start"] = now - t_end_of_speech
                elif et == "answer_delta" and not got_first_answer_delta:
                    got_first_answer_delta = True
                    milestones["first_answer_text_delta"] = now - t_end_of_speech
                elif et == "answer_done":
                    milestones["answer_done"] = now - t_end_of_speech
                    return

        recv_task = asyncio.create_task(receiver())

        chunk_ms = 20
        chunk_bytes = int(TARGET_RATE * 2 * chunk_ms / 1000)
        for i in range(0, len(pcm), chunk_bytes):
            await ws.send(pcm[i:i + chunk_bytes])
            await asyncio.sleep(chunk_ms / 1000)
        t_end_of_speech = time.monotonic()  # last real speech byte just left the wire
        print(f"  [{run_label}] end of speech at t=0, streaming trailing silence ...")

        silence_chunk = b"\x00" * chunk_bytes
        for _ in range(int(1.5 * 1000 / chunk_ms)):
            await ws.send(silence_chunk)
            await asyncio.sleep(chunk_ms / 1000)

        try:
            await asyncio.wait_for(recv_task, timeout=30.0)
        except asyncio.TimeoutError:
            recv_task.cancel()
            print(f"  [{run_label}] WARNING: timed out waiting for answer_done")

    return milestones


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wav", nargs="?", default=None)
    parser.add_argument("url", nargs="?", default=DEFAULT_URL)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    # Allow the documented URL-only form while retaining the optional WAV
    # positional argument: `latency_test.py ws://host:port/ws`.
    if args.wav and args.wav.startswith(("ws://", "wss://")):
        args.url, args.wav = args.wav, None

    if args.wav:
        print(f"loading {args.wav} ...")
        pcm = load_wav_as_24k_pcm(args.wav)
    else:
        text = "What's the weather like for a picnic this weekend?"
        print(f"no audio file given -- synthesizing TTS for: {text!r}")
        pcm = synth_test_audio(text)
    print(f"got {len(pcm)} bytes (~{len(pcm) / (TARGET_RATE * 2):.2f}s) of test audio")
    print(f"connecting to {args.url}, {args.runs} run(s) ...\n")

    all_runs = []
    for i in range(args.runs):
        milestones = await run_once(args.url, pcm, f"run {i + 1}/{args.runs}")
        all_runs.append(milestones)
        for key in ("transcript_locked", "memory_hits", "answer_start",
                    "first_answer_text_delta", "first_reply_audio_byte", "answer_done"):
            v = milestones.get(key)
            print(f"    {key:28s} {v * 1000:7.0f} ms" if v is not None else f"    {key:28s}    (missing)")
        print()
        await asyncio.sleep(1.0)

    print("=" * 60)
    print("summary (mean of successful runs, ms from end-of-speech):")
    for key in ("transcript_locked", "memory_hits", "answer_start",
                "first_answer_text_delta", "first_reply_audio_byte", "answer_done"):
        vals = [m[key] for m in all_runs if key in m]
        if vals:
            mean_ms = sum(vals) / len(vals) * 1000
            print(f"  {key:28s} {mean_ms:7.0f} ms  (n={len(vals)})")
        else:
            print(f"  {key:28s}    no data")


if __name__ == "__main__":
    asyncio.run(main())
