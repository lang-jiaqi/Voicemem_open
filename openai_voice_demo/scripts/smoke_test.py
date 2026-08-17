"""Scripted WebSocket client. Assumes the backend is already running in
`pipeline` mode (e.g. `VOICEMEM_AUDIO_NATIVE=false VOICE_MODE=pipeline python
main.py`). Text-only — no microphone needed.

What this proves: the full turn -> STT-skip -> Search() -> chat completion ->
TTS -> background Ingest() wiring works over a real WebSocket connection, and
that a fact ingested on turn 1 is actually retrievable via memory_hits on a
related turn 2. What it does NOT prove: audio capture/playback quality, or
anything about realtime mode (use the browser client for those).

Usage:
    python scripts/smoke_test.py [ws://localhost:8787/ws]
"""
from __future__ import annotations

import asyncio
import json
import sys

import websockets

DEFAULT_URL = "ws://localhost:8787/ws"


async def main() -> None:
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
    print(f"connecting to {url} ...")

    async with websockets.connect(url) as ws:
        audio_bytes_total = 0
        second_turn_hits: list[dict] = []

        async def send_text_turn(text: str) -> None:
            await ws.send(json.dumps({"type": "user_text", "text": text}))

        async def drain_until(event_type: str, *, collect_hits_into: list | None = None) -> None:
            nonlocal audio_bytes_total
            while True:
                msg = await ws.recv()
                if isinstance(msg, (bytes, bytearray)):
                    audio_bytes_total += len(msg)
                    continue
                payload = json.loads(msg)
                print("  <-", payload.get("type"), {k: v for k, v in payload.items() if k != "left_brain"} or "")
                if payload.get("type") == "memory_hits" and collect_hits_into is not None:
                    collect_hits_into.extend(payload.get("left_brain") or [])
                if payload.get("type") == event_type:
                    return
                if payload.get("type") == "error":
                    raise RuntimeError(f"server error: {payload.get('message')}")

        # session_ready
        await drain_until("session_ready")

        # turn 1: ingest a fact. answer_done now only fires after TTS audio has
        # also finished streaming (see providers/pipeline.py), so it's safe to
        # treat this as "turn 1 fully settled" -- except the background
        # Ingest() task is fired-and-forgotten, not awaited, so the fact isn't
        # guaranteed searchable the instant answer_done arrives. That's a real,
        # by-design eventual-consistency gap (search-fast-path / ingest
        # background-path), so this test retries rather than assuming a fixed
        # delay is enough.
        print("-> user_text: My favorite food is ramen.")
        await send_text_turn("My favorite food is ramen.")
        await drain_until("answer_done")

        # turn 2: ask a related question, retrying until memory_hits surfaces
        # turn 1's fact (or we give up) -- proves Search()/Ingest() wiring
        # works end-to-end, tolerant of ingest's background latency.
        found = False
        for attempt in range(6):
            print(f"-> user_text: What food do I like? (attempt {attempt + 1})")
            second_turn_hits.clear()
            await send_text_turn("What food do I like?")
            await drain_until("answer_done", collect_hits_into=second_turn_hits)
            # Language-agnostic check: the extraction LLM writes the stored
            # fact in whatever language dominates the store's context (real
            # failure seen: a store full of Chinese conversation made it
            # store "用户的最爱食物是拉面", which "ramen" alone missed even
            # though retrieval ranked it #1).
            if any("ramen" in h.get("text", "").lower() or "拉面" in h.get("text", "")
                   for h in second_turn_hits):
                found = True
                break
            await asyncio.sleep(2.0)

        print(f"\ntotal binary (TTS) audio bytes received: {audio_bytes_total}")
        print(f"final memory_hits: {second_turn_hits}")

        assert audio_bytes_total > 0, "expected some TTS audio bytes"
        assert found, "expected memory_hits to eventually recall the ramen fact ingested in turn 1"
        print("\nSMOKE TEST OK")


if __name__ == "__main__":
    asyncio.run(main())
