"""Opt-in mode: browser audio -> Qwen-Omni-Realtime (DashScope, Beijing).
Same shape as providers/realtime.py -- the mic PCM itself streams into the
session so the model HEARS the user, while the LOCAL streaming ASR stays the
control plane (captions, memory-search query, barge-in confirmation,
Ingest()'s transcript). Raw `websockets` client, no dashscope SDK: the wire
protocol is documented JSON events in OpenAI-Realtime style, and the async
SDK class (OmniRealtimeConversation) is callback-based -- a poor fit for
this file's asyncio structure and one more dependency for no gain.

Why this provider exists at all (vs. the OpenAI one): dashscope.aliyuncs.com
terminates in Beijing -- tens of ms RTT from here instead of a trans-Pacific
round trip -- and the qwen-omni models are Chinese-native. It is still a
TURN-BASED model though: no full-duplex backchannel ("嗯嗯/是啊"), same
paradigm as gpt-realtime, just closer and cheaper.

Turn-taking: MANUAL, deliberately. DashScope's semantic_vad documents no
`create_response: false`, so in that mode the SERVER would fire the reply
before we could inject this turn's memory -- the per-turn fresh-memory
guarantee would become a race. With `turn_detection: null` the local ASR's
turn_end drives everything: finalize caption -> inject memory into
`instructions` via session.update -> input_audio_buffer.commit ->
response.create. That restores the local 0.5s silence window to the critical
path (the thing semantic VAD removed in the OpenAI provider), but the
network savings absorb much of it, and memory injection is deterministic
again instead of speculative. If a live probe ever shows this endpoint DOES
accept create_response:false under semantic_vad, the OpenAI provider's
commit-driven architecture can be ported over as a small change.

Schema note: field shapes follow the current Qwen-Omni-Realtime docs
(nested audio.input/output format objects, top-level modalities/voice/
instructions/turn_detection). They are close to, but not identical to,
OpenAI's schema, and could not be verified against a live endpoint while
this was written (no DASHSCOPE_API_KEY on hand) -- so every server `error`
event is printed verbatim AND forwarded to the browser, and the audio/
transcript delta handlers accept both the flat ("response.audio.delta") and
GA-style ("response.output_audio.delta") event names. First live smoke test
pins whichever the endpoint actually speaks.

Barge-in: the same two-stage local flow as the other providers (see
pipeline.py's docstring): speech onset -> reversible speech_tentative;
enough real transcript mid-utterance -> response.cancel + answer_interrupt;
resolves-to-noise -> answer_resume. Manual turn mode means the server never
auto-interrupts, so nothing fights the local logic by construction.

Typed-text turns: this protocol documents no conversation.item.create, so
typed input is delivered as an addendum inside `instructions` (session.update)
followed by a bare response.create -- semantically equivalent for a demo,
disclosed here because it IS a hack.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import time
import traceback
import uuid

import websockets
from fastapi import WebSocket, WebSocketDisconnect

from config import Config
from local_asr import LocalASR
from memory_bridge import MemoryBridge

# Beijing public endpoint. Model is passed as a query param. Override for the
# intl endpoint (dashscope-intl.aliyuncs.com) or a workspace-scoped URL.
QWEN_REALTIME_URL = os.environ.get(
    "QWEN_REALTIME_URL", "wss://dashscope.aliyuncs.com/api-ws/v1/realtime")
# Browser sends 24kHz PCM16 mono; the docs list 24000 among the supported
# input rates, so no resample on the cloud path (the local ASR keeps its own
# internal 24k->16k resampler for the zipformer).
SAMPLE_RATE = 24000
# Same constants + reasoning as pipeline.py / realtime.py.
SPECULATIVE_SEARCH_MIN_CHARS = 15
MIN_INTERRUPT_CHARS = 3


def _log_task_exc(task: "asyncio.Task") -> None:
    """Done-callback for fire-and-forget tasks whose result nobody awaits."""
    if not task.cancelled() and task.exception() is not None:
        traceback.print_exception(task.exception())


def _response_id(event: dict) -> str | None:
    """Events carry the id either flat (response_id) or nested (response.id)
    depending on schema generation -- accept both."""
    return event.get("response_id") or (event.get("response") or {}).get("id")


class QwenRealtimeProvider:
    def __init__(self, config: Config) -> None:
        self._config = config
        self._api_key = os.environ.get("DASHSCOPE_API_KEY", "")

    async def _send(self, conn, payload: dict) -> None:
        await conn.send(json.dumps(payload))

    def _session_payload(self, instructions: str) -> dict:
        return {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "voice": self._config.qwen_voice,
                "instructions": instructions,
                "audio": {
                    "input": {"format": {"type": "pcm", "sample_rate": SAMPLE_RATE}},
                    "output": {"format": {"type": "pcm", "sample_rate": SAMPLE_RATE}},
                },
                # Manual turn mode -- module docstring has the reasoning
                # (memory injection must beat response creation, and this
                # endpoint documents no create_response:false).
                "turn_detection": None,
            },
        }

    async def run_session(self, ws: WebSocket, memory: MemoryBridge) -> None:
        if not self._api_key:
            await ws.send_json({"type": "error", "message":
                                "DASHSCOPE_API_KEY 未设置 -- Qwen realtime 模式需要阿里云百炼的 sk- key"})
            return
        await ws.send_json({"type": "session_ready",
                             "mode": "qwen",
                             "audio_native": self._config.audio_native})

        url = f"{QWEN_REALTIME_URL}?model={self._config.qwen_realtime_model}"
        # max_size=None: the only peer is DashScope itself and audio deltas
        # can be large; the default 1MB cap would kill the session mid-reply.
        async with websockets.connect(
                url,
                additional_headers={"Authorization": f"Bearer {self._api_key}"},
                max_size=None) as qwen:
            await self._send(qwen, self._session_payload(self._config.system_prompt))

            asr = LocalASR(turn_silence_s=float(os.environ.get("QWEN_TURN_SILENCE_S", "0.5")))

            local_pcm = bytearray()
            state = {"turn_counter": 0, "spec_partial_text": "", "spec_task": None,
                     "tentative": False, "utt_transcript_sent": True,
                     "response_active": False, "active_response_id": None,
                     # ids we cancelled ourselves: their response.done must not
                     # emit answer_done even if the payload carries no status.
                     "cancelled_ids": set(),
                     "last_commit_t": 0.0}

            async def handle_asr_event(ev: tuple) -> None:
                kind = ev[0]
                if kind == "speech_start":
                    del local_pcm[:]
                    state["spec_partial_text"] = ""
                    state["spec_task"] = None
                    state["tentative"] = True
                    state["utt_transcript_sent"] = False
                    print("[vad] speech_start -> tentative pause sent", flush=True)
                    await ws.send_json({"type": "speech_tentative"})
                elif kind == "partial":
                    full_text = ev[1]
                    if not state["utt_transcript_sent"]:
                        await ws.send_json({"type": "partial_transcript",
                                            "text": full_text, "replace": True})
                    state["spec_partial_text"] = full_text
                    if (state["tentative"]
                            and len(full_text) >= MIN_INTERRUPT_CHARS):
                        state["tentative"] = False
                        print(f"[vad] confirmed interrupt ({len(full_text)} chars)", flush=True)
                        if state["response_active"]:
                            await self._cancel_active_response(ws, qwen, state)
                        else:
                            # Client may still be playing buffered tail audio;
                            # only it knows (same reasoning as the other
                            # providers -- real bug found via user testing).
                            await ws.send_json({"type": "answer_interrupt"})
                    if (state["spec_task"] is None
                            and len(state["spec_partial_text"]) >= SPECULATIVE_SEARCH_MIN_CHARS):
                        state["spec_task"] = asyncio.create_task(
                            memory.search(query=state["spec_partial_text"], top_k=5))
                elif kind == "turn_end":
                    transcript = ev[1]
                    spec_task = state["spec_task"]
                    state["spec_task"] = None
                    if state["tentative"]:
                        state["tentative"] = False
                        print("[vad] resolved as noise/short -> resume sent", flush=True)
                        await ws.send_json({"type": "answer_resume"})
                    if transcript:
                        # Manual mode: local turn_end IS the turn authority --
                        # caption, ingest, memory injection, commit, respond.
                        print(f"[turn_end] local(+0.5s静音) {len(transcript)}字 -> respond", flush=True)
                        if not state["utt_transcript_sent"]:
                            await ws.send_json({"type": "user_transcript", "text": transcript})
                            state["utt_transcript_sent"] = True
                        pcm = bytes(local_pcm)
                        local_pcm.clear()
                        state["turn_counter"] += 1
                        turn_id = f"{state['turn_counter']}-{uuid.uuid4().hex[:8]}"
                        asyncio.create_task(self._background_ingest(
                            memory, transcript, pcm if pcm else None,
                            turn_id, state["turn_counter"]))
                        # Off the mic-frame critical path: memory.search()
                        # may take a while and audio must keep flowing.
                        task = asyncio.create_task(self._respond_with_memory(
                            ws, memory, qwen, transcript, state,
                            spec_task, commit_audio=True))
                        task.add_done_callback(_log_task_exc)
                    else:
                        # Noise/empty turn: nothing committed, so the junk
                        # audio never becomes conversation context (it just
                        # rides along in the buffer until a real turn's
                        # commit -- leading silence, which the model tolerates).
                        local_pcm.clear()
                        if spec_task is not None and not spec_task.done():
                            spec_task.cancel()

            async def browser_to_qwen() -> None:
                level_window_max = 0
                level_window_start = 0.0
                while True:
                    try:
                        message = await ws.receive()
                    except WebSocketDisconnect:
                        return
                    if message["type"] == "websocket.disconnect":
                        return

                    if (data := message.get("bytes")) is not None:
                        local_pcm.extend(data)
                        # Continuous append, commit only at local turn_end.
                        # Between-turn silence therefore gets committed as
                        # leading audio of the next turn -- accepted trade-off
                        # (clearing the buffer at any "safe-looking" moment
                        # races with barge-in speech onset and would eat the
                        # first syllables).
                        await self._send(qwen, {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(data).decode("ascii")})
                        samples = memoryview(data).cast("h")
                        peak = max(abs(min(samples)), abs(max(samples))) if len(samples) else 0
                        level_window_max = max(level_window_max, peak)
                        now = time.monotonic()
                        if now - level_window_start >= 1.0:
                            print(f"[mic] peak={level_window_max} ({level_window_max / 327.67:.0f}% of full scale)", flush=True)
                            level_window_max = 0
                            level_window_start = now
                        for ev in asr.process(data):
                            await handle_asr_event(ev)
                        continue

                    text = message.get("text")
                    if not text:
                        continue
                    try:
                        payload = json.loads(text)
                    except json.JSONDecodeError:
                        continue
                    if payload.get("type") == "user_text":
                        user_text = payload.get("text", "")
                        if user_text:
                            await ws.send_json({"type": "user_transcript", "text": user_text})
                            state["turn_counter"] += 1
                            turn_id = f"{state['turn_counter']}-{uuid.uuid4().hex[:8]}"
                            asyncio.create_task(self._background_ingest(
                                memory, user_text, None, turn_id, state["turn_counter"]))
                            # Typed-text hack disclosed in the module
                            # docstring: no item.create in this protocol, so
                            # the text rides inside instructions instead.
                            await self._respond_with_memory(
                                ws, memory, qwen, user_text, state,
                                None, commit_audio=False, typed_text=user_text)

            async def qwen_to_browser() -> None:
                async for raw in qwen:
                    try:
                        event = json.loads(raw)
                    except (json.JSONDecodeError, TypeError):
                        continue
                    et = event.get("type", "")
                    if et in ("response.audio.delta", "response.output_audio.delta"):
                        if _response_id(event) == state["active_response_id"]:
                            if state.pop("await_first_audio", False):
                                now = time.monotonic()
                                commit_t = state.get("last_commit_t") or now
                                create_t = state.get("respond_t0") or now
                                print(f"[ttfb] commit→首音 {round((now - commit_t) * 1000)}ms"
                                      f" (create→首音 {round((now - create_t) * 1000)}ms)",
                                      flush=True)
                            await ws.send_bytes(base64.b64decode(event.get("delta", "")))
                    elif et in ("response.audio_transcript.delta",
                                "response.output_audio_transcript.delta"):
                        # response.text.delta (the text-modality channel) is
                        # deliberately NOT forwarded -- it would double the
                        # caption alongside this audio-transcript stream.
                        if _response_id(event) == state["active_response_id"]:
                            await ws.send_json({"type": "answer_delta",
                                                "text": event.get("delta", "")})
                    elif et == "response.created":
                        state["response_active"] = True
                        state["active_response_id"] = _response_id(event)
                        await ws.send_json({"type": "answer_start"})
                    elif et == "response.done":
                        rid = _response_id(event)
                        waiter = state.pop(f"cancel_wait_{rid}", None)
                        if waiter is not None:
                            waiter.set()
                        if rid == state["active_response_id"]:
                            state["response_active"] = False
                            state["active_response_id"] = None
                            status = (event.get("response") or {}).get("status")
                            if status != "cancelled" and rid not in state["cancelled_ids"]:
                                await ws.send_json({"type": "answer_done"})
                        state["cancelled_ids"].discard(rid)
                    elif et in ("session.created", "session.updated"):
                        print(f"[qwen] {et}", flush=True)
                    elif et == "error":
                        # Schema mismatches surface HERE (module docstring:
                        # field shapes are docs-derived, unverified live) --
                        # verbatim print is the debugging lifeline.
                        print(f"[qwen] ERROR event: {json.dumps(event, ensure_ascii=False)}", flush=True)
                        await ws.send_json({"type": "error",
                                            "message": str(event.get("error") or event)})

            await asyncio.gather(browser_to_qwen(), qwen_to_browser())

    async def _respond_with_memory(self, ws: WebSocket, memory: MemoryBridge, qwen,
                                    query: str, state: dict,
                                    spec_task: "asyncio.Task | None" = None,
                                    commit_audio: bool = True,
                                    typed_text: str | None = None) -> None:
        """Search memory (reusing the speculative mid-utterance task when
        there is one), inject the result into `instructions`, commit the
        audio buffer (voice turns), then create the response."""
        t0 = time.monotonic()
        result = None
        if spec_task is not None:
            try:
                result = await spec_task
            except asyncio.CancelledError:
                result = await memory.search(query=query, top_k=5) if query else None
        elif query:
            result = await memory.search(query=query, top_k=5)
        search_ms = round((time.monotonic() - t0) * 1000)
        print(f"[respond] search={'spec命中' if search_ms <= 50 else '关键路径'} {search_ms}ms "
              f"query={len(query)}字", flush=True)

        instructions = self._config.system_prompt
        if result is not None:
            await ws.send_json({"type": "memory_hits", **memory.search_payload(result)})
            instructions = memory.build_instructions(self._config.system_prompt, result)
        if typed_text is not None:
            instructions += f"\n\nThe user just sent this TYPED message; respond to it:\n{typed_text}"
        await self._send(qwen, {"type": "session.update",
                                "session": {"instructions": instructions}})

        if state["response_active"]:
            await self._cancel_active_response(ws, qwen, state)
        if commit_audio:
            state["last_commit_t"] = time.monotonic()
            await self._send(qwen, {"type": "input_audio_buffer.commit"})
        state["respond_t0"] = time.monotonic()
        state["await_first_audio"] = True
        await self._send(qwen, {"type": "response.create"})

    async def _cancel_active_response(self, ws: WebSocket, qwen, state: dict) -> None:
        """Cancel the active response, wait for its response.done, and tell
        the client to flush -- same reasoning (and the same known remaining
        limitation) as the OpenAI provider's version of this method: creating
        the next response before the old one confirmably closes lets stale
        deltas bleed across."""
        cancelled_id = state["active_response_id"]
        if cancelled_id:
            state["cancelled_ids"].add(cancelled_id)
        await self._send(qwen, {"type": "response.cancel"})
        state["response_active"] = False
        await ws.send_json({"type": "answer_interrupt"})

        if cancelled_id:
            done_event = asyncio.Event()
            state[f"cancel_wait_{cancelled_id}"] = done_event
            try:
                await asyncio.wait_for(done_event.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                state.pop(f"cancel_wait_{cancelled_id}", None)

    @staticmethod
    async def _background_ingest(memory: MemoryBridge, transcript: str, pcm: bytes | None,
                                   turn_id: str, session_id: int) -> None:
        try:
            await memory.ingest(transcript, pcm=pcm, turn_id=turn_id, speaker="user",
                                 session_id=session_id)
        except Exception:
            traceback.print_exc()
