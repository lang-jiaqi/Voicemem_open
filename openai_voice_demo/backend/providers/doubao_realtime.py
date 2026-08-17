"""Opt-in mode: browser audio -> 豆包实时语音 3.0 全双工 (Seeduplex),
wss://openspeech.bytedance.com/api/v3/duplex/realtime/dialogue. THE
full-duplex path: unlike gpt-realtime / qwen-omni-realtime (both strictly
turn-based), this model owns turn-taking end-to-end -- it decides when to
listen, when to backchannel ("嗯嗯"), when to answer, and when to stop.

Protocol: the NEW-console JSON protocol (X-Api-Key auth, JSON text frames,
OpenAI-style event names), NOT the older binary-framed
/api/v3/realtime/dialogue API (APP ID + Access Token; being phased out).
Every event shape here was pinned against the official python3.7_duplex_demo
and web_duplex_demo published in the 全双工版本 docs, not guessed:
session.create carries {id, model:"1.2.6.1", instructions, audio{input pcm
16k / output pcm_s16le 24k, voice}, tools} plus an `extension` passthrough;
mic PCM streams as base64 input_audio_buffer.append with NO commit (the
server is the turn authority in duplex); ASR comes back as
conversation.item.input_audio_transcription.started/.delta/.completed; the
reply as response.output_text.delta + response.output_audio.started/.delta/
.done; bot-speech can be forced with speech_text_buffer.commit.

Architecture difference from the other two providers, and why: there is NO
local ASR here. The others need the local zipformer because their cloud
sides either transcribe hallucination-prone (OpenAI, documented in
realtime.py) or need a client-side turn authority. Here the server streams
its own Chinese-native ASR back, and in full duplex the SERVER decides
turns -- a local 0.5s window would have nothing to decide. So: captions
come from transcription deltas, Ingest()'s transcript from the completed
transcript, and barge-in reduces to "flush stale buffered reply audio when
transcription.started says the user began talking" (exactly what the
official demo does with its playback queue); whether the model keeps
talking, backchannels, or goes quiet is its own duplex decision and simply
arrives as further audio deltas.

Memory injection: session.update with rebuilt `instructions` -- the same
mechanism the OpenAI provider uses, and demo-confirmed to exist mid-session
here. The race is different though: in duplex the SERVER starts answering
whenever it judges the user done, so injection cannot gate response
creation. Mitigation is to inject EARLY: memory.search() fires on the
accumulating transcription delta once it's long enough and the result is
injected the moment it lands (usually mid-utterance), with a final-text
fallback injection at transcription.completed for short utterances. The
[inject]/[ttfb] log lines exist to verify who wins that race with real
timings instead of assuming.

The 24k->16k upload resample (browser captures 24k, service requires 16k)
mirrors local_asr.py's proven streaming linear-interp approach. Output is
pcm_s16le 24k, which the frontend already plays natively.

Typed-text turns: sent as conversation.item.create (role user) -- the demo
only uses item.create for tool results, so whether a bare user item makes
the duplex model respond is UNVERIFIED; voice is the primary path here and
a live test will settle it.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import time
import traceback
import uuid

import numpy as np
import websockets
from fastapi import WebSocket, WebSocketDisconnect

from config import Config
from memory_bridge import MemoryBridge

DOUBAO_DUPLEX_URL = os.environ.get(
    "DOUBAO_DUPLEX_URL", "wss://openspeech.bytedance.com/api/v3/duplex/realtime/dialogue")
# 新版控制台 -> API Key 管理 (the UUID-format key; NOT the old APP ID/Access
# Token pair -- those belong to the legacy binary API).
DOUBAO_API_KEY = os.environ.get("DOUBAO_API_KEY", "")
# "1.2.6.1" is the documented fixed value for 实时语音3.0-全双工.
DOUBAO_MODEL = os.environ.get("DOUBAO_MODEL", "1.2.6.1")
DOUBAO_VOICE = os.environ.get("DOUBAO_VOICE", "zh_female_xiaohe_jupiter_bigtts")
# Same constant + reasoning as the other providers.
SPECULATIVE_SEARCH_MIN_CHARS = 15


class _Resampler24to16:
    """Streaming 24k->16k for the UPLOAD path (the service only accepts
    16kHz PCM in). Same carry/fractional-position linear interpolation as
    local_asr.py's _to_16k_floats -- proven against chunk-boundary
    artifacts -- but emitting int16 bytes, not zipformer floats."""

    def __init__(self) -> None:
        self._carry: np.ndarray | None = None
        self._pos = 0.0

    def process(self, pcm24k: bytes) -> bytes:
        x = np.frombuffer(pcm24k, dtype=np.int16).astype(np.float32)
        if self._carry is None:
            self._carry = np.zeros(0, dtype=np.float32)
        buf = np.concatenate((self._carry, x))
        ratio = 1.5  # 24000 / 16000
        if len(buf) < 2 or self._pos > len(buf) - 1:
            self._carry = buf
            return b""
        n_out = int((len(buf) - 1 - self._pos) / ratio) + 1
        idx = self._pos + np.arange(n_out) * ratio
        i0 = idx.astype(np.int64)
        frac = (idx - i0).astype(np.float32)
        vals = buf[i0] * (1.0 - frac) + buf[np.minimum(i0 + 1, len(buf) - 1)] * frac
        next_pos = self._pos + n_out * ratio
        keep_from = int(next_pos)
        self._carry = buf[keep_from:]
        self._pos = next_pos - keep_from
        return np.clip(np.rint(vals), -32768, 32767).astype(np.int16).tobytes()


def _log_task_exc(task: "asyncio.Task") -> None:
    """Done-callback for fire-and-forget tasks whose result nobody awaits."""
    if not task.cancelled() and task.exception() is not None:
        traceback.print_exception(task.exception())


class DoubaoRealtimeProvider:
    # The one CURRENT-TURN memory path this API has. Probed live (scratchpad
    # fc_memory_probe.py): session.update instructions AND conversation items
    # both reach the model exactly one turn late -- the turn's context is
    # assembled before speech even starts -- but a function call pauses
    # generation mid-turn and continues WITH the tool result (判停→FC 605ms,
    # tool回传→首音 939ms measured). So: portrait covers ambient
    # personalization at zero cost, this tool covers explicit recall in the
    # current turn, and the lag-one injection enriches whatever follows.
    _MEMORY_TOOLS = [{
        "type": "function",
        "name": "recall_memory",
        "description": ("回忆你和用户的过往。当用户问你是否记得某件事,或话题涉及用户的"
                         "个人信息、家人朋友、宠物、爱好、计划、偏好时,先调用它查询你的"
                         "长期记忆,再基于结果回答。纯寒暄闲聊不需要调用。"),
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string",
                                                 "description": "要回忆的内容,用一句话描述"}},
                       "required": ["query"]},
    }]

    # Canonical portrait queries, one per facet of who the user is. Run once
    # at session start to bake a standing portrait into the instructions.
    _PORTRAIT_QUERIES = (
        "我叫什么名字 我的身份",
        "我的家人",
        "我的朋友和爱好",
        "我的学习工作近况",
        "我最近的计划和安排",
        "我的饮食偏好和忌口",
        "我情绪不好的时候希望被怎么对待",
    )

    def __init__(self, config: Config) -> None:
        self._config = config
        self._event_seq = 0
        self._base_instructions = config.system_prompt

    def _event_id(self) -> str:
        self._event_seq += 1
        return f"vm_{self._event_seq}"

    async def _send(self, db, payload: dict) -> None:
        await db.send(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))

    def _session_create_payload(self) -> dict:
        return {
            "type": "session.create",
            "event_id": self._event_id(),
            "session": {
                "id": str(uuid.uuid4()),
                "model": DOUBAO_MODEL,
                "instructions": self._base_instructions,
                "tools": self._MEMORY_TOOLS,
                "audio": {
                    "input": {"format": {"type": "pcm", "rate": 16000}},
                    "output": {"format": {"type": "pcm_s16le", "rate": 24000},
                               "voice": DOUBAO_VOICE},
                },
            },
            # Passthrough to the underlying asr/tts/dialog stacks (same
            # structure as the legacy StartSession payload); empty extras =
            # service defaults.
            "extension": {"asr": {"extra": {}}, "tts": {"extra": {}},
                          "dialog": {"extra": {}}},
        }

    async def run_session(self, ws: WebSocket, memory: MemoryBridge) -> None:
        if not DOUBAO_API_KEY:
            await ws.send_json({"type": "error", "message":
                                "DOUBAO_API_KEY 未设置 -- 豆包全双工模式需要新版控制台的 API Key"})
            return
        await ws.send_json({"type": "session_ready",
                             "mode": "doubao",
                             "audio_native": self._config.audio_native})

        self._base_instructions = await self._build_portrait_instructions(memory)

        async with websockets.connect(
                DOUBAO_DUPLEX_URL,
                additional_headers={"X-Api-Key": DOUBAO_API_KEY},
                max_size=None) as db:
            await self._send(db, self._session_create_payload())

            resampler = _Resampler24to16()
            utt_pcm = bytearray()  # 24k原声, per-utterance, for Ingest()'s archival
            state = {"turn_counter": 0, "spec_task": None,
                     "asr_accum": "", "utt_active": False, "injected": False,
                     "answer_open": False, "await_first_audio": False,
                     "t_final": 0.0, "last_result": None}

            async def browser_to_doubao() -> None:
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
                        if state["utt_active"]:
                            utt_pcm.extend(data)
                        pcm16k = resampler.process(data)
                        if pcm16k:
                            # Continuous, uncommitted: the duplex server
                            # hears everything and owns the turn decisions.
                            await self._send(db, {
                                "type": "input_audio_buffer.append",
                                "audio": base64.b64encode(pcm16k).decode("ascii")})
                        samples = memoryview(data).cast("h")
                        peak = max(abs(min(samples)), abs(max(samples))) if len(samples) else 0
                        level_window_max = max(level_window_max, peak)
                        now = time.monotonic()
                        if now - level_window_start >= 1.0:
                            print(f"[mic] peak={level_window_max} ({level_window_max / 327.67:.0f}% of full scale)", flush=True)
                            level_window_max = 0
                            level_window_start = now
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
                            self._spawn_ingest(memory, user_text, None, state)
                            task = asyncio.create_task(self._inject_memory(
                                ws, memory, db, user_text, state, stage="typed"))
                            task.add_done_callback(_log_task_exc)
                            # Module docstring: unverified whether a bare user
                            # item makes the duplex model respond -- voice is
                            # the primary path.
                            await self._send(db, {
                                "type": "conversation.item.create",
                                "event_id": self._event_id(),
                                "items": [{"type": "message", "role": "user",
                                            "content": [{"type": "input_text",
                                                         "text": user_text}]}]})

            async def doubao_to_browser() -> None:
                async for raw in db:
                    try:
                        event = json.loads(raw)
                    except (json.JSONDecodeError, TypeError):
                        continue
                    et = event.get("type", "")

                    if et == "response.output_audio.delta":
                        if state["await_first_audio"] and state["t_final"]:
                            state["await_first_audio"] = False
                            print(f"[ttfb] 判停→首音 "
                                  f"{round((time.monotonic() - state['t_final']) * 1000)}ms",
                                  flush=True)
                        delta = event.get("delta") or ""
                        if delta:
                            await ws.send_bytes(base64.b64decode(delta))
                    elif et == "conversation.item.input_audio_transcription.started":
                        # Server heard the user start talking: flush reply
                        # audio the client buffered but hasn't played (the
                        # official demo clears its playback queue on exactly
                        # this event). What the model does NEXT -- keep
                        # talking, backchannel, go quiet -- is its own duplex
                        # decision, arriving as further audio deltas.
                        print("[asr] speech onset (server)", flush=True)
                        utt_pcm.clear()
                        state["utt_active"] = True
                        state["asr_accum"] = ""
                        state["injected"] = False
                        state["last_result"] = None
                        if state["spec_task"] is not None and not state["spec_task"].done():
                            state["spec_task"].cancel()
                        state["spec_task"] = None
                        state["answer_open"] = False
                        await ws.send_json({"type": "answer_interrupt"})
                    elif et == "conversation.item.input_audio_transcription.delta":
                        # Observed live (fake_browser_probe): each delta is the
                        # FULL transcript-so-far snapshot, not an increment --
                        # appending produced "你你还记得你还记得我的..." queries.
                        snap = event.get("delta") or ""
                        if snap:
                            state["asr_accum"] = snap
                        await ws.send_json({"type": "partial_transcript",
                                            "text": state["asr_accum"], "replace": True})
                        if (state["spec_task"] is None
                                and len(state["asr_accum"]) >= SPECULATIVE_SEARCH_MIN_CHARS):
                            # Inject as soon as the search lands (usually
                            # mid-utterance) -- in duplex the server won't
                            # wait for us, so early beats complete.
                            state["spec_task"] = asyncio.create_task(self._inject_memory(
                                ws, memory, db, state["asr_accum"], state,
                                stage="interim"))
                            state["spec_task"].add_done_callback(_log_task_exc)
                    elif et == "conversation.item.input_audio_transcription.completed":
                        transcript = (event.get("transcript") or event.get("text")
                                      or state["asr_accum"])
                        state["utt_active"] = False
                        state["t_final"] = time.monotonic()
                        state["await_first_audio"] = True
                        print(f"[turn] final(server判停) {len(transcript)}字", flush=True)
                        if transcript:
                            await ws.send_json({"type": "user_transcript", "text": transcript})
                            self._spawn_ingest(memory, transcript,
                                               bytes(utt_pcm) or None, state)
                            utt_pcm.clear()
                            if not state["injected"] and state["spec_task"] is None:
                                task = asyncio.create_task(self._inject_memory(
                                    ws, memory, db, transcript, state, stage="final"))
                                task.add_done_callback(_log_task_exc)
                    elif et == "conversation.item.input_audio_transcription.failed":
                        print(f"[asr] transcription failed: "
                              f"{json.dumps(event, ensure_ascii=False)[:300]}", flush=True)
                    elif et == "response.output_text.delta":
                        if not state["answer_open"]:
                            state["answer_open"] = True
                            await ws.send_json({"type": "answer_start"})
                        delta = event.get("delta") or ""
                        if delta:
                            await ws.send_json({"type": "answer_delta", "text": delta})
                    elif et == "response.output_audio.started":
                        print(f"[tts] start tts_type={event.get('tts_type')}", flush=True)
                        if not state["answer_open"]:
                            state["answer_open"] = True
                            await ws.send_json({"type": "answer_start"})
                    elif et == "response.output_audio.done":
                        if state["answer_open"]:
                            state["answer_open"] = False
                            await ws.send_json({"type": "answer_done"})
                    elif et == "response.function_call_arguments.done":
                        # The model paused THIS turn's generation to ask for
                        # memory -- answer fast so it can continue speaking.
                        task = asyncio.create_task(self._answer_function_call(
                            ws, memory, db, event, state))
                        task.add_done_callback(_log_task_exc)
                    elif et == "response.canceled":
                        # The model aborted its own response (duplex
                        # decision); the flush already happened at speech
                        # onset -- just close the bubble bookkeeping.
                        state["answer_open"] = False
                    elif et in ("session.created", "session.updated"):
                        print(f"[doubao] {et} id={(event.get('session') or {}).get('id')}",
                              flush=True)
                    elif et == "session.closed":
                        print("[doubao] session.closed", flush=True)
                        return
                    elif et == "error":
                        print(f"[doubao] ERROR: {json.dumps(event, ensure_ascii=False)[:400]}",
                              flush=True)
                        await ws.send_json({"type": "error",
                                            "message": str(event.get("error") or event)})
                    elif et in ("input_audio_buffer.committed", "response.done",
                                "response.output_text.done", "conversation.item.added"):
                        pass
                    else:
                        print(f"[doubao] event {et}: "
                              f"{json.dumps(event, ensure_ascii=False)[:200]}", flush=True)

            try:
                await asyncio.gather(browser_to_doubao(), doubao_to_browser())
            finally:
                # Best-effort polite teardown; the socket may already be gone.
                try:
                    await self._send(db, {"type": "session.close",
                                          "event_id": self._event_id()})
                except Exception:
                    pass

    async def _build_portrait_instructions(self, memory: MemoryBridge) -> str:
        """Standing user portrait baked into the session's base instructions.

        Live-tested race finding ([inject] timing vs actual replies): 豆包全双工
        freezes a turn's context at/before speech onset -- a session.update
        sent mid-utterance takes effect only for the NEXT turn, even when it
        lands well before 判停. Per-turn injection alone therefore always runs
        one turn late; the fix is a standing portrait present from turn 1,
        with per-turn injection kept as enrichment for follow-up turns.
        """
        t0 = time.monotonic()
        seen: set[str] = set()
        items: list[str] = []
        try:
            for q in self._PORTRAIT_QUERIES:
                result = await memory.search(query=q, top_k=5)
                for h in result.hits:
                    text = h.text.strip()
                    if text and text not in seen:
                        seen.add(text)
                        items.append(text)
        except Exception:
            traceback.print_exc()
        if not items:
            return self._config.system_prompt
        body: list[str] = []
        total = 0
        for text in items:
            total += len(text)
            if total > 1600:
                break
            body.append(f"- {text}")
        block = ("\n\n你早就了解的用户画像(常驻背景。像对老朋友一样自然地用,"
                 "回答时挑相关的一两条融进去,绝不要成段背出来):\n" + "\n".join(body))
        print(f"[portrait] {len(body)} facts, {len(block)}字, "
              f"{round((time.monotonic() - t0) * 1000)}ms", flush=True)
        return self._config.system_prompt + block

    @staticmethod
    def _format_recall(result) -> str:
        """SearchResult -> the tool-return text the model continues from."""
        lines = ["你记得这些相关的事:"]
        lines.extend(f"- {h.text}" for h in result.hits)
        lines.extend(f"- {h.content}"
                     for h in (getattr(result, "rb_hits", None) or [])[:2])
        return "\n".join(lines)[:900]

    async def _answer_function_call(self, ws: WebSocket, memory: MemoryBridge, db,
                                     event: dict, state: dict) -> None:
        """Answer a recall_memory call from the model, preferring the search
        that the speculative mid-utterance injection ALREADY ran this turn
        (cached in state) so the model's pause costs ~0ms of our time; fall
        back to a fresh search with the model's own distilled query."""
        t0 = time.monotonic()
        items_out = []
        source = "cached"
        # Seed has emitted both shapes in different realtime deployments:
        #
        #   {"items": [{"name": ..., "call_id": ..., "arguments": ...}]}
        #   {"name": ..., "call_id": ..., "arguments": ...}
        #
        # The old handler only accepted the first one.  In the second case
        # it silently sent no tool output, so the model stayed paused after
        # the FC and the browser appeared to have no current-turn memory.
        raw_items = event.get("items")
        if raw_items is None:
            raw_item = event.get("item")
            if isinstance(raw_item, dict):
                raw_items = [raw_item]
            elif event.get("name") or event.get("call_id"):
                raw_items = [event]
            else:
                raw_items = []

        for it in raw_items:
            if not isinstance(it, dict):
                continue
            call_id = it.get("call_id")
            name = it.get("name")
            if name != "recall_memory":
                items_out.append({"type": "message", "role": "tool", "call_id": call_id,
                                  "content": [{"type": "input_text",
                                               "text": f"{name} 工具不存在"}]})
                continue
            result = state.get("last_result")
            if result is None:
                try:
                    args = json.loads(it.get("arguments") or "{}")
                except (json.JSONDecodeError, TypeError):
                    args = {}
                query = (args.get("query") or "").strip() or state["asr_accum"]
                result = await memory.search(query=query, top_k=5)
                state["last_result"] = result
                source = "fresh"
                await ws.send_json({"type": "memory_hits",
                                    **memory.search_payload(result)})
            items_out.append({"type": "message", "role": "tool", "call_id": call_id,
                              "content": [{"type": "input_text",
                                           "text": self._format_recall(result)}]})
        if items_out:
            await self._send(db, {"type": "conversation.item.create",
                                  "event_id": self._event_id(),
                                  "items": items_out})
        else:
            print(f"[fc] ignored event without callable item: "
                  f"{json.dumps(event, ensure_ascii=False)[:500]}", flush=True)
        print(f"[fc] recall answered ({source}) "
              f"{round((time.monotonic() - t0) * 1000)}ms", flush=True)

    async def _inject_memory(self, ws: WebSocket, memory: MemoryBridge, db,
                              query: str, state: dict, stage: str) -> None:
        """Search memory and push it into the session `instructions` via
        session.update, layered ON TOP of the standing portrait (an update
        replaces instructions wholesale, so building from the bare system
        prompt here would silently wipe the portrait after turn 1). Note the
        update only reaches the model on its NEXT turn (see
        _build_portrait_instructions); it enriches follow-ups, it does not
        win the current turn. `stage` is diagnostic only: interim
        (mid-utterance, the normal case), final (short-utterance fallback),
        typed."""
        t0 = time.monotonic()
        result = await memory.search(query=query, top_k=5)
        search_ms = round((time.monotonic() - t0) * 1000)
        state["injected"] = True
        state["last_result"] = result  # recall_memory FC answers from this
        await ws.send_json({"type": "memory_hits", **memory.search_payload(result)})
        instructions = memory.build_instructions(self._base_instructions, result)
        await self._send(db, {"type": "session.update",
                              "event_id": self._event_id(),
                              "session": {"instructions": instructions}})
        print(f"[inject] {stage} search {search_ms}ms query={len(query)}字 "
              f"-> instructions {len(instructions)}字", flush=True)

    def _spawn_ingest(self, memory: MemoryBridge, transcript: str, pcm: bytes | None,
                       state: dict) -> None:
        state["turn_counter"] += 1
        turn_id = f"{state['turn_counter']}-{uuid.uuid4().hex[:8]}"
        task = asyncio.create_task(self._background_ingest(
            memory, transcript, pcm, turn_id, state["turn_counter"]))
        task.add_done_callback(_log_task_exc)

    @staticmethod
    async def _background_ingest(memory: MemoryBridge, transcript: str, pcm: bytes | None,
                                   turn_id: str, session_id: int) -> None:
        try:
            await memory.ingest(transcript, pcm=pcm, turn_id=turn_id, speaker="user",
                                 session_id=session_id)
        except Exception:
            traceback.print_exc()
