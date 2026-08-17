"""Opt-in mode: browser audio -> OpenAI Realtime API, FULL DUPLEX: the mic
PCM itself streams into the Realtime session, so the model HEARS the user
(tone of voice included) instead of reading a transcript. A LOCAL streaming
ASR runs in parallel as the control plane: live captions, the memory-search
query, barge-in confirmation, and Ingest()'s transcript all still come from
it. Uses `client.realtime` (the current, nested-schema namespace) rather
than the older, deprecated `client.beta.realtime` -- verified directly
against the installed SDK's type definitions (`openai/types/realtime/*.py`),
not assumed from docs.

Turn-taking: server-side SEMANTIC VAD with `create_response: false`. It
judges whether the sentence sounds FINISHED, so a complete sentence turns
around faster than any fixed silence window, and a mid-thought hesitation
is given longer -- the thing a fixed window can't do (the old fixed local
window measured 0.2-0.4s all broke mid-sentence; see the LocalASR note
below). `create_response: false` keeps response timing in OUR hands: on
each server commit we inject fresh `voicemem.Search()` context into
`instructions` via session.update, THEN call response.create() -- the same
per-turn memory guarantee the previous text-in design had.

Input-side history, kept because the failure modes are real: the FIRST
audio-in version of this file let cloud transcription + server VAD drive
turns, and user testing surfaced hallucinated turns (breathing transcribed
as characteristically Korean text, a known Whisper-family failure mode), no
live captions, and ~1.2-1.5s end-of-speech-to-transcript latency. The file
then retreated to text-in (local ASR, `input_text` items). This full-duplex
revival does NOT re-enable the hallucination-prone component: cloud
transcription stays OFF, the model hears audio, and every piece of TEXT in
the system still comes from the local zipformer (solid Chinese, weak
English -- see local_asr.py; caption/memory-query quality is still bound by
it, but the MODEL's understanding of the user no longer is). If the server
never commits an utterance the local ASR clearly heard (audio path broken,
too quiet for its VAD), a grace-period fallback re-sends the local
transcript as a text item -- the reply degrades to the proven text-in path
instead of never coming. A commit with NO local transcript and only a short
burst of server-heard speech is dropped as noise (echo tail, chair creak)
rather than answered (knob: REALTIME_NOISE_GUARD_MS).

Barge-in is the same two-stage flow as pipeline mode (see that module's
docstring): speech onset -> reversible speech_tentative pause; enough real
transcript mid-utterance -> answer_interrupt (cancels the active Realtime
response AND tells the client to flush); resolves-to-noise -> answer_resume.
Signals are not gated on a response being active server-side -- the client
may still be playing buffered audio after the response finished, and only
it knows (real bug found via user testing in pipeline mode; same reasoning
applies unchanged here). `interrupt_response` stays off so the server never
fights this local logic.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import time
import traceback
import uuid

from fastapi import WebSocket, WebSocketDisconnect
from openai import AsyncOpenAI

from config import Config
from local_asr import LocalASR
from memory_bridge import MemoryBridge

AUDIO_FORMAT = {"type": "audio/pcm", "rate": 24000}  # input AND output; only 24000 is valid per the SDK
# Fire memory.search() speculatively on the live partial transcript once it
# looks long enough to be worth a real search -- by turn end it's usually
# already done. Once per turn. (Same constant + reasoning as pipeline.py.)
SPECULATIVE_SEARCH_MIN_CHARS = 15
# Require a few real transcribed characters before treating speech as an
# actual interruption (keyboard clicks can mis-transcribe as a stray char or
# two). 3, not 4: common Chinese interruption phrases are exactly 3 chars
# ("等一下"/"别说了") -- same value + reasoning as pipeline.py.
MIN_INTERRUPT_CHARS = 3
# Semantic-VAD eagerness: how aggressively the server decides "the user is
# done". "high" turns a finished sentence around faster than the old fixed
# 500ms window ever could; "auto"/"low" hold longer on hesitations. Exposed
# because this is exactly the responsiveness-vs-cutting-people-off dial.
SEMANTIC_VAD_EAGERNESS = os.environ.get("SEMANTIC_VAD_EAGERNESS", "high")
# A server commit with NO local transcript and less than this much
# server-heard speech is treated as noise: no reply, and the audio item is
# deleted so the junk doesn't linger in the conversation context.
NOISE_GUARD_MS = int(os.environ.get("REALTIME_NOISE_GUARD_MS", "600"))
# If the server never commits an utterance the local ASR clearly heard, fall
# back to the proven text-in path after this long. Generous on purpose:
# semantic VAD legitimately holds back longer than the local 0.5s window on
# trailing/hesitant speech, and a fallback firing early would double-reply.
FALLBACK_GRACE_S = float(os.environ.get("REALTIME_FALLBACK_GRACE_S", "2.0"))

# A short, memory-free response is allowed to start as soon as the server
# commits the user's turn.  Retrieval runs in parallel with this acknowledgement;
# the real answer is not created until retrieval has finished and its context has
# been installed.  This improves perceived first-audio latency without giving up
# the guarantee that the substantive answer sees this turn's memories.
ACK_INSTRUCTIONS = (
    "Respond with only a very brief natural acknowledgement in the user's language, "
    "such as '嗯，我想一下。' Do not answer the question, mention memory, or add facts."
)


def _log_task_exc(task: "asyncio.Task") -> None:
    """Done-callback for fire-and-forget tasks whose result nobody awaits."""
    if not task.cancelled() and task.exception() is not None:
        traceback.print_exception(task.exception())


class RealtimeProvider:
    def __init__(self, config: Config) -> None:
        self._config = config
        self._client = AsyncOpenAI()

    async def run_session(self, ws: WebSocket, memory: MemoryBridge) -> None:
        await ws.send_json({"type": "session_ready",
                             "mode": "realtime",
                             "audio_native": self._config.audio_native})

        async with self._client.realtime.connect(model=self._config.openai_realtime_model) as connection:
            await connection.session.update(session={
                "type": "realtime",
                "instructions": self._config.system_prompt,
                "audio": {
                    # Full duplex: mic PCM streams into the session (see
                    # browser_to_openai), so the model hears the actual
                    # voice. Turn detection is server-side SEMANTIC VAD, but
                    # with create_response off: WE create the response after
                    # injecting this turn's memory context (see the
                    # input_audio_buffer.committed handler). No
                    # "transcription" block on purpose -- every piece of text
                    # still comes from the local zipformer (module docstring
                    # has the hallucination history behind that choice).
                    "input": {
                        "format": AUDIO_FORMAT,
                        "turn_detection": {
                            "type": "semantic_vad",
                            "eagerness": SEMANTIC_VAD_EAGERNESS,
                            "create_response": False,
                            # Local two-stage barge-in owns interruption; a
                            # server auto-interrupt would fight it.
                            "interrupt_response": False,
                        },
                    },
                    "output": {
                        "format": AUDIO_FORMAT,
                        "voice": self._config.openai_tts_voice,
                    },
                },
            })

            # Local streaming ASR + VAD (shared recognizer weights with
            # pipeline mode; per-connection stream/VAD state).
            # 0.5s 断句窗已经不在回答关键路径上了——回答时机归服务端语义 VAD
            # 管;本地 turn_end 只负责定格字幕、喂 Ingest()、武装兜底。窗口
            # 曾实测过逐档 0.2/0.3/0.4 全会句中碎裂(Silero 把弱辅音/尾音淡出
            # 连成 400ms+ "静音"),语义断句正是当时结论指向的出路。旋钮保留:
            # REALTIME_TURN_SILENCE_S。
            asr = LocalASR(turn_silence_s=float(os.environ.get("REALTIME_TURN_SILENCE_S", "0.5")))

            local_pcm = bytearray()
            state = {"turn_counter": 0, "spec_partial_text": "", "spec_task": None,
                     "tentative": False,
                     # full-duplex bookkeeping (all per-utterance except the
                     # commit clock): who has finalized the "you:" caption,
                     # local speech onset vs. last server commit (monotonic
                     # clocks, compared by the fallback), the server's own
                     # speech-span measurement (noise guard), and the armed
                     # text-path fallback task.
                     "utt_transcript_sent": True,
                     "speech_start_t": 0.0, "last_commit_t": 0.0,
                     "server_speech_start_ms": None, "server_utt_ms": None,
                     "fallback_task": None,
                     "response_done_waiters": {},
                     "pending_response_waiter": None,
                     "active_response_stage": None}

            async def handle_asr_event(ev: tuple) -> None:
                kind = ev[0]
                if kind == "speech_start":
                    del local_pcm[:]  # fresh per-utterance audio for Ingest()'s archival
                    state["spec_partial_text"] = ""
                    state["spec_task"] = None
                    state["tentative"] = True
                    state["utt_transcript_sent"] = False
                    state["speech_start_t"] = time.monotonic()
                    if state["fallback_task"] is not None:
                        state["fallback_task"].cancel()  # previous utterance's, now stale
                        state["fallback_task"] = None
                    print("[vad] speech_start -> tentative pause sent", flush=True)
                    await ws.send_json({"type": "speech_tentative"})
                elif kind == "partial":
                    full_text = ev[1]
                    # FULL text, replace mode -- greedy decoding can revise
                    # earlier words (see pipeline.py / frontend). UI send is
                    # gated: once the server commit finalized the "you:" line,
                    # straggler partials (the zipformer decodes a beat behind
                    # the audio) would re-create a stale "listening" line
                    # below the reply. They still update spec_partial_text --
                    # turn_end's Ingest() wants the fullest text.
                    if not state["utt_transcript_sent"]:
                        await ws.send_json({"type": "partial_transcript",
                                            "text": full_text, "replace": True})
                    state["spec_partial_text"] = full_text
                    if (state["tentative"]
                            and len(full_text) >= MIN_INTERRUPT_CHARS):
                        state["tentative"] = False  # resolved: real interrupt
                        print(f"[vad] confirmed interrupt ({len(full_text)} chars)", flush=True)
                        if state.get("response_active"):
                            # Cancels server-side AND sends answer_interrupt.
                            await self._cancel_active_response(ws, connection, state)
                        else:
                            # Nothing active server-side, but the client may
                            # still be playing buffered tail audio -- it
                            # decides what the signal means.
                            await ws.send_json({"type": "answer_interrupt"})
                    if (state["spec_task"] is None
                            and len(state["spec_partial_text"]) >= SPECULATIVE_SEARCH_MIN_CHARS):
                        state["spec_task"] = asyncio.create_task(
                            memory.search(query=state["spec_partial_text"], top_k=5))
                elif kind == "turn_end":
                    transcript = ev[1]
                    # NOT popped: the server-commit handler owns consuming
                    # state["spec_task"]; this snapshot only feeds the
                    # fallback (or the empty-turn cancel below).
                    spec_task = state["spec_task"]
                    if state["tentative"]:
                        state["tentative"] = False
                        print("[vad] resolved as noise/short -> resume sent", flush=True)
                        await ws.send_json({"type": "answer_resume"})
                    if transcript:
                        # The reply is normally already in flight off the
                        # server commit (which usually beats this event by up
                        # to the 0.5s window). Here: finalize the caption if
                        # the commit didn't, archive the utterance audio for
                        # Ingest(), and arm the text-path fallback in case
                        # the server never commits at all.
                        committed_already = state["last_commit_t"] >= state["speech_start_t"]
                        print(f"[turn_end] local(+0.5s静音) {len(transcript)}字 "
                              f"{'(commit已先行)' if committed_already else '(commit未到,兜底已武装)'}",
                              flush=True)
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
                        state["fallback_task"] = asyncio.create_task(
                            self._fallback_respond(ws, memory, connection, transcript,
                                                    state, spec_task,
                                                    utterance_start=state["speech_start_t"]))
                        state["fallback_task"].add_done_callback(_log_task_exc)
                    else:
                        local_pcm.clear()
                        state["spec_task"] = None
                        if spec_task is not None and not spec_task.done():
                            spec_task.cancel()  # empty turn -- don't leave it dangling
                # ("silence_hint" stays ignored: reply timing is owned by
                # the server's semantic-VAD commit now, so a 200ms-early
                # local hint has nothing left to accelerate.)

            async def browser_to_openai() -> None:
                # Once-per-second peak level of the incoming mic audio --
                # same operational diagnostic as pipeline.py ([mic] lines),
                # kept for the same reason: it's the only way to tell
                # "silence" apart from "speech arriving too quiet/suppressed".
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
                        # Full duplex: the same 24kHz frame goes to the
                        # Realtime session (base64 per the wire schema).
                        # Continuous on purpose -- semantic VAD needs to hear
                        # the silences too, and only COMMITTED speech becomes
                        # (billed) conversation context; the rest of the
                        # buffer is transient.
                        await connection.input_audio_buffer.append(
                            audio=base64.b64encode(data).decode("ascii"))
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
                            await connection.conversation.item.create(item={
                                "type": "message", "role": "user",
                                "content": [{"type": "input_text", "text": user_text}],
                            })
                            await self._on_user_turn_complete(ws, memory, connection, user_text,
                                                               local_pcm, state)

            async def openai_to_browser() -> None:
                async for event in connection:
                    et = event.type
                    if et == "response.output_audio.delta":
                        # Filter by response_id, not just event order. This
                        # alone is NOT enough to fully solve cross-response
                        # bleed (see the cancel-then-wait logic below for why),
                        # but it's still correct defense in depth.
                        if event.response_id == state.get("active_response_id"):
                            if state.pop("await_first_audio", False):
                                now = time.monotonic()
                                commit_t = state.get("last_commit_t") or 0.0
                                create_t = state.get("respond_t0") or now
                                print(f"[ttfb] commit→首音 {round((now - commit_t) * 1000)}ms"
                                      f" (create→首音 {round((now - create_t) * 1000)}ms)",
                                      flush=True)
                            await ws.send_bytes(base64.b64decode(event.delta))
                    elif et == "response.output_audio_transcript.delta":
                        if (event.response_id == state.get("active_response_id")
                                and state.get("active_response_stage") != "ack"):
                            await ws.send_json({"type": "answer_delta", "text": event.delta})
                    elif et == "response.created":
                        state["response_active"] = True
                        state["active_response_id"] = event.response.id
                        pending_waiter = state.pop("pending_response_waiter", None)
                        if pending_waiter is not None:
                            state["response_done_waiters"][event.response.id] = pending_waiter
                        # Real bug found via live browser testing: this file
                        # never sent "answer_start" at all -- the frontend's
                        # answer_delta handler only appends text to a line
                        # created by answer_start, so replies were spoken but
                        # never appeared as text. response.created fires
                        # exactly once per response, before any content
                        # deltas -- the correct single place to signal this.
                        if state.get("active_response_stage") != "ack":
                            await ws.send_json({"type": "answer_start"})
                    elif et == "response.done":
                        # Resolve anyone in _cancel_active_response() blocked
                        # waiting for confirmation that THIS response actually
                        # finished closing out server-side.
                        waiter = state.pop(f"cancel_wait_{event.response.id}", None)
                        if waiter is not None:
                            waiter.set()
                        done_waiter = state.get("response_done_waiters", {}).pop(
                            event.response.id, None)
                        if done_waiter is not None and not done_waiter.done():
                            done_waiter.set_result(event.response.status != "cancelled")
                        # A stale "done" for a response we've already superseded
                        # (id no longer matches) -- state was already updated
                        # when the newer response.created arrived, nothing else to do.
                        if event.response.id == state.get("active_response_id"):
                            state["response_active"] = False
                            state["active_response_id"] = None
                            # A cancelled response already got answer_interrupt
                            # (or the client never heard anything from it) --
                            # sending answer_done for it too would be a spurious
                            # "turn finished" signal for a turn that was cut off.
                            if (event.response.status != "cancelled"
                                    and state.get("active_response_stage") != "ack"):
                                await ws.send_json({"type": "answer_done"})
                    elif et == "input_audio_buffer.speech_started":
                        state["server_speech_start_ms"] = event.audio_start_ms
                    elif et == "input_audio_buffer.speech_stopped":
                        start_ms = state.get("server_speech_start_ms")
                        state["server_utt_ms"] = (event.audio_end_ms - start_ms
                                                   if start_ms is not None else None)
                    elif et == "input_audio_buffer.committed":
                        # Server semantic VAD says: the user is done. THIS is
                        # the reply trigger now, not local turn_end -- it
                        # fires as soon as the sentence sounds finished
                        # instead of after a fixed 500ms of silence.
                        state["last_commit_t"] = time.monotonic()
                        # Timing visibility (real session showed none): this
                        # line printing BEFORE the local [turn_end] line =
                        # semantic VAD beat the old fixed 500ms window.
                        print(f"[commit] server断句 utt={state.get('server_utt_ms')}ms "
                              f"local_partial={len(state['spec_partial_text'])}字", flush=True)
                        if state["fallback_task"] is not None:
                            state["fallback_task"].cancel()
                            state["fallback_task"] = None
                        query = state["spec_partial_text"]
                        dur = state.get("server_utt_ms")
                        if not query and dur is not None and dur < NOISE_GUARD_MS:
                            # Nothing transcribed locally AND barely any
                            # server-heard speech: echo tail / chair creak.
                            # Delete the item too, or the junk audio would
                            # sit in the conversation as context.
                            print(f"[commit] dropped as noise ({dur}ms, no local text)", flush=True)
                            await connection.conversation.item.delete(item_id=event.item_id)
                        else:
                            if not state["utt_transcript_sent"] and query:
                                # Finalize the caption NOW so the "you:" line
                                # lands before the reply bubble (local
                                # turn_end is up to 500ms behind us; Ingest()
                                # still gets its slightly-fuller final text).
                                await ws.send_json({"type": "user_transcript", "text": query})
                                state["utt_transcript_sent"] = True
                            spec_task = state["spec_task"]
                            state["spec_task"] = None
                            # Off the event loop's critical path: this awaits
                            # memory.search() when the speculative task missed,
                            # and audio deltas must keep flowing meanwhile.
                            task = asyncio.create_task(self._respond_with_memory(
                                ws, memory, connection, query, state, spec_task))
                            task.add_done_callback(_log_task_exc)
                    elif et == "error":
                        # response.create()/cancel() are fire-and-forget sends
                        # (the SDK doesn't raise on rejection) -- rejections
                        # come back later as this async event instead, so
                        # recovery has to happen here, not at the call site.
                        code = getattr(event.error, "code", None)
                        if code == "response_cancel_not_active":
                            # Harmless, expected race: our response_active
                            # heuristic said "cancel this" but the response had
                            # already finished server-side by the time the
                            # cancel arrived. SDK docs: "safe to call... an
                            # error will be returned [but] the session will
                            # remain unaffected." Nothing to recover from.
                            pass
                        elif code == "conversation_already_has_active_response":
                            # Opposite race: our heuristic said "nothing active"
                            # but the server still had a response in flight, so
                            # response.create() for this turn was rejected --
                            # without recovery, that turn's reply is silently
                            # dropped. Cancel whatever's actually active, wait
                            # for it to actually finish closing out server-side,
                            # then retry; the session's instructions were
                            # already updated with this turn's memory context
                            # before the failed create(), so a bare retry (no
                            # need to repeat session.update) is enough.
                            await self._cancel_active_response(ws, connection, state)
                            await connection.response.create()
                        else:
                            await ws.send_json({"type": "error", "message": str(event.error)})

            await asyncio.gather(browser_to_openai(), openai_to_browser())

    async def _on_user_turn_complete(self, ws: WebSocket, memory: MemoryBridge, connection,
                                      transcript: str, local_pcm: bytearray, state: dict,
                                      spec_task: "asyncio.Task | None" = None) -> None:
        """Typed-text turns only -- voice turns are driven by the server
        commit (reply) + local turn_end (caption/ingest/fallback) pair now."""
        await ws.send_json({"type": "user_transcript", "text": transcript})

        pcm = bytes(local_pcm)
        local_pcm.clear()
        state["turn_counter"] += 1
        turn_id = f"{state['turn_counter']}-{uuid.uuid4().hex[:8]}"
        asyncio.create_task(self._background_ingest(memory, transcript, pcm if pcm else None,
                                                      turn_id, state["turn_counter"]))

        await self._respond_with_memory(ws, memory, connection, transcript, state, spec_task)

    async def _respond_with_memory(self, ws: WebSocket, memory: MemoryBridge, connection,
                                    query: str, state: dict,
                                    spec_task: "asyncio.Task | None" = None) -> None:
        """Search memory (reusing the speculative task fired mid-utterance
        when there is one -- see SPECULATIVE_SEARCH_MIN_CHARS), inject the
        result into `instructions`, then create the response. `query` may be
        "" (the zipformer heard nothing usable, e.g. English it whiffed on):
        the model still heard the audio itself, so reply anyway -- just on
        the previous instructions, without fresh memory context this turn."""
        # Do not let a stale response occupy the Realtime session while we
        # start the two branches for this turn.
        if state.get("response_active"):
            await self._cancel_active_response(ws, connection, state)

        t0 = time.monotonic()
        if spec_task is not None:
            search_task = spec_task
        elif query:
            search_task = asyncio.create_task(memory.search(query=query, top_k=5))
        else:
            search_task = None

        # Branch A: start an audio acknowledgement immediately.  Branch B is
        # the search task above.  The acknowledgement is deliberately not
        # sent to the browser UI; only its audio is used as the perceived
        # first response.
        ack_task = asyncio.create_task(self._play_memory_ack(connection, state))
        result = None
        if search_task is not None:
            try:
                result = await search_task
            except asyncio.CancelledError:
                result = await memory.search(query=query, top_k=5) if query else None
        search_ms = round((time.monotonic() - t0) * 1000)
        ack_ok = await ack_task
        ack_ms = round((time.monotonic() - t0) * 1000) - search_ms
        # >50ms means the search ran on the reply's critical path (the
        # speculative task missed or wasn't fired); same diagnostic the
        # service/ stack prints as "prefetch 命中/未命中".
        print(f"[respond] search={'spec命中' if search_ms <= 50 else '关键路径'} {search_ms}ms "
              f"ack={ack_ms}ms query={len(query)}字", flush=True)
        if result is not None:
            await ws.send_json({"type": "memory_hits", **memory.search_payload(result)})
            instructions = memory.build_instructions(self._config.system_prompt, result)
            await connection.session.update(session={"type": "realtime", "instructions": instructions})

        if not ack_ok:
            return  # the user interrupted the acknowledgement
        state["respond_t0"] = time.monotonic()
        state["await_first_audio"] = True
        state["active_response_stage"] = "answer"
        await connection.response.create()

    @staticmethod
    async def _play_memory_ack(connection, state: dict) -> bool:
        """Play a tiny first-audio acknowledgement while memory search runs.

        The future is registered before response.create() is sent because the
        server can return response.created immediately.  The caller only
        proceeds to the substantive response after this response is done.
        """
        loop = asyncio.get_running_loop()
        done = loop.create_future()
        state["pending_response_waiter"] = done
        state["active_response_stage"] = "ack"
        try:
            await connection.response.create(response={
                "instructions": ACK_INSTRUCTIONS,
                "max_output_tokens": 24,
                "metadata": {"voicemem_stage": "ack"},
            })
            return bool(await asyncio.wait_for(done, timeout=5.0))
        except (asyncio.CancelledError, asyncio.TimeoutError):
            state.pop("pending_response_waiter", None)
            return False

    async def _fallback_respond(self, ws: WebSocket, memory: MemoryBridge, connection,
                                 transcript: str, state: dict,
                                 spec_task: "asyncio.Task | None",
                                 utterance_start: float) -> None:
        """Belt and suspenders: the local ASR clearly heard an utterance but
        the server never committed one (audio path broken, or too quiet for
        its VAD). After the grace period -- generous, because semantic VAD
        legitimately holds longer than the local 0.5s window on hesitant
        speech -- re-send the local transcript as a text item, the proven old
        path, so the turn still gets answered. Known rare edge: a commit
        landing after the grace expired would produce a second response; the
        pre-create safety cancel makes the newer one supersede the older
        rather than overlap it."""
        await asyncio.sleep(FALLBACK_GRACE_S)
        if state["last_commit_t"] >= utterance_start:
            return  # the server did commit (this utterance or a newer one)
        print("[fallback] no server commit for a heard utterance -> text path", flush=True)
        await connection.conversation.item.create(item={
            "type": "message", "role": "user",
            "content": [{"type": "input_text", "text": transcript}],
        })
        await self._respond_with_memory(ws, memory, connection, transcript, state, spec_task)

    @staticmethod
    async def _cancel_active_response(ws: WebSocket, connection, state: dict) -> None:
        """Cancel whatever response is currently active, wait for it to
        actually finish closing out server-side, and tell the client to flush
        anything it may have already started receiving for it.

        The wait matters: reproduced for real (via an overlapping-turns stress
        test) that calling response.create() for the next turn immediately
        after cancel() -- without waiting -- lets the server keep emitting a
        few more transcript/audio deltas *still stamped with the OLD response's
        id* for content that is actually the NEW turn's answer, right up until
        the old response's own response.done(cancelled) finally arrives.

        KNOWN REMAINING LIMITATION (disclosed, not silently swept under the
        rug): waiting for confirmed cancellation greatly reduces but does not
        100% eliminate the issue -- in stress testing (two text turns fired
        ~0.3s apart, repeatedly), occasionally the retried response still
        references or partially repeats the cancelled one's fragment. This
        looks like server-side behavior (the cancelled item may still be
        visible in conversation context when the retry is created) rather than
        a client-side ordering bug we can fully close from here.

        Shared by every place a stale/still-active response gets discovered:
        real barge-in (confirmed interrupt), the pre-create safety check, and
        the conversation_already_has_active_response error-recovery retry.
        """
        cancelled_id = state.get("active_response_id")
        await connection.response.cancel()
        state["response_active"] = False
        await ws.send_json({"type": "answer_interrupt"})

        if cancelled_id:
            done_event = asyncio.Event()
            state[f"cancel_wait_{cancelled_id}"] = done_event
            try:
                await asyncio.wait_for(done_event.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                # Give up waiting rather than hang the turn forever -- worst
                # case we're back to the original (rare) race, not a deadlock.
                state.pop(f"cancel_wait_{cancelled_id}", None)

    @staticmethod
    async def _background_ingest(memory: MemoryBridge, transcript: str, pcm: bytes | None,
                                   turn_id: str, session_id: int) -> None:
        try:
            await memory.ingest(transcript, pcm=pcm, turn_id=turn_id, speaker="user",
                                 session_id=session_id)
        except Exception:
            traceback.print_exc()  # this task's return value is never awaited/checked anywhere else
