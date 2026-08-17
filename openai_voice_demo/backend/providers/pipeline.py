"""Default mode: continuous conversation, no push-to-talk. LOCAL streaming
ASR + VAD (sherpa-onnx, see local_asr.py) for input -> voicemem.Search() ->
streamed GPT-4o chat completion -> sentence-chunked OpenAI TTS ->
voicemem.Ingest() (background).

Input-side history, in order, each step driven by real user testing:
(1) local Silero VAD + push-to-talk + whisper-1 REST per turn -- no live
captions at all (whisper-1 has no streaming variant), held-button UX;
(2) OpenAI Realtime *transcription-only* session -- continuous listening
and streaming deltas in principle, but two real measured problems: for
continuous speech the API released ALL deltas in a burst only after the
utterance ended (no live captions in practice, confirmed via isolated A/B
against semantic_vad too), and end-of-speech-to-caption latency was
~1.2-1.5s at best (VAD wait + non-tunable server-side finalization), which
the user judged the single worst latency in the whole loop;
(3) NOW: local sherpa-onnx streaming zipformer + Silero VAD (the same
stack, and the same already-downloaded model files, as service/) --
genuinely live word-by-word captions while the user is still talking,
near-zero finalization at turn end, and barge-in confirmable mid-utterance.
Real, disclosed trade-offs: transcript accuracy below the cloud model
(greedy zipformer: English comes out UPPERCASE, no punctuation -- GPT-4o
copes fine, stored memories inherit the rougher text), and ASR now costs
local CPU. providers/realtime.py still uses OpenAI's cloud stack
end-to-end; only this pipeline mode's input side is local.

Turns are processed strictly sequentially through _turn_queue/_turn_worker.

Barge-in is two-stage: (1) the moment speech starts, speech_tentative tells
the client to PAUSE playback -- reversible, safe on that weak signal;
(2) once >= MIN_INTERRUPT_CHARS of real transcript accumulate,
answer_interrupt makes it permanent (client flushes buffered audio); if the
speech resolves to nothing real (noise/cough), answer_resume un-pauses.
Signals are deliberately NOT gated on the server-side turn task being alive
-- TTS delivers audio much faster than real-time, so the browser is usually
still playing buffered audio long after the task finished; only the client
knows whether it's still audibly playing, so the server always sends and
the client decides (real bug found via user testing; scripted repros that
always interrupted at reply start kept passing right through it).
"""
from __future__ import annotations

import asyncio
import json
import re
import time
import traceback
import uuid

from fastapi import WebSocket, WebSocketDisconnect
from openai import AsyncOpenAI

from config import Config
from local_asr import LocalASR
from memory_bridge import MemoryBridge
# Real, measured bottleneck (this session's own e2e_latency_test.py):
# memory.search() -- Classify() (an LLM call) + Search() (an embedding-API
# call) -- costs ~1.5-2s on its own, and normally sits entirely BEFORE the
# reply can start. Since transcription now streams in live (deltas arrive
# WHILE the user is still talking, well before they finish), that 1.5-2s can
# be spent concurrently with the rest of their sentence instead of after it:
# once the partial transcript looks long enough to be worth a real search,
# fire memory.search() speculatively on the text-so-far; by the time
# transcription.completed arrives, that search is often already done. Only
# fires once per turn (not on every delta) to avoid stacking up redundant
# Classify()+Search() calls while one utterance is still streaming in.
SPECULATIVE_SEARCH_MIN_CHARS = 15
# Real bug found via live testing: gating barge-in on "any transcribed
# delta at all" was still too sensitive -- keyboard clicks and other short
# percussive noise sometimes get mis-transcribed as a stray character or
# two, which was enough to cancel the assistant's answer. Require a few real
# characters before treating it as an actual interruption; a genuine
# interruption keeps growing past this within one more delta or two anyway.
# Was 4 under the cloud ASR; 3 now: common Chinese interruption phrases are
# exactly 3 characters ("等一下"/"别说了"/"停一下"), and 4 made real test
# audio ("礼拜二") resolve as noise. The local Silero VAD is also less prone
# to triggering on keyboard clicks than the cloud VAD was (speech-specific
# model), so the noise-mis-transcription risk this constant guards against
# is smaller than when 4 was picked.
MIN_INTERRUPT_CHARS = 3


class PipelineProvider:
    def __init__(self, config: Config) -> None:
        self._config = config
        self._client = AsyncOpenAI()  # reads OPENAI_API_KEY / OPENAI_BASE_URL from env
        self._history: list[dict] = []
        self._turn_counter = 0

    async def run_session(self, ws: WebSocket, memory: MemoryBridge) -> None:
        await ws.send_json({"type": "session_ready",
                             "mode": "pipeline",
                             "audio_native": self._config.audio_native})

        turn_queue: asyncio.Queue[tuple[str, bytes | None, "asyncio.Task | None"]] = asyncio.Queue()
        turn_pcm = bytearray()  # best-effort per-turn audio, for Ingest()'s audio_native archival
        state = {"current_task": None}  # the in-flight _process_turn() Task, for barge-in cancellation
        # partial_text/task: speculative memory.search(), see SPECULATIVE_SEARCH_MIN_CHARS.
        # tentative: a speech_tentative pause was sent to the client and not yet
        # resolved into either a real interrupt (answer_interrupt) or a resume
        # (answer_resume) -- see the barge-in comments in openai_to_browser.
        spec = {"partial_text": "", "task": None, "tentative": False}

        async def turn_worker() -> None:
            while True:
                transcript, pcm, spec_task = await turn_queue.get()
                task = asyncio.create_task(self._process_turn(ws, memory, transcript, pcm, spec_task))
                state["current_task"] = task
                try:
                    await task
                except asyncio.CancelledError:
                    pass  # barge-in cancelled this turn -- see speech_started handling below
                except Exception:
                    traceback.print_exc()
                finally:
                    if state["current_task"] is task:
                        state["current_task"] = None

        worker_task = asyncio.create_task(turn_worker())

        try:
            # Local streaming ASR + VAD -- see local_asr.py's module
            # docstring for why this replaced the OpenAI Realtime
            # transcription session (real measured caption latency + no
            # live partials for continuous speech), and what the accuracy
            # trade-off is. First construction loads the zipformer weights
            # (~1-3s, then cached process-wide).
            asr = LocalASR()

            # Once-per-second peak level of the incoming mic audio.
            # Operational diagnostic, deliberately kept: "interrupt doesn't
            # work" investigations keep needing to know whether the user's
            # speech is REACHING the server at meaningful volume (browser
            # echo cancellation suppresses double-talk audio -- speech over
            # a playing reply can arrive heavily attenuated), and silence
            # and suppressed speech both produce no ASR output, so this is
            # the only way to tell them apart.
            level_window_max = 0
            level_window_start = 0.0

            async def handle_asr_event(ev: tuple) -> None:
                kind = ev[0]
                if kind == "speech_start":
                    del turn_pcm[:]  # start collecting fresh raw audio for this turn
                    spec["partial_text"] = ""
                    spec["task"] = None
                    spec["early"] = None
                    # Tentative pause: reversible, so safe to fire on bare
                    # VAD onset -- with the LOCAL VAD this now happens
                    # ~100-200ms after the user opens their mouth (the cloud
                    # session took ~700ms+). See the module docstring for
                    # the two-stage barge-in design and why the signals are
                    # not gated on the server-side task being alive.
                    spec["tentative"] = True
                    print("[vad] speech_start -> tentative pause sent", flush=True)
                    await ws.send_json({"type": "speech_tentative"})
                elif kind == "partial":
                    full_text = ev[1]
                    # FULL text, replace mode -- greedy decoding can revise
                    # earlier words, so append-style deltas would garble the
                    # caption (the frontend replaces the line content when
                    # `replace` is set).
                    await ws.send_json({"type": "partial_transcript",
                                        "text": full_text, "replace": True})
                    spec["partial_text"] = full_text
                    # The user resumed talking after an early-committed
                    # speculative turn (see silence_hint below) and the
                    # transcript has grown past what that turn is answering
                    # -- kill it, this utterance isn't over. Safe to do
                    # bluntly: the reply pipeline's first audio is ~1.5s+
                    # out, so a speculative turn cancelled this early was
                    # never audible.
                    if spec.get("early") is not None and full_text != spec["early"]:
                        print("[vad] speech resumed past early commit -> speculative turn cancelled", flush=True)
                        spec["early"] = None
                        task = state["current_task"]
                        if task is not None and not task.done():
                            task.cancel()
                    if (spec["tentative"]
                            and len(full_text) >= MIN_INTERRUPT_CHARS):
                        spec["tentative"] = False  # resolved: real interrupt
                        task = state["current_task"]
                        if task is not None and not task.done():
                            task.cancel()
                        print(f"[vad] confirmed interrupt ({len(full_text)} chars)", flush=True)
                        await ws.send_json({"type": "answer_interrupt"})
                    # Speculative memory.search() -- see the module-level
                    # comment on SPECULATIVE_SEARCH_MIN_CHARS. Once per
                    # turn: with live local partials this now fires while
                    # the user is still mid-sentence on essentially every
                    # turn, not just long ones.
                    if spec["task"] is None and len(spec["partial_text"]) >= SPECULATIVE_SEARCH_MIN_CHARS:
                        spec["task"] = asyncio.create_task(
                            memory.search(query=spec["partial_text"], top_k=5))
                elif kind == "silence_hint":
                    # ~200ms of silence: the user is PROBABLY done (the
                    # formal turn close needs 500ms). Start the reply
                    # pipeline now on the transcript-so-far instead of
                    # sitting out the remaining ~300ms -- speculative, and
                    # safe for the same reason the early-cancel above is:
                    # nothing is audible for the first ~1.5s of a turn, so
                    # a wrong guess cancelled at turn_end (or on resumed
                    # speech) never reaches the user's ears. Costs a few
                    # wasted LLM tokens on a wrong guess, nothing else.
                    hint_text = ev[1]
                    if hint_text and spec.get("early") is None:
                        spec["early"] = hint_text
                        pcm = bytes(turn_pcm) if turn_pcm else None
                        spec_task = spec["task"]
                        spec["task"] = None
                        print(f"[vad] silence hint -> speculative turn started ({len(hint_text)} chars)", flush=True)
                        await turn_queue.put((hint_text, pcm, spec_task))
                elif kind == "turn_end":
                    transcript = ev[1]
                    pcm = bytes(turn_pcm) if turn_pcm else None
                    del turn_pcm[:]
                    spec_task = spec["task"]
                    spec["task"] = None
                    # Tentative still unresolved here means the utterance
                    # never accumulated MIN_INTERRUPT_CHARS of transcript
                    # (noise/cough/too short) -- un-pause the still-playing
                    # reply.
                    if spec["tentative"]:
                        spec["tentative"] = False
                        print("[vad] resolved as noise/short -> resume sent", flush=True)
                        await ws.send_json({"type": "answer_resume"})
                    early = spec.get("early")
                    spec["early"] = None
                    if early is not None:
                        if transcript == early:
                            return  # speculative turn already answering exactly this -- done
                        # Final transcript differs from what the speculative
                        # turn is answering (the tail 300ms revised the
                        # decode) -- kill it and start over with the real
                        # text. Still inaudible at this point.
                        print("[vad] final transcript differs from early commit -> restarting turn", flush=True)
                        task = state["current_task"]
                        if task is not None and not task.done():
                            task.cancel()
                    if transcript:
                        await turn_queue.put((transcript, pcm, spec_task))
                    elif spec_task is not None and not spec_task.done():
                        spec_task.cancel()  # empty turn -- don't leave it dangling

            while True:
                try:
                    message = await ws.receive()
                except WebSocketDisconnect:
                    break
                if message["type"] == "websocket.disconnect":
                    break

                if (data := message.get("bytes")) is not None:
                    turn_pcm.extend(data)
                    samples = memoryview(data).cast("h")
                    peak = max(abs(min(samples)), abs(max(samples))) if len(samples) else 0
                    level_window_max = max(level_window_max, peak)
                    now = time.monotonic()
                    if now - level_window_start >= 1.0:
                        print(f"[mic] peak={level_window_max} ({level_window_max / 327.67:.0f}% of full scale)", flush=True)
                        level_window_max = 0
                        level_window_start = now
                    # Synchronous local DSP, ~1-3ms per 32ms chunk -- cheap
                    # enough to run inline on the event loop.
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
                        await turn_queue.put((user_text, None, None))
        finally:
            worker_task.cancel()

    # ---------- turn handling ----------

    async def _process_turn(self, ws: WebSocket, memory: MemoryBridge, transcript: str,
                             pcm: bytes | None, spec_task: "asyncio.Task | None" = None) -> None:
        try:
            await ws.send_json({"type": "user_transcript", "text": transcript})

            self._turn_counter += 1
            turn_id = f"{self._turn_counter}-{uuid.uuid4().hex[:8]}"

            # Reuse the speculative memory.search() fired while the user was
            # still mid-sentence (see SPECULATIVE_SEARCH_MIN_CHARS) instead of
            # searching again now -- if it's still running, this just awaits
            # it in place, no wasted duplicate call either way.
            if spec_task is not None:
                try:
                    result = await spec_task
                except asyncio.CancelledError:
                    result = await memory.search(query=transcript, top_k=5)
            else:
                result = await memory.search(query=transcript, top_k=5)
            await ws.send_json({"type": "memory_hits", **memory.search_payload(result)})

            instructions = memory.build_instructions(self._config.system_prompt, result)
            messages = [{"role": "system", "content": instructions}, *self._history,
                        {"role": "user", "content": transcript}]

            await ws.send_json({"type": "answer_start"})

            # Sentence-chunked streaming TTS: TTS starts on the FIRST completed
            # sentence as soon as it's available, while GPT-4o is still
            # generating the rest -- a producer (accumulates deltas, detects
            # sentence boundaries, sends answer_delta) and a consumer (pulls
            # completed sentences off a queue, synthesizes+forwards audio,
            # strictly in order so audio never plays out of sequence) run
            # concurrently. Real measurement (e2e_latency_test.py /
            # test_speculative_prefetch.py): first-audio-byte latency drops
            # substantially vs. waiting for the full reply before any TTS call.
            reply_parts: list[str] = []
            sentence_queue: asyncio.Queue[str | None] = asyncio.Queue()
            _SENTENCE_END = re.compile(r"[.!?。!?\n]")
            # Flushing to TTS on EVERY sentence boundary (the first version of
            # this code) made real replies sound choppy/robotic: each short
            # sentence got its own independent TTS call with no prosody
            # continuity from the previous one, plus a real network gap
            # between calls that the player's prebuffer cushion couldn't
            # always absorb -- reported directly from real testing of this
            # exact demo. Batching multiple sentences up to a minimum length
            # before flushing cuts the number of separate TTS calls (and
            # therefore the number of prosody resets + inter-call gaps)
            # while still not waiting for the ENTIRE reply like the original
            # (pre-streaming) version did.
            #
            # An earlier attempt lowered this to 40 GLOBALLY and measured no
            # first-audio improvement (838ms vs 842ms text-to-audio gap) --
            # the TTS API's ~700-1000ms per-call time-to-first-byte floor
            # dominates. What that experiment DID leave on the table: the
            # floor starts being paid only when the first chunk is
            # DISPATCHED, and at 80 chars GPT-4o needs ~0.5s of generation
            # before dispatch. So: small FIRST chunk (start paying the TTS
            # floor as early as possible), full-size chunks after (prosody
            # continuity for the body of the reply, which is what the
            # 80-char batching was protecting all along).
            MIN_TTS_CHARS = 80
            FIRST_TTS_CHARS = 30

            async def _produce_text() -> None:
                stream = await self._client.chat.completions.create(
                    model=self._config.openai_chat_model, messages=messages, stream=True,
                )
                current = ""
                first_flush_done = False
                async for chunk in stream:
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    if not delta:
                        continue
                    reply_parts.append(delta)
                    await ws.send_json({"type": "answer_delta", "text": delta})
                    current += delta
                    threshold = MIN_TTS_CHARS if first_flush_done else FIRST_TTS_CHARS
                    if len(current) >= threshold:
                        last_end = None
                        for m in _SENTENCE_END.finditer(current):
                            last_end = m.end()
                        if last_end is not None:
                            sentence, current = current[:last_end], current[last_end:]
                            if sentence.strip():
                                await sentence_queue.put(sentence.strip())
                                first_flush_done = True
                if current.strip():
                    await sentence_queue.put(current.strip())
                await sentence_queue.put(None)  # end-of-turn sentinel

            async def _consume_tts() -> None:
                # TARGET_CHUNK_BYTES: re-chunk to a fixed, even-byte-aligned
                # size before forwarding. response.iter_bytes() yields
                # whatever the HTTP layer happens to deliver -- if a chunk
                # boundary fell mid-sample (odd byte offset into the 16-bit
                # PCM stream), the browser's per-frame `new
                # Int16Array(arrayBuffer)` would misalign every sample after
                # that point, sounding like clicks/stutter rather than a
                # clean gap. TARGET_CHUNK_BYTES is even, so every flush stays
                # sample-aligned; this also smooths out arrival granularity.
                TARGET_CHUNK_BYTES = 4800  # ~100ms @ 24kHz 16-bit mono
                while True:
                    sentence = await sentence_queue.get()
                    if sentence is None:
                        return
                    buf = bytearray()
                    async with self._client.audio.speech.with_streaming_response.create(
                        model=self._config.openai_tts_model,
                        voice=self._config.openai_tts_voice,
                        input=sentence,
                        response_format="pcm",
                    ) as response:
                        async for audio_chunk in response.iter_bytes():
                            buf.extend(audio_chunk)
                            while len(buf) >= TARGET_CHUNK_BYTES:
                                await ws.send_bytes(bytes(buf[:TARGET_CHUNK_BYTES]))
                                del buf[:TARGET_CHUNK_BYTES]
                    if len(buf) % 2:
                        buf = buf[:-1]  # drop a trailing half-sample byte, if any
                    if buf:
                        await ws.send_bytes(bytes(buf))

            await asyncio.gather(_produce_text(), _consume_tts())
            reply_text = "".join(reply_parts)

            self._history.append({"role": "user", "content": transcript})
            self._history.append({"role": "assistant", "content": reply_text})

            # answer_done fires only after TTS audio has finished streaming too,
            # so the client can rely on it as "this turn's reply, text AND
            # audio, is fully delivered" rather than racing trailing audio frames.
            await ws.send_json({"type": "answer_done"})

            asyncio.create_task(self._background_ingest(memory, transcript, pcm, turn_id))
        except Exception as exc:  # noqa: BLE001 - surface any provider error to the client, don't swallow
            traceback.print_exc()  # background/fire-and-forget errors are otherwise easy to lose entirely
            try:
                await ws.send_json({"type": "error", "message": str(exc)})
            except Exception:
                pass  # socket already closed/closing — nothing more we can do

    async def _background_ingest(self, memory: MemoryBridge, transcript: str,
                                  pcm: bytes | None, turn_id: str) -> None:
        try:
            await memory.ingest(transcript, pcm=pcm, turn_id=turn_id, speaker="user",
                                 session_id=self._turn_counter)
        except Exception:
            traceback.print_exc()  # this task's return value is never awaited/checked anywhere else
