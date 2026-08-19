# openai_voice_demo

A minimal, standalone reference implementation showing how to wire the
[`voicemem`](../voicemem) memory engine into an **OpenAI**-based voice
conversation loop. This is a sibling of [`../service`](../service) (which
uses Gemini Live and has a lot more built-in machinery — speaker/scene/
emotion wiring, a silence-timer turn-taking state machine, Gemini-specific
streaming). This folder does not import from or modify `service/` — it
exists purely to be a smaller, easier-to-read second example, built around
OpenAI's stack instead. It also has its own [`desktop/`](desktop/) — an
Electron wrapper around this same backend, unrelated to any other desktop
app that may have existed elsewhere in this repo's history.

`voicemem/` itself is unchanged. Its audio-native perception (speaker
voiceprint ID, environment/scene detection, emotion-from-audio) is a built-in
capability of the package, not something this folder reimplements — it's
controlled by a single on/off toggle (`VOICEMEM_AUDIO_NATIVE`) that maps
directly onto the five `enable_*` constructor flags `VoiceMem` already has.

## Two voice modes, one config switch

Both modes share the same LOCAL input side (streaming ASR + VAD, see
`backend/local_asr.py`): live word-by-word captions, ~20ms turn
finalization, barge-in confirmable mid-utterance. They differ in how the
REPLY is produced:

- **`pipeline`** (default) — streamed GPT-4o chat completion + a separate
  OpenAI TTS call (sentence-chunked streaming). The well-documented,
  standard-REST-API path.
- **`realtime`** — the finished turn's transcript goes to OpenAI's Realtime
  API (`gpt-realtime`) as a text turn and the reply comes back as native
  model-generated VOICE (no separate TTS step; typically faster to first
  sound, more natural prosody). Trade-off: the model reads your words
  rather than hearing your voice, so tone-of-voice nuance in the INPUT is
  lost to it. An actively-evolving API surface — read the comment at the
  top of `backend/providers/realtime.py` before relying on it.

Switch with `VOICE_MODE=pipeline|realtime|qwen|doubao` in `.env`. The Seed
Realtime/豆包全双工 provider is `VOICE_MODE=doubao` and additionally requires
`DOUBAO_API_KEY`.

## Quick start — two ways to use this demo

You need: **Python 3.12, Node.js (orb path only), and your own
`OPENAI_API_KEY`**.

### Path 1: desktop orb (the frontend)

A transparent, always-on-top floating orb on your desktop, with the
left-brain / right-brain retrieved memories displayed live on either side.
One `npm start` launches everything (Electron spawns the Python backend as
a child process and shuts it down on quit):

```bash
# 1. clone
git clone <repo-url> && cd voicemem_opensource

# 2. python deps (voicemem core + this demo's own)
pip install -e .
pip install -r openai_voice_demo/requirements.txt

# 3. local streaming-ASR models (~570MB, one time)
bash scripts/download_models.sh models

# 4. your key
cp openai_voice_demo/.env.example openai_voice_demo/.env
#    edit it: OPENAI_API_KEY=sk-...

# 5. desktop shell deps (one time)
cd openai_voice_demo/desktop && npm install

# 6. launch — the orb appears
npm start
```

(The E5 local model, ~450MB, auto-downloads from HuggingFace on first
backend start; no manual step.) If `python3` on your PATH isn't the env you
installed into: `VOICEMEM_PYTHON=/path/to/python3 npm start`. More in
[`desktop/README.md`](desktop/README.md) — including a browser-only
variant (`python backend/main.py`, then open
`http://localhost:8787/orb.html`, or `/` for the debug page) if you don't
want Electron.

### Path 2: headless backend service (no frontend)

The backend is a self-contained WebSocket service; the orb is just one
client of it. Run steps 1-4 above, then:

```bash
cd openai_voice_demo/backend
python main.py          # serves ws://localhost:8787/ws
```

Integrate against the WS protocol directly: send **binary frames** of raw
PCM16 / 24kHz / mono mic audio (turn boundaries are detected server-side —
just stream continuously), or JSON `{"type": "user_text", "text": "..."}`
for text turns; receive JSON events (`partial_transcript`,
`user_transcript`, `memory_hits` with the left/right-brain split,
`answer_start/delta/done`, `speech_tentative`/`answer_resume`/
`answer_interrupt` for barge-in) plus binary frames of 24kHz PCM16 reply
audio. Two runnable reference clients double as verification:

```bash
python scripts/smoke_test.py                        # text-only round trip
python scripts/audio_smoke_test.py my_recording.wav  # feed a real audio file
```

## Install variants

The quick start above installs the text-only base. To also enable
voicemem's audio-native perception layers (speaker voiceprint ID,
environment/scene detection, emotion-from-audio; heavy —
torch/sherpa-onnx/funasr), install with extras and set
`VOICEMEM_AUDIO_NATIVE=true` in `.env`:

```bash
# from the repo root:
pip install -e ".[audio,environment,omni,voiceprint]"
```

In the browser variant, both voice modes are continuous-listening: click
the mic button once to start a session (the server-side VAD decides turn
boundaries, not a push-to-talk button), or just type into the text box,
which works in both modes and needs no microphone.

## Architecture

```
backend/
  main.py            FastAPI app, /ws route, mounts frontend/ as static files
  config.py           env -> Config (voice_mode, audio_native, model names, paths)
  audio_utils.py       PCM16 <-> WAV helpers
  memory_bridge.py     the ONLY module that touches VoiceMem (Search/Ingest/instructions)
  local_classifier.py  local embedding-based slot classification (demo-only speed path, see below)
  local_embedder.py    local embedding for Ingest()/Rank() memory storage (demo-only speed path, see below)
  local_emotion_classifier.py  local embedding-based emotion classification (replaces an LLM call, see below)
  local_asr.py         local streaming ASR + VAD for pipeline mode's input (sherpa-onnx, see below)
  providers/
    pipeline.py         local streaming ASR -> speculative Search -> chat completion -> TTS -> background Ingest
    realtime.py          OpenAI Realtime API bridge, memory injected via session.update
frontend/
  index.html            single file: mic capture, PCM playback, WS client (debug page)
  orb.html              floating-orb UI: orb + live left-brain/right-brain memory panels
desktop/               Electron wrapper: transparent always-on-top orb window (see desktop/README.md)
scripts/
  smoke_test.py         scripted WS client, text-turn round trip (see Testing below)
  audio_smoke_test.py   scripted WS client, feeds a REAL AUDIO FILE straight into the
                         backend -- no browser, no microphone (see Testing below)
  latency_test.py       end-of-speech-to-each-milestone timing measurement
data/                  runtime-created, gitignored: this demo's own voicemem memory store
```

## The backend is usable without the browser frontend

`frontend/index.html` is a thin UI: it captures mic PCM and streams it over
the WebSocket, and plays back whatever PCM comes back. All the actual work
(speech-to-text, memory search, chat completion, text-to-speech) happens in
`backend/`, and the backend has no idea whether the audio bytes it's
receiving came from a browser tab or anywhere else -- it's just a WebSocket
that accepts binary PCM16/24kHz/mono frames. `scripts/audio_smoke_test.py`
demonstrates this directly: it feeds a `.wav` file (or synthesizes one via
OpenAI TTS if you don't have one handy) straight into the backend over the
wire and prints back the transcript, memory hits, and answer -- see
"Testing" below.

Browser↔backend protocol (this demo's own, separate from `docs/PROTOCOL.md`):
binary WebSocket frames = 24kHz PCM16 mono audio (both directions, both
modes); JSON text frames carry everything else (`turn_start`/`turn_end`/
`user_text` from the client; `session_ready`/`user_transcript`/`memory_hits`/
`answer_start`/`answer_delta`/`answer_done`/`error` from the server).
`memory_hits` surfaces each `voicemem.Search()` call's results straight from
the `SearchResult` object, so you can see what memory context is going into
the prompt for each turn.

Pipeline mode's input side is a LOCAL streaming ASR (`local_asr.py`:
sherpa-onnx streaming zipformer zh-en + Silero VAD -- the same stack and
the same already-downloaded model files as `service/`). Real measured
before/after of the switch away from the OpenAI Realtime transcription
session: end-of-speech-to-final-transcript went from ~1.2-1.5s to ~20ms,
and captions became genuinely live word-by-word while you're still talking
(the cloud session only ever released deltas in a burst after the utterance
ended). Real, disclosed trade-offs: transcript accuracy is below the cloud
model -- English especially (comes out UPPERCASE, no punctuation, and
synthesized/TTS English audio decodes poorly; Chinese is solid), and ASR
now costs local CPU (~12x faster than real-time on this machine, so a small
fraction of one core).

Barge-in (interrupting the assistant mid-reply) is a two-stage flow, both
stages verified via scripted repro: the moment you start talking over a
reply, `speech_tentative` pauses playback (~0.5s from opening your mouth,
measured); once enough real transcript accumulates
(`MIN_INTERRUPT_CHARS`, now confirmable MID-utterance thanks to the live
local partials), `answer_interrupt` cancels the reply and flushes buffered
audio -- or, if the speech resolves to noise/nothing (e.g. a cough),
`answer_resume` un-pauses and the reply continues. The frontend keeps
playback and mic capture on SEPARATE AudioContexts specifically so pausing
playback can't freeze the mic (same context would deadlock the whole flow
-- see the comment in `frontend/index.html`).

Out of scope for this MVP: replaying historical audio, multi-user/session
auth. A single `VoiceMem` instance and one fixed `user_id`
(`VOICE_DEMO_USER_ID`) are shared across all connections.

## Testing status — read this before trusting anything untested

- **Verified**: all backend files import cleanly; `scripts/smoke_test.py`
  exercises the full `Search()`/`Ingest()` wiring end-to-end over a live
  WebSocket connection using the `user_text` path (no audio needed) — see
  that script for exact steps and what it actually proves.
  `scripts/audio_smoke_test.py` does the same but with real audio, fed
  directly into the backend (no browser): `python scripts/audio_smoke_test.py
  [path/to/recording.wav] [ws://host:port/ws]` — with no file given it
  synthesizes its own test speech via OpenAI TTS. Verified against both a
  synthesized single utterance and a real 10-second bilingual recording
  containing two separate turns — transcription, memory retrieval, and the
  answer (text + TTS audio) all came back correctly for both. Real microphone
  audio through a browser (over an SSH tunnel to this backend) has also been
  exercised live in both `pipeline` and `realtime` modes across many turns —
  several real bugs (self-triggered barge-in from the assistant's own TTS
  echo, missing streamed answer text in realtime mode, an empty right-brain
  panel traced to a legacy-data crash swallowed by a blanket
  `except Exception`) were found and fixed this way, not just guessed at.
- **Not verified**: the Electron wrapper in `desktop/` has not been visually
  verified (built on a headless dev machine with no display) — see its own
  README's "Known limitations". Browsers/OS/microphone combinations other
  than the one used for the live testing above are untested.

## Known limitations

- No concurrency hardening: `Search()` then a background `Ingest()` task,
  never true parallel calls into `VoiceMem`, but this hasn't been stress
  tested with multiple simultaneous browser connections.
- `frontend/index.html` uses the deprecated (but still functional)
  `ScriptProcessorNode` for mic capture, specifically to keep the frontend a
  single dependency-free file rather than loading a separate AudioWorklet module.
- The OpenAI Realtime API's event/session schema changes over time — if
  `providers/realtime.py` throws schema-validation errors, re-check
  `openai/types/realtime/*.py` in your installed SDK version the same way
  this file's header comment describes.
- **Search() slot classification is local, not voicemem's own `Classify()`**:
  `memory_bridge.search()` uses `local_classifier.py` (a local embedding
  model, `intfloat/multilingual-e5-small`) instead of `VoiceMem.Classify()`'s
  LLM call, purely for this demo's speed — ~5ms/query instead of ~1.2s,
  measured at 93% slot-match agreement with `Classify()`'s own answers on a
  bilingual test set (see that module's docstring for the real numbers).
  voicemem's own `Classify()` is untouched and remains the standard,
  documented implementation. Real, disclosed cost: cosine similarity over a
  fixed slot list can't do open-vocabulary entity extraction the way an LLM
  can, so this demo always searches with `entities=[]` — slot-only
  narrowing, not slot+entity.
- **Rank()'s embedding and emotion classification are also local now, not
  OpenAI network calls**: `memory_bridge.py` constructs
  `VoiceMem(embedder=LocalMemoryEmbedder(...))` (see `local_embedder.py`) —
  voicemem's own official default remains OpenAI embeddings (unaffected for
  anyone not passing a custom `embedder`); this demo swaps in the same
  local E5 model `local_classifier.py` already uses, shared as one loaded
  model instance. Emotion classification (`local_emotion_classifier.py`)
  used to be an LLM call (`gpt-4o-mini`) — unlike slot classification and
  embedding, there was no pre-existing "official core voicemem" version of
  text-only emotion classification to preserve as a fallback (it was a
  demo-only bridge to begin with), so this is simply the implementation now,
  in both the demo and — since nothing else in this repo does text-only
  emotion classification — the only version there currently is.
  Real, disclosed cost of the local embedder specifically: it changes
  vector dimensions (384 vs. OpenAI's 1536), so it is NOT compatible with a
  memory store built using OpenAI embeddings — `data/` had to be wiped when
  this was introduced, and it must be wiped again for anyone upgrading an
  existing installation into this.
  Real, measured effect: with both changes in place, `transcript_locked`
  and `memory_hits` arrived at the exact same millisecond timestamp in
  every run of a real 4-run test (e.g. 933ms/933ms, 2139ms/2139ms,
  1311ms/1311ms, 984ms/984ms from end-of-speech) — the entire search step
  (slot classification + emotion classification + memory ranking) is now
  fast enough to not register as separate from transcript locking, down
  from a measured ~1565ms gap before either change. Response generation
  itself still takes several seconds on top of this — see
  `providers/pipeline.py`/`realtime.py`'s speculative-search wiring
  (`SPECULATIVE_SEARCH_MIN_CHARS`) for how this demo hides most of the
  remaining cost behind the user's own speech time instead of eliminating
  it outright.
