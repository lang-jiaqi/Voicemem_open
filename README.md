# VoiceMem

VoiceMem is a memory framework for conversational agents built around a
**left-brain / right-brain split**:

- **Left brain** (`voicemem_core/leftbrain/`) — structured factual memory: entity
  extraction, a cognitive graph over "slots" (topics, unified under one
  7-value taxonomy — see `cognitive_graph/slot_v2.py`), retrieval by
  slot-classification + entity-narrowing. Raw facts are stored via
  [mem0](https://github.com/mem0ai/mem0) with a local/embedded Qdrant vector
  store (`leftbrain/mem0_backend_store.py`); the cognitive graph itself
  references mem0's memory ids rather than duplicating storage.
- **Right brain** (`voicemem_core/rightbrain/`, `voicemem_core/emotion/`) — episodic /
  emotional memory: per-turn valence-arousal tracking, anomaly detection, and
  memory attribution for emotionally significant turns.
- **Fusion** (`voicemem_core/fusion/`) — orchestrates both hemispheres into a
  single `Search()` / `Ingest()` API and builds the final prompt context.
- **Persona / prestimulus** (`voicemem_core/persona/`, `voicemem_core/prestimulus/`) —
  a standing user-preference snapshot and pre-loaded task context injected
  ahead of retrieval.

## Beyond text: an audio-native perception layer

Unlike a text-only memory system, VoiceMem has several components that
operate directly on raw audio rather than on transcripts:

| Component | What it does | Input |
|---|---|---|
| `emotion/vad_audio.py` | Prosodic valence/arousal estimation straight from the waveform (RMS, zero-crossing rate, dynamic range) | raw audio, no LLM |
| `environment_detector_ast.py` (+ optional `environment_detector_clap.py`) | Audio Spectrogram Transformer tagging — background scene, music/humming, abnormal sounds (glass breaking, alarms, screaming); optional CLAP refinement for the background-scene description | raw audio |
| `speaker_encoder.py` + `voiceprint_store.py` | 3D-Speaker ERes2Net speaker embeddings, adaptive multi-centroid voiceprint profiles for speaker ID | raw audio |
| `emotion/attribution_qwen_omni.py` | Feeds audio directly into Qwen2.5-Omni (native audio input, not ASR-then-text) for emotion attribution | raw audio |

What VoiceMem does **not** do itself: speech-to-text transcription and
per-utterance `emotion2vec` tagging are expected to happen upstream (see
`voice_input.py`) — `Ingest()` takes already-transcribed `text`, and
optionally `audio_path` for the perception layers above. There is no bundled
ASR or voice-output pipeline in this package; see
[`examples/ingest_audio.py`](examples/ingest_audio.py) for a self-contained
way to exercise the audio-native layers on a `.wav` file.

## Install

```bash
pip install -e .                                       # core: leftbrain/rightbrain/fusion, text only
pip install -e ".[audio,environment,omni,voiceprint]"   # + scene/speaker/emotion from raw audio
```

Set `OPENAI_API_KEY` (optionally `OPENAI_BASE_URL` / `OPENAI_MODEL` /
`OPENAI_EMBEDDING_MODEL`) — the left-brain extraction/classification and
retrieval ranking call the OpenAI-compatible chat/embeddings API.

## Two layers: `voicemem` (front-desk) vs `voicemem_core` (engine)

- **`voicemem`** — the front-desk (`voicemem.py`): all capabilities wrapped as
  flat, directly-callable functions returning plain dicts/lists. This is what
  demos and quick usage import.
- **`voicemem_core`** — the engine package: the actual implementation
  (`VoiceMem` class, left/right-brain, fusion, audio-native components). Use it
  when you need fine-grained control.

## Quick start

```python
import voicemem                       # 前台：扁平函数

voicemem.ingest("Had ramen with Alex near the office at noon.")
for hit in voicemem.search("what did I eat for lunch"):
    print(hit)                        # {"text": ..., "score": ...}
```

Need full control? Use the engine directly:

```python
from voicemem_core import VoiceMem
vm = VoiceMem()
result = vm.Search("what did I eat for lunch", slots=["food"])
```

Memory is stored under `memory/leftbrain/` by default (mem0/Qdrant for raw
facts, SQLite for the cognitive graph and right-brain data); override the
root with the `VOICEMEM_MEMORY_ROOT` environment variable.

## Where the front-desk meets the voice layer

The engine's components are all exported from `voicemem_core`, so
`from voicemem_core import <Component>` works for all of them — the audio-native
ones (`SpeakerEncoder`, `ASTEnvironmentDetector`, …) load lazily, so plain
`import voicemem_core` never pulls in torch/sherpa and the text-only core
install stays light. See `voicemem_core/__init__.py` for the full component
directory.

`Ingest()` is a three-step pipeline: **preprocess → assemble → write**. The
first step is a standalone, public **streaming-preprocessing** seam that runs
all acoustic perception (scene / speaker / emotion …) on one turn of audio and
returns an `AudioPerception` — via the front-desk `voicemem.preprocess(...)`
gives you the signals without writing a memory:

```python
sig = voicemem.preprocess("...", audio_path="turn.wav")   # no write
print(sig["scene"], sig["speaker_id"], sig["emotion"])
```

## Released model adapter

Our QLoRA adapter, trained for the VoiceMem memory workflow, is available on
[Hugging Face: `LangJiaqi77/Voicemem-Qwen3_6-35B-A3B-QLoRA-v2`](https://huggingface.co/LangJiaqi77/Voicemem-Qwen3_6-35B-A3B-QLoRA-v2).
It is an adapter-only release for
[`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B); release
details, loading code, checksums, and evaluation notes are in
[`models/voicemem-qwen3.6-35b-a3b-qlora-v2/`](models/voicemem-qwen3.6-35b-a3b-qlora-v2/).

## Audio-native features

```python
vm.Ingest("...", audio_path="turn.wav")   # also runs scene/speaker/VAD detection
vm.IngestEnv(audio_path="turn.wav")       # background-scene detection only
```

`speaker_encoder.py` extracts a 192-dim 3D-Speaker ERes2Net embedding for each
utterance via a worker subprocess (`voicemem_core/voiceprint/campplus_worker.py`,
kept separate because it needs `sherpa-onnx`). The worker reads the ONNX
weights from `service/models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx`
by default — run `bash scripts/download_models.sh service/models` once to
fetch it (this works whether or not you use `service/`), or point
`VOICEMEM_SPEAKER_MODEL` at a model file elsewhere. By default the worker
runs in the same interpreter (`sys.executable`); to isolate it into a
dedicated environment, set `VOICEMEM_AUDIOMEM_PYTHON=/path/to/other/python`.

Environment classification uses the Audio Spectrogram Transformer (AST) by
default. Install the optional dependencies with `pip install -e ".[environment]"`;
the model downloads on first use, or set `VOICEMEM_ENVIRONMENT_MODEL_DIR` to a
pre-downloaded local model directory.

For a more accurate background-sound *description* (4-second-segmented CLAP,
tested at ~60% accuracy vs. AST's raw score), install `pip install -e ".[clap]"`
and set `VOICEMEM_CLAP_CHECKPOINT=/path/to/630k-audioset-best.pt` — it takes
over the description memory write automatically once configured. AST still
supplies the immediate hint and still covers music/abnormal-sound detection
and the place-clustering embedding. Set `VOICEMEM_ENVIRONMENT_MEMORY_BACKEND=ast`
to opt back out.

For multimodal emotion attribution with Qwen2.5-Omni, see
[`examples/load_qwen_omni_attributor.py`](examples/load_qwen_omni_attributor.py)
— `QwenOmniEmotionAttributor` takes an already-loaded processor/model/tokenizer
via dependency injection; the example shows how to load them.

Try the perception layers directly on a `.wav` file:

```bash
python examples/ingest_audio.py path/to/turn.wav
python examples/ingest_audio.py path/to/turn.wav --text "..." --ingest   # + full memory pipeline
```

## Demos

Two independent, working demos build on top of `voicemem_core/` — neither is
required to use the core package as a library.

**[`openai_voice_demo/`](openai_voice_demo/)** — the actively-maintained
reference demo: an OpenAI-based (GPT-4o pipeline or Realtime API) voice
conversation loop with a browser frontend and an Electron desktop wrapper.
See its own README for setup and architecture. On launch it first runs a
per-component **startup self-check** (`voicemem_core/startup_check.py`) that times
each component against a speed budget and prints a short console report: if all
pass it starts directly, otherwise it asks whether to launch anyway. Set
`VOICEMEM_SKIP_STARTUP_CHECK=1` to skip it, or `VOICEMEM_STARTUP_BUDGET_<KEY>`
to tune a budget.

**[`service/`](service/)** — an independent, Gemini Live-based WebSocket
backend (ASR, VAD, speaker/emotion signals, memory retrieval, voice replies).
It streams microphone PCM audio in and JSON/PCM frames out; the protocol is
documented in [`docs/PROTOCOL.md`](docs/PROTOCOL.md). This repo does not
currently bundle a browser client for it — see
[`service/README.md`](service/README.md) for how to run the backend and what
you'd need to build or bring your own frontend against it.

## License

MIT — see [LICENSE](LICENSE).
