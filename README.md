# VoiceMem
test
VoiceMem is a memory framework for conversational agents built around a
**left-brain / right-brain split**:

- **Left brain** (`voicemem/leftbrain/`) — structured factual memory: entity
  extraction, a cognitive graph over "slots" (topics, unified under one
  7-value taxonomy — see `cognitive_graph/slot_v2.py`), retrieval by
  slot-classification + entity-narrowing. Raw facts are stored via
  [mem0](https://github.com/mem0ai/mem0) with a local/embedded Qdrant vector
  store (`leftbrain/mem0_backend_store.py`); the cognitive graph itself
  references mem0's memory ids rather than duplicating storage.
- **Right brain** (`voicemem/rightbrain/`, `voicemem/utils/audio/emotion/`) — episodic /
  emotional memory: per-turn valence-arousal tracking, anomaly detection, and
  memory attribution for emotionally significant turns.
- **Fusion** (`voicemem/fusion/`) — orchestrates both hemispheres into a
  single `Search()` / `Ingest()` API and builds the final prompt context.
  (The user profile / persona lives in the right brain as a `source="profile"`
  hit, retrieved by query like any other slot — no separate pre-stimulus layer.)

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

## API structure: `VoiceMem` = left_brain / right_brain / utils

`voicemem/core.py` is a small, readable top layer; the heavy implementation
lives in `voicemem/engine.py` and is only delegated to.

- **`VoiceMem`** — top entry. Takes `api_key`, `mode`, and per-util overrides.
- **`.left_brain` / `.right_brain`** — each has `search()` / `store()`.
- **`.utils`** — the swappable capabilities: entity / schema / ASR / emotion /
  voiceprint / embedding / **memory_engine** (mem0 by default; pass another to
  use e.g. zep). Each defaults to a builtin; override by passing a function.
- **`mode`** ∈ `left_brain_single` / `text_mode` / `multi_modal` — decides which
  utils load. Utils load lazily, only when needed.

## Quick start

```python
from voicemem import VoiceMem

vm = VoiceMem(api_key="sk-...", mode="text_mode")
vm.ingest("Had ramen with Alex near the office at noon.")

for hit in vm.left_brain.search("what did I eat for lunch"):
    print(hit)                        # left-brain facts

vm.test()                             # startup self-check: 4-tier speed table
```

Swap a util by passing a function (default is a builtin):

```python
vm = VoiceMem(mode="text_mode",
              embedding=lambda: MyEmbedder(),
              memory_engine=lambda: MyZepStore())   # replace mem0
```

Memory is stored under `memory/leftbrain/` by default (mem0/Qdrant for raw
facts, SQLite for the cognitive graph and right-brain data); override the
root with the `VOICEMEM_MEMORY_ROOT` environment variable.

## Components

Every engine component is also exported lazily from the package, so
`from voicemem import SpeakerEncoder` etc. works — the audio-native ones load
lazily, so plain `import voicemem` never pulls in torch/sherpa and the
text-only core install stays light. See `voicemem/__init__.py`.

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
utterance via a worker subprocess (`voicemem/utils/audio/voiceprint/campplus_worker.py`,
kept separate because it needs `sherpa-onnx`). The worker reads the ONNX
weights from `models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx`
by default — run `bash scripts/download_models.sh models` once to
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

**[`web/`](web/)** — the single, minimal demo: `run.py` (core conversation logic)
+ `utils.py` (pipeline) + one brain-graph `index.html` (rendering only). Talk to
it, watch the left/right-brain graph grow live. It shows the **EOU-anticipatory
0–500ms pipeline**: local streaming ASR + Silero VAD drive a *speculative* memory
prefetch that runs on a LOCAL E5 embedder + a LOCAL slot classifier (injected via
`VoiceMem(embedding=…, schema=LocalQueryClassifier(…))`) — so nothing in the hot
path hits the network, and the LLM `Classify` is skipped. By the time VAD confirms
end-of-utterance, the memory is already fetched off the critical path; the two
dead-simple control flows (`voicemem_llm_tts`, `voicemem_realtime`) just *consume*
it and reply. Barge-in (a mid-utterance pause then resume) cancels the speculation
instead of firing a turn. See [QUICKSTART.md](QUICKSTART.md).

**[`openai_voice_demo/`](openai_voice_demo/)** — the full reference demo:
OpenAI-based (GPT-4o pipeline or Realtime API) voice loop with a browser
frontend, an Electron desktop floating-orb, and `.app` packaging. Heavier; keep
it around when you want the battle-tested pipeline/realtime backends. On launch
it runs the per-component **startup self-check** (`voicemem/startup_check.py`);
set `VOICEMEM_SKIP_STARTUP_CHECK=1` to skip.

## License

MIT — see [LICENSE](LICENSE).
