# examples

Five runnable examples, in two groups.

```bash
pip install -e .
export OPENAI_API_KEY=sk-...
```

## 1. VoiceMem as a memory engine

Your program calls it to **store** and **recall**. It does not reply.

| | stores | uses |
|---|---|---|
| [`01_memory_text.py`](01_memory_text.py) | a line of text | left brain only |
| [`02_memory_audio.py`](02_memory_audio.py) | a wav file | both brains, plus ASR / voiceprint / scene / emotion |

```bash
python examples/01_memory_text.py
python examples/02_memory_audio.py speech.wav
```

Start with `01`. `02` is the same thing with audio input; it needs
`bash scripts/download_models.sh` for the local models.

## 2. Building a voice agent

Mic in, memory prefetched while you speak, reply out, turn stored.
**These two differ only in who generates the reply** — the memory half is identical.

| | reply model | needs |
|---|---|---|
| [`03_voice_agent.py`](03_voice_agent.py) | OpenAI (default) | `pip install sounddevice` |
| [`04_voice_agent_own_model.py`](04_voice_agent_own_model.py) | your fine-tuned Qwen adapter | a 35B base, see the file header |

```bash
python examples/03_voice_agent.py
python examples/04_voice_agent_own_model.py
```

To watch what the stream does internally (partials, each speculation stage),
see [`example.py`](../example.py) at the repo root — the long-form version of `03`
that also accepts a wav file.

## 3. Swapping internal components

[`05_custom_asr_vad.py`](05_custom_asr_vad.py) is not a separate application — it is
**the parts-replacement guide for `03`**. Everything above uses the built-in ASR/VAD;
this shows how to use yours instead. Two ways:

**Way 1 — swap the components.** For an ASR that eats audio chunk by chunk and can
report the transcript so far:

```python
vm = VoiceMem(asr=lambda: MyASR(), vad=lambda: MyVAD())   # zero-arg factories
```

The whole protocol, with `samples` as float32 / 16 kHz / mono / [-1, 1]:

```python
class MyASR:
    def reset(self): ...                 # start of a turn
    def feed(self, samples) -> str: ...  # transcript so far
    def flush(self) -> str: ...          # optional, tail flush

class MyVAD:
    def is_speech(self, samples) -> bool: ...
```

**Way 2 — feed text only.** Your ASR/VAD already run elsewhere (cloud, OS, another
process), so VoiceMem never touches audio and loads no audio model:

```python
await stream.feed_partial("I'm allergic")
turn = (await stream.feed_partial("I'm allergic to nuts", ended=True)).turn
```

`ended=True` is your VAD saying the utterance is complete.

## One gotcha: memory stores are not interchangeable

Without `memory_root` every example shares one store (`results/voice_memory` under
the package). Point them somewhere else to keep them separate:

```python
VoiceMem(..., memory_root="/tmp/my_mem")
```

**Changing the embedding dimension makes an existing store unusable.** The default
OpenAI embedding is 1536-d; the local E5 from
`from_config({"embedding": {"provider": "local"}})` is 384-d. Aiming both at the same
`memory_root` fails with a shape mismatch.
