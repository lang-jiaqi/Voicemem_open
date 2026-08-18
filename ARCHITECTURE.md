# Voicemem Agent architecture

```text
voicemem_opensource/
├── voicemem_core/                 Core package: left brain, right brain, fusion, retrieval, memory storage
├── openai_voice_demo/        Working reference demo (pipeline + realtime voice modes)
│   ├── backend/               FastAPI app; the only place that imports `voicemem` in this demo
│   ├── frontend/              Single-file browser client
│   └── desktop/               Electron wrapper around the same backend (see its own README)
├── service/                   Standalone WebSocket backend (Gemini Live) — see service/README.md
├── models/                    Released fine-tuned model card(s), e.g. the QLoRA adapter
├── docs/                      Protocol docs
├── examples/                  Minimal integrations against `voicemem` directly
├── scripts/                   Development scripts (model downloads, etc.)
└── tests/                     Unit tests for `voicemem`
```

## Runtime boundaries

- `voicemem_core/` has no UI/network code of its own — it's a library. Anything that
  talks to a model provider, a browser, or a socket lives in one of the demo
  folders below it and imports `voicemem` as a dependency.
- `openai_voice_demo/backend/memory_bridge.py` is the *only* module in that
  demo that imports `voicemem` — both voice providers (`pipeline.py`,
  `realtime.py`) go through it, so there is exactly one place that knows
  about `VoiceMem`'s constructor flags and `Ingest()`/`Search()`/`Classify()`
  signatures.
- `service/` is an independent, older WebSocket backend (Gemini Live) that
  also calls `voicemem` directly; it does not share code with
  `openai_voice_demo/`. It currently has no bundled browser frontend — see
  `service/README.md` for what's needed to drive it.

## Rules

- `voicemem_core/` must remain usable without any of the demo folders (no UI
  imports leaking into the core package).
- Each demo (`openai_voice_demo/`, `service/`) owns its own provider/API-key
  handling; `voicemem_core/` itself only needs `OPENAI_API_KEY` (and friends) for
  its own internal extraction/classification/ranking calls.
