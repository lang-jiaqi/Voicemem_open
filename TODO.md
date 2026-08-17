# Voicemem TODO

## service/

- `service/` (the Gemini Live WebSocket backend) currently has no bundled
  browser frontend in this repo — the protocol it speaks is documented in
  [`docs/PROTOCOL.md`](docs/PROTOCOL.md), but building a client against it is
  left to the integrator.
- Its dependencies (`sherpa-onnx`, `funasr`, `websockets`, `jieba`,
  `model2vec`, `google-genai`) need to be installed separately — see
  [`service/README.md`](service/README.md).

## Test coverage

- `voicemem`'s mem0/Qdrant storage backend, the right-brain heartnote-write
  path, and legacy-slot coercion in `cognitive_graph/store.py` have targeted
  regression tests (see `tests/`), but broader end-to-end coverage of the
  full left-brain/right-brain retrieval pipeline is still thin.

## Audio-native perception layers

- Scene/speaker/emotion detection (`enable_scene`, `enable_voiceprint`,
  `enable_emotion`, …) are optional installs (`pip install -e
  ".[audio,environment,omni,voiceprint]"`) and are exercised manually via
  [`examples/ingest_audio.py`](examples/ingest_audio.py) rather than in CI.
