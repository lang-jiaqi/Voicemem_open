#!/usr/bin/env python3
"""Store and recall audio. Both brains: ASR, voiceprint, scene and emotion all run.

    bash scripts/download_models.sh
    export OPENAI_API_KEY=sk-...
    python examples/02_memory_audio.py speech.wav
"""
import os
import sys

from voicemem import VoiceMem

wav = sys.argv[1] if len(sys.argv) > 1 else "speech.wav"

vm = VoiceMem(mode="normal", openai_key=os.environ["OPENAI_API_KEY"], top_k=5)

vm.ingest(audio=wav)
print("stored:", wav, "->", vm.transcribe(wav))

result = vm.search("What are my dietary restrictions?")
print("\nleft brain (facts):")
print(result.result_leftbrain or "  (empty)")
print("\nright brain (who they are, how to respond):")
print(result.result_rightbrain or "  (empty)")
