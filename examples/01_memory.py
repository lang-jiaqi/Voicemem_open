#!/usr/bin/env python3
"""Store and recall. VoiceMem as a memory engine -- your program calls it, it does
not reply.

Text in, left brain only:

    export OPENAI_API_KEY=sk-...
    python examples/01_memory.py

Audio in, both brains (ASR, voiceprint, scene and emotion all run):

    bash scripts/download_models.sh
    python examples/01_memory.py --audio speech.wav --ask "what did I say about food?"

Same two calls either way -- ``ingest`` then ``search``. Only the input differs.
"""
import argparse
import os

from voicemem import VoiceMem

LINES = [
    "I'm a vegetarian and allergic to nuts.",
    "I like biking along East Coast on weekends.",
    "Group meeting with my advisor next Wednesday at 3pm.",
]

p = argparse.ArgumentParser()
p.add_argument("--audio", help="wav to store, instead of the built-in text lines")
p.add_argument("--ask", default="", help="what to recall")
args = p.parse_args()

key = os.environ["OPENAI_API_KEY"]

if args.audio:
    vm = VoiceMem(mode="normal", openai_key=key, top_k=5)
    vm.ingest(audio=args.audio)
    print("stored:", args.audio, "->", vm.transcribe(args.audio))
    # Nothing is seeded here, so a fixed question would just print an empty result.
    # Pass --ask something the recording actually talks about.
    query = args.ask or "What do you know about me?"
else:
    vm = VoiceMem(mode="leftbrain_only", openai_key=key, top_k=5)
    for line in LINES:
        vm.ingest(line)
        print("stored:", line)
    query = args.ask or "What are my dietary restrictions?"

result = vm.search(query)
print(f"\nquery: {query}")

print("\nleft brain (facts):")
for hit in result.hits:
    print("  -", hit.text)
if not result.hits:
    print("  (empty)")

# Only mode="normal" fills the right brain; leftbrain_only leaves it empty.
if result.result_rightbrain:
    print("\nright brain (who they are, how to respond):")
    print(result.result_rightbrain)
