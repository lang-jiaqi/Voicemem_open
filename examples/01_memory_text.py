#!/usr/bin/env python3
"""Store and recall text. Left brain only.

    export OPENAI_API_KEY=sk-...
    python examples/01_memory_text.py
"""
import os

from voicemem import VoiceMem

vm = VoiceMem(mode="leftbrain_only", openai_key=os.environ["OPENAI_API_KEY"], top_k=5)

for line in [
    "I'm a vegetarian and allergic to nuts.",
    "I like biking along East Coast on weekends.",
    "Group meeting with my advisor next Wednesday at 3pm.",
]:
    vm.ingest(line)
    print("stored:", line)

result = vm.search("What are my dietary restrictions?")
print("\nquery: What are my dietary restrictions?")
for hit in result.hits:
    print("  -", hit.text)
