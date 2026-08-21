#!/usr/bin/env python3
"""文本模式：只用左脑，存文字、查文字。voicemem 最小的用法。

    export OPENAI_API_KEY=sk-...
    python examples/01_text.py
"""
import os

from voicemem import VoiceMem

vm = VoiceMem(mode="leftbrain_only", openai_key=os.environ["OPENAI_API_KEY"], top_k=5)

for line in [
    "我是素食主义者，对坚果过敏。",
    "周末喜欢去 East Coast 骑车看海。",
    "下周三下午三点要跟导师开组会。",
]:
    vm.ingest(line)
    print("存:", line)

result = vm.search("我的饮食禁忌是什么？")
print("\n查: 我的饮食禁忌是什么？")
for hit in result.hits:
    print("  -", hit.text)
