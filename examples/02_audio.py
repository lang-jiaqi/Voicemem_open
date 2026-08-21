#!/usr/bin/env python3
"""音频模式：喂一个 wav，内部跑 ASR / 声纹 / 场景 / 情绪，双脑都写。

跟文本模式的区别：mode="normal"，ingest 给 audio= 而不是文本。查询时右脑
（人物画像、情绪、回复倾向）也会一起返回。

    bash scripts/download_models.sh          # 本地模型
    export OPENAI_API_KEY=sk-...
    python examples/02_audio.py speech.wav
"""
import os
import sys

from voicemem import VoiceMem

wav = sys.argv[1] if len(sys.argv) > 1 else "speech.wav"

vm = VoiceMem(mode="normal", openai_key=os.environ["OPENAI_API_KEY"], top_k=5)

vm.ingest(audio=wav)
print("存:", wav, "→", vm.transcribe(wav))

result = vm.search("我的饮食禁忌是什么？")
print("\n左脑（事实）:")
print(result.result_leftbrain or "  （空）")
print("\n右脑（人怎么样、该怎么回）:")
print(result.result_rightbrain or "  （空）")
