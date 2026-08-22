"""存和查：音频进 → 双脑；文本进 → 只有左脑。

    export OPENAI_API_KEY=sk-...
    python examples/01_memory.py
"""
import os

from voicemem import VoiceMem

KEY = os.environ["OPENAI_API_KEY"]

vm = VoiceMem(
    mode="normal",
    openai_key=KEY,
    top_k=5,
)

# 存：音频文件
# 内部跑 ASR / 声纹 / 场景 / 情绪感知 / Embedding 抽取
vm.ingest(audio="input.wav")  # 我是素食主义者，对坚果过敏。

result = vm.search("我的饮食禁忌是什么？")

print(result.result_leftbrain, result.result_rightbrain)


# 存：左脑信息文本（无情感）
vm = VoiceMem(
    mode="leftbrain_only",
    openai_key=KEY,
    top_k=5,
)

vm.ingest("我是素食主义者，对坚果过敏。")

result = vm.search("我的饮食禁忌是什么？")
