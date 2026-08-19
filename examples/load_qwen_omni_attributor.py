#!/usr/bin/env python3
"""如何加载 Qwen2.5-Omni 并接进 voicemem 的多模态情绪归因层。

voicemem.utils.audio.emotion.attribution_qwen_omni.QwenOmniEmotionAttributor 用依赖注入
接收已经加载好的 processor/model/tokenizer，本身不负责下载/加载模型权重。
这个脚本展示最小可跑通的加载方式，替代直接把音频塞进多模态 LLM 做情绪归因。

用法::

    python examples/load_qwen_omni_attributor.py \\
        --model Qwen/Qwen2.5-Omni-7B \\
        --audio path/to/turn.wav \\
        --asr-text "刚才用户说的那句话（如果已有转录，不给也可以）"

--model 可以是 HuggingFace Hub 模型 id（会自动下载），也可以是本地目录路径。
"""
from __future__ import annotations

import argparse
import os


def load_omni(model_path: str, *, dtype: str = "auto", device_map: str = "auto"):
    """加载 Qwen2.5-Omni 的 processor / tokenizer / model 三件套。

    仅需要文本输出（情绪归因不需要模型吐语音），优先用
    Qwen2_5OmniThinkerForConditionalGeneration ——比完整 Omni 省显存。
    """
    import torch
    from transformers import AutoTokenizer

    torch_dtype = {
        "auto": "auto",
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[dtype]

    try:
        from transformers import Qwen2_5OmniProcessor
        processor = Qwen2_5OmniProcessor.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        raise RuntimeError(
            f"加载 Qwen2_5OmniProcessor 失败：{e}\n"
            "需要较新版本 transformers，参考模型卡：pip install -U \"transformers>=4.50\""
        ) from e

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    from transformers import Qwen2_5OmniThinkerForConditionalGeneration

    load_kwargs = {"device_map": device_map, "trust_remote_code": True}
    if torch_dtype == "auto":
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(model_path, **load_kwargs)
    else:
        try:
            model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
                model_path, dtype=torch_dtype, **load_kwargs
            )
        except TypeError:
            model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=torch_dtype, **load_kwargs
            )

    model.eval()
    return processor, tokenizer, model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("QWEN_OMNI_MODEL", "Qwen/Qwen2.5-Omni-7B"))
    parser.add_argument("--audio", required=True, help="待归因的音频文件路径（该轮用户发言）")
    parser.add_argument("--asr-text", default=None, help="该轮的转录文本（可选，没有就不传）")
    parser.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--device-map", default="auto")
    args = parser.parse_args()

    from voicemem.utils.audio.emotion.attribution_qwen_omni import QwenOmniEmotionAttributor
    from voicemem.utils.audio.emotion.types import VAD, TurnEmotionRecord

    processor, tokenizer, model = load_omni(args.model, dtype=args.dtype, device_map=args.device_map)
    attributor = QwenOmniEmotionAttributor(processor=processor, model=model, tokenizer=tokenizer)

    # 演示用最小 turn：真实接入时这条记录应该来自 EmotionLayer 的 VAD 估计结果。
    turn = TurnEmotionRecord(
        turn_id="demo-turn-1",
        session_id="demo-session",
        vad=VAD(valence=-0.5, arousal=0.6),
    )

    result = attributor.analyze_turn_with_audio(
        audio_path=args.audio,
        asr_text=args.asr_text,
        left_memory_block="",
        emotion_graph_context=None,
        turn=turn,
    )

    print("analysis_text:", result.analysis_text)
    print("emotion:", result.emotion)
    print("acoustic_evidence:", result.acoustic_evidence)
    print("semantic_evidence:", result.semantic_evidence)


if __name__ == "__main__":
    main()
