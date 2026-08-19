#!/usr/bin/env python3
"""演示 VoiceMem 的音频原生感知层——不需要 WebSocket / 浏览器 / 麦克风。

两部分:
  1. perception-only：直接跑场景/声纹/韵律情绪检测，需要
     `pip install -e ".[audio,environment,voiceprint]"`。
  2. full ingest：把音频和转录文本一起交给 VoiceMem.Ingest()，走完整记忆管线
     （需要额外 OPENAI_API_KEY，见 README「Quick start」）。

用法::

    python examples/ingest_audio.py path/to/turn.wav
    python examples/ingest_audio.py path/to/turn.wav --text "刚才说的那句话" --ingest
"""
from __future__ import annotations

import argparse
from pathlib import Path


def run_perception_only(audio_path: Path) -> None:
    from voicemem.utils.audio.emotion.vad_audio import HeuristicWavVADEstimator
    from voicemem.utils.audio.environment.environment_detector_ast import ASTEnvironmentDetector

    vad = HeuristicWavVADEstimator().estimate(str(audio_path))
    print(f"VAD（韵律情绪）:  valence={vad.valence:+.2f}  arousal={vad.arousal:.2f}")

    detector = ASTEnvironmentDetector()
    full = detector.detect_full(audio_path)
    print("背景场景标签:", full["pairs"] or "(无)")
    print("背景音乐/哼唱:", full["music"]["labels"] if full["music"] else "(无)")
    print("异常环境音  :", full["abnormal"] or "(无)")

    try:
        from voicemem.utils.audio.voiceprint.speaker_encoder import SpeakerEncoder

        vec = SpeakerEncoder().embed(audio_path)
        print("声纹向量    :", "提取成功，维度=" + str(vec.shape) if vec is not None else "提取失败")
    except Exception as e:  # funasr/modelscope 未安装时给出明确提示
        print(f"声纹向量    : 跳过（{e}） —— 需要 pip install -e \".[voiceprint]\"")


def run_full_ingest(audio_path: Path, text: str) -> None:
    from voicemem.engine import VoiceMem

    vm = VoiceMem()
    result = vm.Ingest(text, audio_path=str(audio_path))
    print("Ingest 结果:", result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", type=Path, help="wav 文件路径")
    parser.add_argument("--text", default="", help="该音频对应的转录文本（--ingest 时必填）")
    parser.add_argument("--ingest", action="store_true", help="额外跑一遍完整 VoiceMem.Ingest()（需要 OPENAI_API_KEY）")
    args = parser.parse_args()

    if not args.audio.exists():
        raise SystemExit(f"音频文件不存在: {args.audio}")

    print("=== perception-only（无需 LLM）===")
    run_perception_only(args.audio)

    if args.ingest:
        if not args.text:
            raise SystemExit("--ingest 需要同时提供 --text（该音频对应的转录）")
        print("\n=== full ingest（走 VoiceMem 记忆管线）===")
        run_full_ingest(args.audio, args.text)


if __name__ == "__main__":
    main()
