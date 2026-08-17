#!/usr/bin/env bash
# 下载语音服务需要的三个模型文件，落到 models/ 下（不进 git，首次跑一次就够）。
# 用法: bash download_models.sh [目标目录，默认 ./models]
set -euo pipefail

DEST="${1:-models}"
mkdir -p "$DEST"
cd "$DEST"

echo "[1/3] streaming ASR (sherpa-onnx, 中英双语 zipformer)..."
ASR_NAME="sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"
if [ ! -d "$ASR_NAME" ]; then
    wget -q --show-progress \
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/${ASR_NAME}.tar.bz2"
    tar xf "${ASR_NAME}.tar.bz2"
    rm "${ASR_NAME}.tar.bz2"
else
    echo "  已存在，跳过"
fi

echo "[2/3] 说话人识别 (3D-Speaker ERes2Net, ONNX)..."
SPK_FILE="3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
if [ ! -f "$SPK_FILE" ]; then
    wget -q --show-progress \
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/${SPK_FILE}"
else
    echo "  已存在，跳过"
fi

echo "[3/3] VAD (Silero VAD, ONNX)..."
VAD_FILE="silero_vad.onnx"
if [ ! -f "$VAD_FILE" ]; then
    wget -q --show-progress \
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/${VAD_FILE}"
else
    echo "  已存在，跳过"
fi

echo
echo "完成，模型都在: $(pwd)"
echo "（声纹（3D-Speaker ERes2Net）权重已在上面第 [2/3] 步下载好，voicemem 核心包的"
echo " speaker_encoder.py / campplus_worker.py 默认也读这个目录，无需再手动下载）"
