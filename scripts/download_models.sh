#!/usr/bin/env bash
# 下载 voicemem 用到的**本地开源**语音模型（流式 ASR / VAD / 声纹），
# 从各自官方公开来源拉取——无需 token、任何人 clone 都能用。
# 完整模型清单（每个能力有哪些本地/API 选项）见 docs/MODELS.md。
#
# 用法（从仓库根目录）:
#   bash scripts/download_models.sh [目标目录，默认 ./models]
set -euo pipefail

DEST="${1:-models}"
mkdir -p "${DEST}"

REL="https://github.com/k2-fsa/sherpa-onnx/releases/download"
ASR_DIR="sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"

# 1) 流式 ASR：sherpa-onnx streaming zipformer bilingual zh-en（Apache-2.0，k2-fsa）
if [ ! -d "${DEST}/${ASR_DIR}" ]; then
  echo "[1/4] 流式 ASR ${ASR_DIR} …"
  curl -L "${REL}/asr-models/${ASR_DIR}.tar.bz2" | tar xj -C "${DEST}"
fi

# 2) VAD：silero（MIT）
echo "[2/4] VAD silero_vad.onnx …"
curl -L -o "${DEST}/silero_vad.onnx" "${REL}/asr-models/silero_vad.onnx"

# 3) 声纹：3D-Speaker ERes2Net（Apache-2.0）。注意官方 release tag 拼写是 recongition。
echo "[3/4] 声纹 3D-Speaker …"
curl -L -o "${DEST}/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx" \
  "${REL}/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"

# 4) 本地记忆/投机模型 E5（intfloat/multilingual-e5-small，MIT）——预拉进 HF 缓存，离线可跑。
E5_REPO="${VOICEMEM_E5_REPO:-intfloat/multilingual-e5-small}"
echo "[4/4] 本地 E5 ${E5_REPO} …"
python3 - "${E5_REPO}" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1])   # 默认缓存目录，断点续传
PY

echo
echo "完成：语音模型在 ${DEST}/，E5 在 HF 缓存。"
echo "其余能力（AST 场景 / SenseVoice / Qwen-Omni）首次用时 transformers 自动下——见 docs/MODELS.md。"
