#!/usr/bin/env bash
# 下载 voicemem 用到的**本地开源**语音模型（VAD / 流式 ASR / 声纹）。
# 默认从打包好的 HF 仓库一次拉全，按用途分目录：
#
#   models/
#     vad/      silero_vad.onnx                                   判「说完了」
#     asr/      sherpa-onnx-streaming-zipformer-bilingual-zh-en…  回退流式 ASR
#     speaker/  3dspeaker_speech_eres2net_base_sv_zh-cn…onnx      声纹
#
# 回复模型不在这里（adapter 在 LangJiaqi77/Voicemem-Qwen3_6-35B-A3B-QLoRA-v2，
# 基座另外取）；默认流式 ASR 是 FunASR paraformer，首次运行自动下，也不用管。
# 完整模型清单见 docs/MODELS.md。
#
# 用法（从仓库根目录）:
#   bash scripts/download_models.sh [目标目录，默认 ./models]
#   VOICEMEM_MODELS_REPO=别的仓库 bash scripts/download_models.sh
#   VOICEMEM_FROM_UPSTREAM=1 bash scripts/download_models.sh   # 改从各家官方源逐个拉
set -euo pipefail

DEST="${1:-models}"
REPO="${VOICEMEM_MODELS_REPO:-zhifeixie/VoiceMem_default}"
mkdir -p "${DEST}"

if [ "${VOICEMEM_FROM_UPSTREAM:-0}" != "1" ]; then
  echo "[1/2] 从 ${REPO} 拉取 VAD / ASR / 声纹 …"
  python3 - "${REPO}" "${DEST}" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, dest = sys.argv[1], sys.argv[2]
snapshot_download(repo_id=repo, local_dir=dest)   # 断点续传，重复跑不会重下
PY
else
  # 逐个从各家官方公开来源拉（HF 不可达、或想核对来源时用）
  REL="https://github.com/k2-fsa/sherpa-onnx/releases/download"
  ASR_DIR="sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"
  mkdir -p "${DEST}/vad" "${DEST}/asr" "${DEST}/speaker"

  echo "[1/2] 官方源：VAD silero（MIT）…"
  curl -L -o "${DEST}/vad/silero_vad.onnx" "${REL}/asr-models/silero_vad.onnx"

  echo "      官方源：回退流式 ASR ${ASR_DIR}（Apache-2.0, k2-fsa）…"
  [ -d "${DEST}/asr/${ASR_DIR}" ] || curl -L "${REL}/asr-models/${ASR_DIR}.tar.bz2" | tar xj -C "${DEST}/asr"

  # 注意官方 release tag 拼写就是 recongition
  echo "      官方源：声纹 3D-Speaker ERes2Net（Apache-2.0）…"
  curl -L -o "${DEST}/speaker/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx" \
    "${REL}/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
fi

# 本地记忆/投机用的 E5（intfloat/multilingual-e5-small，MIT）——预拉进 HF 缓存，离线可跑
E5_REPO="${VOICEMEM_E5_REPO:-intfloat/multilingual-e5-small}"
echo "[2/2] 本地 E5 ${E5_REPO} …"
python3 - "${E5_REPO}" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1])   # 默认缓存目录，断点续传
PY

echo
echo "完成：语音模型在 ${DEST}/（vad / asr / speaker），E5 在 HF 缓存。"
echo "其余能力（AST 场景 / SenseVoice / Qwen-Omni）首次用时 transformers 自动下——见 docs/MODELS.md。"
