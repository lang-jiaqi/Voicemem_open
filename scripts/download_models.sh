#!/usr/bin/env bash
# 从 HuggingFace 下载语音模型（ASR / VAD / 声纹）。用 huggingface_hub，自带**断点续传**，
# 中断后重复运行会接着下、已下好的不重下。
#
# 用法（从仓库根目录）:
#   bash scripts/download_models.sh [目标目录，默认 ./models]
#
# 换成你自己的 repo:
#   VOICEMEM_HF_REPO=你的用户名/你的repo bash scripts/download_models.sh service/models
# 国内加速（镜像）:
#   HF_ENDPOINT=https://hf-mirror.com bash scripts/download_models.sh service/models
set -euo pipefail

DEST="${1:-models}"
REPO="${VOICEMEM_HF_REPO:-LangJiaqi77/voicemem-speech-models}"
mkdir -p "$DEST"

echo "[HF] 下载 ${REPO} -> ${DEST}（断点续传，可随时中断/重跑）..."
python3 - "$REPO" "$DEST" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, dest = sys.argv[1], sys.argv[2]
snapshot_download(repo_id=repo, local_dir=dest)   # 默认断点续传
PY

# 本地记忆/投机模型 E5：demo 的 0–500ms 投机预取用它做本地 embedding + slot 分类
# （见 demo/local.py）。sentence-transformers 按名字从 HF 加载、命中默认缓存，所以
# 这里预拉进 HF 缓存，user 一次 download_models.sh 就把 sherpa + E5 全拉齐、离线可跑。
E5_REPO="${VOICEMEM_E5_REPO:-intfloat/multilingual-e5-small}"
echo
echo "[HF] 预拉本地记忆/投机模型 ${E5_REPO}（进 HF 缓存，sentence-transformers 会命中）..."
python3 - "$E5_REPO" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1])   # 默认缓存目录，断点续传
PY

echo
echo "完成：sherpa 语音模型在 ${DEST}；E5 记忆模型在 HF 缓存。"
echo "（声纹 3D-Speaker 权重也在 ${DEST} 里；voicemem 的 speaker_encoder.py 默认读这个目录）"
