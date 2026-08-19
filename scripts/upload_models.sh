#!/usr/bin/env bash
# 首次一次性：把本地已下好的语音模型（ASR / VAD / 声纹）上传到你的 HuggingFace repo，
# 之后大家就能用 download_models.sh 从 HF 断点续传地下。upload_folder 也支持大文件续传。
#
# 先登录一次（二选一）:
#   huggingface-cli login          # 或
#   export HF_TOKEN=hf_xxx
#
# 用法（从仓库根目录）:
#   VOICEMEM_HF_REPO=你的用户名/你的repo bash scripts/upload_models.sh [本地模型目录，默认 service/models]
set -euo pipefail

SRC="${1:-service/models}"
REPO="${VOICEMEM_HF_REPO:-LangJiaqi77/voicemem-speech-models}"

echo "[HF] 上传 $SRC -> $REPO（不存在会自动建 repo）..."
python3 - "$REPO" "$SRC" <<'PY'
import sys
from huggingface_hub import HfApi
repo, src = sys.argv[1], sys.argv[2]
api = HfApi()
api.create_repo(repo, repo_type="model", exist_ok=True)
api.upload_folder(folder_path=src, repo_id=repo, repo_type="model")
print("上传完成:", repo)
PY
