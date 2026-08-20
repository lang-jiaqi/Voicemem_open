#!/usr/bin/env bash
# 把本地 models/ 下按用途分好的模型传到发布仓库，之后别人用
# download_models.sh 一条命令就能拉全（upload/download 都支持大文件断点续传）。
#
# 先登录一次（二选一）:
#   hf auth login                  # 新版 CLI；旧版是 huggingface-cli login
#   export HF_TOKEN=hf_xxx
#
# 用法（**从仓库根目录**跑，不是 web/ 里）:
#   bash scripts/upload_models.sh                    # 传 vad/asr/speaker/embedding/scene/emotion
#   bash scripts/upload_models.sh models/vad         # 只传某一个目录
#   VOICEMEM_MODELS_REPO=别人/别的repo bash scripts/upload_models.sh
#
# 传之前请确认各模型的许可允许再分发，并在仓库 README 里标明来源与 license。
set -euo pipefail

cd "$(dirname "$0")/.."          # 允许从任何目录调用，路径一律按仓库根算

REPO="${VOICEMEM_MODELS_REPO:-zhifeixie/VoiceMem_default}"
DEST_ROOT="${VOICEMEM_MODELS_ROOT:-models}"

# 不传的两样：voicemem-qwen3.6-*（发布清单，跟着代码仓库走）、slm（另有仓库）
KINDS=(vad asr speaker embedding scene emotion tts)
if [ $# -gt 0 ]; then
  KINDS=()
  for a in "$@"; do KINDS+=("$(basename "$a")"); done
fi

FOUND=()
for k in "${KINDS[@]}"; do
  if [ -d "${DEST_ROOT}/${k}" ] && [ -n "$(ls -A "${DEST_ROOT}/${k}" 2>/dev/null)" ]; then
    FOUND+=("$k")
  else
    echo "跳过 ${k}/（不存在或为空）"
  fi
done

if [ ${#FOUND[@]} -eq 0 ]; then
  echo "没有可传的目录。先跑：bash scripts/download_models.sh"
  exit 1
fi

echo "[HF] 目标仓库 ${REPO}"
echo "[HF] 将上传：${FOUND[*]}"
for k in "${FOUND[@]}"; do
  echo
  echo "── ${k}/ （$(du -sh "${DEST_ROOT}/${k}" | cut -f1)）"
  python3 - "$REPO" "${DEST_ROOT}/${k}" "$k" <<'PY'
import sys
from huggingface_hub import HfApi
repo, src, kind = sys.argv[1], sys.argv[2], sys.argv[3]
api = HfApi()
api.create_repo(repo, repo_type="model", exist_ok=True)
# path_in_repo=kind：仓库里就是一个用途一个文件夹，跟本地 models/ 布局一致，
# 这样 download 下来直接能用，代码不用再做映射。
api.upload_folder(folder_path=src, repo_id=repo, repo_type="model", path_in_repo=kind)
print(f"   ✓ {kind}/")
PY
done

echo
echo "完成。验证：bash scripts/download_models.sh /tmp/vm_models_check"
