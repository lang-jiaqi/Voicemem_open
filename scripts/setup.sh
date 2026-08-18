#!/usr/bin/env bash
# 一键准备语音 demo 需要的所有依赖 + 模型（一次性，跑完就不用再管）。
# 用法（从仓库根目录）: bash scripts/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."          # 切到仓库根

echo "==> [1/3] 安装依赖（voicemem 核心 + service extra: sherpa-onnx/funasr 等）"
pip install -e ".[service]"

echo "==> [2/3] 下载语音模型（ASR / VAD / 声纹，来自 GitHub，约 570MB）"
bash scripts/download_models.sh service/models

echo "==> [3/3] 预下载记忆嵌入模型 E5（intfloat/multilingual-e5-small，来自 HuggingFace，约 470MB）"
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')"

echo
echo "✅ 全部就绪。之后只需开两个进程即可（无需再下载）："
echo "   后端:  cd openai_voice_demo/backend && OPENAI_API_KEY=sk-... python main.py   # :8787"
echo "   网页:  cd web && OPENAI_API_KEY=sk-... python server.py                        # :8000"
