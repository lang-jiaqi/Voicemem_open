"""本地模型目录的唯一解析入口。

在这之前三处各写各的：``defaults.py`` 用 ``"models"``（cwd 相对）、``stream_io.py`` 用
``"../models"``（假设从 web/ 起）、``campplus_worker.py`` 从 ``__file__`` 往上数——而且
数错了层数，指到了 ``voicemem/utils/models/``（不存在）。统一到这里。
"""
from __future__ import annotations

import os
from pathlib import Path


def models_dir() -> Path:
    """``VOICEMEM_MODELS_DIR`` > 仓库根的 ``models/`` > cwd 下的 ``models/``。"""
    env = os.environ.get("VOICEMEM_MODELS_DIR")
    if env:
        return Path(env)
    repo = Path(__file__).resolve().parents[3] / "models"   # utils/common/paths.py → 仓库根
    return repo if repo.is_dir() else Path("models")


def model_path(name: str, env_override: str | None = None) -> Path:
    """取一个具体模型文件的路径；``env_override`` 指定的环境变量优先级最高。"""
    if env_override:
        explicit = os.environ.get(env_override)
        if explicit:
            return Path(explicit)
    return models_dir() / name


def require(path: Path, what: str, how: str = "bash scripts/download_models.sh models") -> Path:
    """模型文件不在就报一句人能看懂的话，而不是让底层库抛个看不懂的错。"""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{what} 找不到：{path}\n"
            f"下载：{how}\n"
            f"或用 VOICEMEM_MODELS_DIR 指向你已有的模型目录。"
        )
    return Path(path)
