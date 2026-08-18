# PyInstaller spec — 把 openai_voice_demo 后端冻结成自包含可执行文件。
#
# 产物：dist/voicemem-backend/（onedir，含 voicemem-backend 可执行文件），
# 之后由 electron-builder 通过 extraResources 整个拷进 .app。
#
# 构建（在装好 voicemem + 后端所有依赖的那个 Python 环境里，从 backend/ 目录）：
#     pip install pyinstaller
#     pyinstaller voicemem_backend.spec --noconfirm
#
# ⚠️ 这是一个**起点**，不是保证一次成功的配置。voicemem 依赖 torch /
#    sherpa-onnx / funasr / transformers / mem0 / qdrant 等，PyInstaller 常会漏
#    收动态导入的子模块或数据文件。构建后**一定要实际运行 dist 里的可执行文件**，
#    根据 "ModuleNotFoundError / FileNotFoundError" 往下面 hiddenimports /
#    collect_all 里继续补，直到能正常起 uvicorn。这一步只能在你的 Mac 上迭代。

from PyInstaller.utils.hooks import collect_all, collect_submodules

datas, binaries, hiddenimports = [], [], []

# 整包收集（代码 + 数据 + 动态库）——这些库都有大量动态导入 / 附带模型配置文件。
for pkg in [
    "voicemem", "fastapi", "starlette", "uvicorn", "pydantic",
    "sherpa_onnx", "funasr", "transformers", "tokenizers",
    "mem0", "qdrant_client", "openai", "numpy",
]:
    try:
        d, b, h = collect_all(pkg)
        datas += d; binaries += b; hiddenimports += h
    except Exception as e:            # 该环境没装某个可选包就跳过
        print(f"[spec] skip collect_all({pkg}): {e}")

# 后端自己的本地模块（backend/ 下的同级模块 + providers 子包）。
hiddenimports += [
    "config", "memory_bridge", "audio_utils",
    "local_asr", "local_embedder", "local_classifier", "local_emotion_classifier",
]
hiddenimports += collect_submodules("providers")

# 后端用 StaticFiles 伺服 ../frontend/ 下的 orb.html / index.html，一并打包进去。
datas += [("../frontend", "frontend")]

block_cipher = None

a = Analysis(
    ["main.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter"],
    cipher=block_cipher,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="voicemem-backend",
    console=True,            # 保留 stdout，main.js 靠 "Uvicorn running on" 判就绪
    disable_windowed_traceback=False,
)
coll = COLLECT(
    exe, a.binaries, a.zipfiles, a.datas,
    strip=False, upx=False, name="voicemem-backend",
)
