# 最快本地启动（语音 + 脑图 web demo）

## 0. 一次性准备（只做一次，跑完不用再管）

```bash
cd voicemem_opensource
export OPENAI_API_KEY=sk-...          # 你的 key
bash scripts/setup.sh                 # 装依赖 + 下所有模型（ASR/VAD/声纹/E5，约 1GB，一次）
```

> 只想先用「打字」测试、不下 1GB 模型？跳过 `setup.sh`，直接 `pip install -e .` 即可
> （语音识别用不了，但打字通道能跑通记忆+脑图+回复）。

## 1. 每次启动：开两个终端

**终端 A — 语音后端（:8787）**
```bash
cd openai_voice_demo/backend
export OPENAI_API_KEY=sk-...
export VOICEMEM_SKIP_STARTUP_CHECK=1     # 跳过启动自检（否则组件偏慢时会停在 y/N 提示挡住启动）
python main.py            # 等到打印 "Uvicorn running on http://0.0.0.0:8787"
```

**终端 B — 网页（:8000）**
```bash
cd web
export OPENAI_API_KEY=sk-...
python server.py          # 打印 "voicemem demo -> http://localhost:8000/"
```

## 2. 用

浏览器打开 **http://localhost:8000/**

- **打字**：在「流式输入」框里输入回车 —— 无需麦克风/ASR，只需 OpenAI。
- **语音**：点右下「开始对话」→ 允许麦克风 → 直接说话（需已跑过 `setup.sh`）。

左脑按 SlotV2 槽实时长节点 + 栏板填充，右脑长情绪/经历/偏好节点，中间四块流式更新，AI 语音回复。

## 排查

- 点「开始对话」秒退回待机 / 状态显示 `code=1006` → **终端 A 的后端没在跑**。确认它打印了 `Uvicorn running on ...:8787` 且没退出。
- 后端卡住不动、最后是一张「启动自检」表格 → 它在等你按 `y/N`。按 `y` 回车放行，或用上面的 `VOICEMEM_SKIP_STARTUP_CHECK=1` 直接跳过。
- 状态显示「请用 http://localhost:8000 打开」→ 别双击 html 文件，要走 :8000。
- 想连别的机器/端口的后端：`http://localhost:8000/?ws=主机:端口`。
- 用 Chrome / Edge 最稳。
