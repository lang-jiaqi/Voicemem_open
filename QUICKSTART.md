# 最快本地启动（语音 + 脑图 demo）

## 0. 一次性准备（只做一次）

```bash
cd voicemem_opensource
export OPENAI_API_KEY=sk-...                # 你的 key
pip install -e ".[web]"                     # 三个入口全套 + demo 服务器
bash scripts/download_models.sh models      # VAD/声纹/回退ASR（k2-fsa 官方 release）
# 默认流式 ASR（FunASR paraformer）和本地 E5 首次运行自动下，无需手动
```

> 只想先用「打字」测试、不下语音模型？跑 `pip install -e ".[text]" fastapi uvicorn` 即可
> （语音识别用不了，但打字通道能跑通记忆 + 脑图 + 回复）。

> 换过 demo？本版把记忆向量换成**本地 384 维 E5**（投机预取的 0–500ms 预算内不能走网络）。
> 若你之前用旧 demo（OpenAI 1536 维）跑过、留下了记忆库，维度不兼容——先清掉记忆目录再跑。

## 1. 启动（一个进程）

```bash
export OPENAI_API_KEY=sk-...
python web/run.py                           # http://localhost:8787
# 想要更快更自然的原生语音： python web/run.py --mode realtime  （需 Realtime API 权限）
# 全部参数：python web/run.py --help
```

## 2. 用

浏览器打开 **http://localhost:8787/**

- **打字**：在「流式输入」框里输入回车 —— 无需麦克风/ASR，只需 OpenAI。
- **语音**：点右下「开始对话」→ 允许麦克风 → 直接说话（需已下模型）。

左脑按 SlotV2 槽实时长节点 + 栏板填充，右脑长情绪/经历/画像节点，中间四块流式更新，AI 语音回复。

## 排查

- 点「开始对话」秒退回待机 / `code=1006` → demo 进程没在跑，确认它打印了 `-> http://localhost:8787/`。
- 「请用 http://localhost:8000 打开」这类提示 → 别双击 html 文件，走 http://localhost:8787。
- 用 Chrome / Edge 最稳。
```
