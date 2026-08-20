<p align="center">
  <img src="assets/Voicemem_logo.png" alt="VoiceMem Logo" width="100%">
</p>

---

<p align="center">
  <a href="#quick-start">Quick Start 🚀</a> /
  <a href="#architecture">Architecture 🧠</a> /
  <a href="QUICKSTART.md">语音 Demo 🎙️</a>
</p>

<p align="center">
  <a href="https://huggingface.co/LangJiaqi77/Voicemem-Qwen3_6-35B-A3B-QLoRA-v2"><img src="https://img.shields.io/badge/HuggingFace-Model-FFD21E?logo=huggingface&logoColor=black" alt="HF Model"></a>&nbsp;<a href="https://lang-jiaqi.github.io/Voicemem_open/"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page"></a>&nbsp;<img src="https://img.shields.io/badge/License-Apache%202.0-green" alt="License">&nbsp;<a href="https://github.com/lang-jiaqi/Voicemem_open"><img src="https://img.shields.io/github/stars/lang-jiaqi/Voicemem_open?style=social" alt="GitHub stars"></a>&nbsp;<a href="#"><img src="https://img.shields.io/badge/WeChat-Join%20Group-07C160?logo=wechat&logoColor=white" alt="WeChat"></a>&nbsp;<a href="#"><img src="https://img.shields.io/badge/X-VoiceMem-black?logo=x&logoColor=white" alt="X"></a>
</p>

---

我们带来 **VoiceMem**，为语音模型增加最后一个组件：灵魂，让它真正越来越懂你。VoiceMem 建立在
「流式双脑」架构之上，提供精准、有情感、懂人格、低延迟且最便宜的记忆服务。快速理解 VoiceMem：

- **左脑：** 直接管理信息，在 top-3 限制下维持 Mem0 的满载性能。
- **右脑：** 用长短期情绪归因管理「情商」，含交叉节点、与左脑信息联合维护。
- **低延迟：** 通过压缩信息、分层存储、流式查询（0–500ms 投机预取），几乎不增加延迟。
- **简单实用：** 单轮查询约 300 token；架构全部解耦，全部组件（含底层记忆引擎）都可更换。

---

## Overview

* **[Quick Start](#quick-start)** —— 安装 · [三种输入接口](#interfaces) · 跑 demo
* **[Architecture](#architecture)**
* **[Models](#models)**
* **[Demo](#demo)**
* **[Released Model](#released-model)**
* **[Acknowledgements](#acknowledgements)**
* **[License](#license)**

## Quick Start

### 安装

```bash
git clone https://github.com/lang-jiaqi/Voicemem_open.git
cd Voicemem_open

pip install -e ".[demo,audio,environment,voiceprint]"     # 记忆核心 + 音频感知 + 流式
bash scripts/download_models.sh models                    # VAD / 声纹 / 回退 ASR + 本地 E5
export OPENAI_API_KEY=sk-...
```

> 只想先用纯文本（下面的 ①）？`pip install -e ".[demo]"` 就够，一个模型都不用下。

### 默认配置

装完不用配任何东西，开箱就是这一套：

| 能力 | 默认 | 跑在哪 |
|---|---|---|
| 流式 ASR | FunASR `paraformer-zh-streaming` | 本地，首次运行自动下 |
| VAD（判「说完了」） | silero | 本地，`download_models.sh` 下 |
| 记忆向量 | `text-embedding-3-small` | OpenAI API |
| slot 分类 | 本地 E5 余弦（0 LLM） | 本地 |
| 回复 | `gpt-4o-mini` | OpenAI API |
| 声纹 / 声学场景 / 情绪 | 3D-Speaker / AST / 声学启发式 | 本地 |

每一项都能换成本地模型或你自己的——但**先按默认跑通再说**，换法见 [Models](#models)。

### 三种输入接口 <a id="interfaces"></a>

文本 / wav / 流式，三者在核心并列。① 复制就能跑；② ③ 把 `turn.wav` / `speech.wav`
换成你自己的音频（16k mono 最省事，其它采样率 `src_rate` 会自动重采样）。

**① 文本**

```python
from voicemem import VoiceMem

vm = VoiceMem(mode="text_mode")               # 读上面 export 的 OPENAI_API_KEY

vm.ingest("我是素食主义者，对坚果过敏。")
result = vm.search("我的饮食禁忌是什么？")     # 左右脑一起检索
for hit in result.hits:
    print(hit.text)
```

**② 语音（wav）**

```python
from voicemem import VoiceMem

vm = VoiceMem(mode="multi_modal")             # 读上面 export 的 OPENAI_API_KEY

# 存：上游转好的文本 + 音频文件（内部跑声纹/场景/情绪感知）
vm.ingest("今天在咖啡馆和 Alex 聊了创业。", audio="turn.wav")

# 只拿声学信号、不写记忆
sig = vm.preprocess("今天在咖啡馆和 Alex 聊了创业。", audio="turn.wav")
print(sig.speaker, sig.emotion, sig.scene_tag)      # 说话人 / 情绪 / 声学场景

for hit in vm.search("我和 Alex 聊了什么？").hits:
    print(hit.text)
```

**③ 流式（逐块喂音频）**

```python
import asyncio
import numpy as np
import soundfile as sf
from voicemem import VoiceMem

# 统一 config：本地 E5，0 网络
vm = VoiceMem.from_config({
    "mode": "multi_modal",
    "embedding": {"provider": "local"},
    "slots":     {"provider": "local"},
})

async def main():
    audio, sr = sf.read("speech.wav", dtype="float32")     # mono
    pcm16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)

    # 边喂边出 partial；VAD 判说完时拿到一轮记忆结果（0–500ms 投机预取、barge-in）
    stream = vm.stream(on_partial=lambda t: print(f"\r{t}", end="", flush=True), src_rate=sr)

    step = int(sr * 0.6)                                   # 600ms 一块
    for i in range(0, len(pcm16), step):
        st = await stream.feed(pcm16[i:i+step].tobytes())  # 每块都返回 StreamState
        # st.state  "<speak>" | "<silence>"
        # st.memory 边说边预取到的 SearchResult；还没算好时 None
        if st.turn:                                        # 一轮说完才非 None
            print("\n[说完]", st.turn.text)
            for hit in st.turn.result.hits:
                print("  记忆:", hit.text)

asyncio.run(main())
```

> **返回类型**：`feed` / `feed_partial` 每块都返回 **`StreamState`**，一轮说完时
> `st.turn` 才是 `Turn`（`.text` / `.result` / `.memory_context`）；`feed_text("…")`
> 是打字轮，直接返回 `Turn`。
>
> 接**外部 ASR**（FunASR / Whisper / 云 ASR）就把 ③ 的 `feed` 换成
> `await stream.feed_partial(累积转写, ended=外部VAD判说完)`——换 ASR 只改喂进来的
> 那一行。完整示例见 [`scripts/realtime_funasr_qwen.py`](scripts/realtime_funasr_qwen.py)。

### 语音 Demo（脑图 + 0–500ms 投机预取）

```bash
cd web && DEMO_MODE=llm_tts python run.py     # http://localhost:8787
# 想要更自然的原生语音： DEMO_MODE=realtime python run.py  （需 Realtime API 权限）
```
详见 [QUICKSTART.md](QUICKSTART.md)。

## Architecture

VoiceMem 是**一层薄门面 + 三个自包含组件**（组合式，参考 [mem0](https://github.com/mem0ai/mem0)）——
打开 `core.py` 就一眼看懂整个系统，重逻辑都藏在组件里、依赖显式注入、可整块替换：

```
core.py            VoiceMem 门面（对外 ~70 行）
  ├─ leftbrain/    LeftBrain      事实记忆：实体 + 认知图（slot 分类/检索），底层 mem0 向量库
  ├─ rightbrain/   RightBrain     情绪记忆：valence-arousal、长短期情绪归因、人格画像、交叉节点
  ├─ utils/audio/  AudioPerceiver 音频原生感知：声纹 / 声学场景 / 情绪 / 音乐（直接吃波形）
  └─ stream.py     VoiceStream    流式输入：本地 ASR + VAD + 0–500ms 投机预取
orchestrator.py    编排实现（把三组件串成 Search/Ingest pipeline）
```

- **组件可换**：`embedding` / `schema`(分类器) / `memory_engine`(默认 mem0) 等每个能力都有内置默认，
  传一个函数就换成自己的（本地模型、别的向量库…）。
- **读写分离**：抽取事实、更新摘要、刷新描述等「慢而智能」的 LLM 活儿都在写入侧；读（检索）路径
  0 次 LLM、走本地向量，实测 Search 本体 ~10ms。

## 主业务逻辑：实时对话一轮

麦克风逐帧喂进 `vm.stream()`，说完一轮就拿到 `Turn`——**记忆早在你说话时就查好了**，
关键路径上只剩回复模型。这就是全部：

```python
import asyncio, queue
import numpy as np, sounddevice as sd
from voicemem import VoiceMem

vm = VoiceMem.from_config({                     # 检索侧全本地 → 投机预取 0 网络
    "embedding": {"provider": "local"},
    "slots":     {"provider": "local"},
})
stream = vm.stream(on_partial=lambda t: print(f"\r🎙️  {t}", end="", flush=True),
                   src_rate=16000)

mic_q = queue.Queue()                           # callback 只丢数据，不被生成阻塞
def on_mic(indata, *_):
    mic_q.put((np.clip(indata[:, 0], -1, 1) * 32767).astype(np.int16).tobytes())

async def main():
    loop = asyncio.get_running_loop()
    with sd.InputStream(samplerate=16000, channels=1, dtype="float32",
                        blocksize=320, callback=on_mic):        # 20ms 一帧
        while True:
            st = await stream.feed(await loop.run_in_executor(None, mic_q.get))
            if st.turn:                                          # VAD 确认说完
                answer = await vm.reply(st.turn)                 # 记忆已就绪，直接生成
                print(f"\n🧑 {st.turn.text}\n🤖 {answer}")
                vm.ingest(st.turn.text, async_facts=True)        # 存这轮，抽事实走后台

asyncio.run(main())
```

## Models

每个能力都**可插拔**——本地开源 ↔ API，一行 config 就切换。完整清单（哪个功能有哪些模型选项）见
**[docs/MODELS.md](docs/MODELS.md)**。速览：

- **语音感知**(ASR / VAD / 声纹 / 声学场景) = **纯本地开源**（`bash scripts/download_models.sh models` 从官方拉）;
- **记忆**(embedding / slot 分类 / 事实抽取) = **默认 OpenAI API，但都有本地开源替代**（E5 / 本地分类器 / 本地 LLM）;
- **回复**(对话 LLM) = **核心能力**，内置 OpenAI 兼容 provider，`VoiceMem(reply=fn)` 换自己的；
  TTS / Realtime 仍在 demo 层（TTS 可本地可 API，Realtime 目前仅 OpenAI）。

```python
# 一个 dict 配齐：记忆全本地（0 网络），VAD 调阈值
vm = VoiceMem.from_config({
    "embedding": {"provider": "local"},          # 本地 E5
    "slots":     {"provider": "local"},          # 本地分类器，0 LLM
    "vad":       {"provider": "silero", "config": {"threshold": 0.6}},
})
vm = VoiceMem(reply=my_fn, vad=lambda: MyVad())  # 或者直接注入函数/对象
```

**哪些模型必须下。** `download_models.sh` 下四样，只有 VAD 那份没有自动下载兜底：

| 下的东西 | 谁在用 | 不下行不行 |
|---|---|---|
| `silero_vad.onnx` | `vm.stream()` 判「说完了」 | 不下就得注入自己的 VAD（`VoiceMem(vad=…)`） |
| sherpa 回退 ASR | 只在 `VOICEMEM_ASR=sherpa` 时 | 行，默认 ASR 首次运行自动下 |
| 3D-Speaker 声纹 | `multi_modal` 的声纹识别 | 行，不开声纹就用不到 |
| 本地 E5 | `provider: "local"` 的 embedding / slots | 行，首次运行自动下；这步只是预拉方便离线 |

ASR 还能整个换掉：`VOICEMEM_ASR=sherpa` 切回 sherpa-onnx，或用
[`feed_partial`](#interfaces) 接任意外部 ASR（Whisper / 云 ASR）。

## Demo

**[`web/`](web/)** —— 单一极简 demo：`run.py`（对话核心）+ `utils.py`（管道）+ 脑图 `index.html`。
本地 ASR + VAD 边听边算，VAD 确认说完时记忆早已在关键路径外投机预取好，两条回复控制流
（`llm_tts` = GPT 流→TTS 流 / `realtime` = OpenAI Realtime 原生语音）拿去回复。WebSocket 协议见
[docs/PROTOCOL.md](docs/PROTOCOL.md)。

## Released Model

VoiceMem 记忆工作流微调的 QLoRA adapter：
[**LangJiaqi77/Voicemem-Qwen3_6-35B-A3B-QLoRA-v2**](https://huggingface.co/LangJiaqi77/Voicemem-Qwen3_6-35B-A3B-QLoRA-v2)
（adapter-only，基座 [`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)；
加载代码、校验、评测见 [`models/voicemem-qwen3.6-35b-a3b-qlora-v2/`](models/voicemem-qwen3.6-35b-a3b-qlora-v2/)）。

## Acknowledgements

衷心感谢这些出色的开源项目：[mem0](https://github.com/mem0ai/mem0)（底层向量记忆引擎）、
[FunASR](https://github.com/modelscope/FunASR)（流式 ASR paraformer-zh-streaming）、[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)（Silero VAD / 3D-Speaker 声纹 / 回退流式 ASR）、
[intfloat/multilingual-e5](https://huggingface.co/intfloat/multilingual-e5-small)（本地 embedding 与 slot 分类），
以及 OpenAI（chat / TTS / Realtime）。

## License

本项目以 **Apache License 2.0** 开源 — 见 [LICENSE](LICENSE)。用 VoiceMem 尽情构建 🎉
