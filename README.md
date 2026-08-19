<p align="center">
  <img src="assets/Voicemem_logo.png" alt="VoiceMem Logo" width="100%">
</p>

---

<p align="center">
  <a href="https://huggingface.co/LangJiaqi77/Voicemem-Qwen3_6-35B-A3B-QLoRA-v2">Model (QLoRA adapter) 🤗</a> /
  <a href="#quick-start">Quick Start 🚀</a> /
  <a href="#architecture">Architecture 🧠</a> /
  <a href="QUICKSTART.md">语音 Demo 🎙️</a>
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

* **[Quick Start](#quick-start)**
* **[Architecture](#architecture)**
* **[三种输入接口](#interfaces)**
* **[Demo](#demo)**
* **[Released Model](#released-model)**
* **[Acknowledgements](#acknowledgements)**
* **[License](#license)**

## Quick Start

**安装**
```bash
git clone https://github.com/lang-jiaqi/Voicemem_open.git
cd Voicemem_open

pip install -e ".[demo,audio,environment,voiceprint]"     # 记忆核心 + 音频感知 + 流式
bash scripts/download_models.sh models                    # sherpa ASR/VAD/声纹 + 本地 E5（HuggingFace）
export OPENAI_API_KEY=sk-...
```

> 只想纯文本试用、不下语音模型？`pip install -e ".[demo]"` 即可（打字通道能跑通记忆 + 回复）。

**三行上手（文本）**
```python
from voicemem import VoiceMem

vm = VoiceMem(api_key="sk-...", mode="text_mode")
vm.ingest("中午和 Alex 吃了拉面")
vm.search("我中午吃了什么？")                 # 左右脑一起检索
vm.left_brain.search("我中午吃了什么？")       # 只要左脑事实、更快
```

**语音 Demo（脑图 + 0–500ms 投机预取）**
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

## 三种输入接口 <a id="interfaces"></a>

文本 / wav / 流式，三者在核心并列：

```python
# ① 文本
vm.ingest("中午吃了拉面");   vm.search("我中午吃了什么")

# ② wav（+ 声学感知：声纹/场景/情绪）
vm.ingest(text, audio="turn.wav")
sig = vm.preprocess(text, audio="turn.wav")     # 只拿声学信号、不写记忆

# ③ 流式（喂音频块 → 说完得到一轮记忆结果）
stream = vm.stream(on_partial=cb)
turn = await stream.feed(pcm_chunk)             # None（还在说）或 Turn(text, result=SearchResult)
turn = await stream.feed_text("我在哪工作")       # 打字轮
turn.text / turn.result / turn.memory_context
```

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
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)（流式 ASR / Silero VAD / 3D-Speaker 声纹）、
[intfloat/multilingual-e5](https://huggingface.co/intfloat/multilingual-e5-small)（本地 embedding 与 slot 分类），
以及 OpenAI（chat / TTS / Realtime）。

## License

**MIT** — 见 [LICENSE](LICENSE)。你可以用 VoiceMem 做任何事 🎉
