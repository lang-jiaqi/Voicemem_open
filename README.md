<p align="center">
  <img src="assets/Voicemem_logo.png" alt="VoiceMem Logo" width="100%">
</p>


<p align="center">
  <a href="https://arxiv.org/abs/2605.19833">Technical Report 📖</a> /
  <a href="https://huggingface.co/datasets/zhifeixie/Voices-in-the-Wild-2M">Voices-in-the-wild-2M 🤗</a> /
  <a href="https://huggingface.co/zhifeixie/Mega-ASR">Mega-ASR Weights 🤗</a> /
  <a href="https://github.com/xzf-thu/Voices-in-the-Wild-Bench">Voices-in-the-Wild-Bench 🏆</a>
</p>

<p align="center">
  <a href="https://github.com/xzf-thu/Mega-ASR/raw/main/assets/wechat.jpg"><img src="https://img.shields.io/badge/WeChat-Join%20Group-07C160?logo=wechat&logoColor=white" alt="WeChat"></a> <a href="https://xzf-thu.github.io/Mega-ASR/"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page"></a> <a href="https://x.com/XieZhifei14110"><img src="https://img.shields.io/badge/X-@XieZhifei14110-black?logo=x&logoColor=white" alt="X"></a>
</p>

<p align="cente

---

我们带来 **VoiceMem**，为语音模型增加最后一个组件：灵魂，让它真正越来越懂你。VoiceMem 建立在
「流式双脑」架构之上，提供精准、有情感、懂人格、低延迟且最便宜的记忆服务。快速理解 VoiceMem：

- **左脑：** 直接管理信息，在 top-3 限制下维持 Mem0 的满载性能。
- **右脑：** 用长短期情绪归因管理「情商」，含交叉节点、与左脑信息联合维护。
- **低延迟：** 通过压缩信息、分层存储、流式查询（0–500ms 投机预取），几乎不增加延迟。
- **简单实用：** 单轮查询约 300 token；架构全部解耦，全部组件（含底层记忆引擎）都可更换。
<p align="center">
  <img src="assets/teaser.png" alt="VoiceMem Logo" width="100%">
</p>

## Demo
NOTE: need to unmute first.
<div align="center">
  <video
    src="https://private-user-images.githubusercontent.com/201621992/637588589-34d46638-20db-4943-a88b-b3826c16f156.mp4?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3ODcwNjIwNzIsIm5iZiI6MTc4NzA2MTc3MiwicGF0aCI6Ii8yMDE2MjE5OTIvNjM3NTg4NTg5LTM0ZDQ2NjM4LTIwZGItNDk0My1hODhiLWIzODI2YzE2ZjE1Ni5tcDQ_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwODE4JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDgxOFQxNDAyNTJaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT00ZTdkY2IxYzhiYjk2MzAzMGYxN2MyYjE1YTNjODk1MTMxNWY4ZmRlYzZlNTQzOWM4YzE5YjQ3M2M3MjE0OWQ0JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZyZXNwb25zZS1jb250ZW50LXR5cGU9dmlkZW8lMkZtcDQifQ.GxzWfZKRHMcGRQe4Kc__YKqJrtNSnUcPIFPnPv7n2zQ"
    width="1000"
    controls>
  </video>
</div>

## Overview

* **[Quick Start](#quick-start)** 
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

## Memory system installation
pip install voicemem

## Use VoiceMem Model Families:
pip install "voicemem[slm]"
```

### Required Model Download

```bash
pip install -U huggingface_hub
hf download zhifeixie/VoiceMem_default --local-dir ./models
```


### Basic usage <a id="interfaces"></a>

**Run as an Offline Memory Engine**
```python
from voicemem import VoiceMem

vm = VoiceMem(mode="normal",
              openai_key="api_xxx",
              top_k = 5)          

# 存：音频文件（内部跑ASR/声纹/场景/情绪感知/Embedding抽取）
vm.ingest(audio="input.wav") # 我是素食主义者，对坚果过敏
result = vm.search("我的饮食禁忌是什么？")  
print(result.result_leftbrain, result.result_rightbrain)


# 存：左脑信息文本（无情感）

vm = VoiceMem(mode="leftbrain_only",
              openai_key="api_xxx",
              top_k = 5)             

vm.ingest("我是素食主义者，对坚果过敏。")
result = vm.search("我的饮食禁忌是什么？")     
```

**Run VoiceMem in streaming (Just treat it as VAD)**

```python
import asyncio
from pprint import pprint
import numpy as np
import soundfile as sf



async def main():
    audio, sr = sf.read("speech.wav", dtype="float32")
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    stream = vm.stream(
        src_rate=sr,
        vad_threshold=0.5,
        on_partial=lambda t: print(f"\r[partial] {t}", end="", flush=True),
    )
    step = int(sr * .032)

    for i in range(0, len(pcm), step):
        st = await stream.feed(pcm[i:i + step].tobytes())
        print(f"\n[state] {st.state}")

        FIELDS = ["result_leftbrain", "result_rightbrain", "speaker_id", "speaker_voiceprint", "emotion", "transcript", "entity", "schema", "text_embedding"]
        if st.state == "turn_over":
            pprint({key: getattr(st, key) for key in FIELDS})

asyncio.run(main())
```

### Interactive demo with VoiceMem

```bash
python web/run.py    # http://localhost:8787
```

## VoiceMem: Memory with streaming dual-brain archetecture

**VoiceMem** is a memory system built for real-time voice agents. Instead of treating memory as a single retrieval database, VoiceMem separates it into two complementary brains:

<p align="center">
  <img src="./docs/images/fig-architecture.webp" alt="VoiceMem Logo" width="80%">
</p>



- **Left Brain** organizes factual memory with schemas and entities for precise retrieval.
- **Right Brain** models personality, emotion, and relationships through independent and cross-entity memory nodes.

<p align="center">
  <img src="./docs/images/stages.png" alt="VoiceMem Logo" width="90%">
</p>


The entire pipeline is **streaming**. While the user is still speaking, VoiceMem continuously segments audio, transcribes speech, extracts memory, and writes structured information into the graph. At query time, VoiceMem **routes first, ranks second, and injects only the Top-K memories** into the model context.

### Features

- 🎯 **Accurate** — Get **91.2% on LoCoMo**, vs. **61.68% for Mem0**, with only **Top-5** memories.

- ❤️ **Emotional & Personal** — Remember **who the user is and how they feel**, not just what they said. Reach **69.44% on PersonaMem**.

- 🎧 **Multimodal** — Remember **speech, speakers, sound events, multi-speaker conversations, and music** from real-world audio.

- ⚡ **Fast** — Respond in **134 ms**, vs. **1,440 ms for Mem0**, with streaming retrieval inside the voice turn.

- 💰 **Cheap** — Use only **302 memory tokens**, vs. **6,956 for Mem0** and **1,899 for EverMemOS**.


### VoiceMem Model Families

We build **ChatMem-400K** through a three-stage pipeline: **memory-world construction, SLM-validated online on-policy distillation (OPD), and human refinement**. The same pipeline produces **ChatMem-Bench** for evaluation. Our models including **Qwen2.5-Omni, Qwen3-Omni, and Step-Audio2-Mini** learn to proactively invoke VoiceMem when memory is useful.

<p align="center">
  <img src="./docs/images/fig-opd.webp" alt="VoiceMem Logo" width="90%">
</p>

## Voicechatting with Voiem Model Families

[todo. jiaqi.] 运行，见文件。 / finetune 给指令。


## Evaluation

[todo. jiaqi.] 运行指令。 以上两个参考mega-asr，应该是voicemem外的两个文件夹。 依然，用sb都能看懂的代码，一键运行。




## Acknowledgements

衷心感谢这些出色的开源项目：[mem0](https://github.com/mem0ai/mem0)（底层向量记忆引擎）、
[FunASR](https://github.com/modelscope/FunASR)（流式 ASR paraformer-zh-streaming）、[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)（Silero VAD / 3D-Speaker 声纹 / 回退流式 ASR）、
[intfloat/multilingual-e5](https://huggingface.co/intfloat/multilingual-e5-small)（本地 embedding 与 slot 分类），
以及 OpenAI（chat / TTS / Realtime）。

## License

本项目以 **Apache License 2.0** 开源 — 见 [LICENSE](LICENSE)。用 VoiceMem 尽情构建 🎉
