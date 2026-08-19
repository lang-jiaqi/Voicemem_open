<p align="center">
  <img src="assets/images/logo.jpg" alt="VoiceMem Logo" width="100%">
</p>

---

<p align="center">
  <a href="https://arxiv.org/abs/2605.19833">Technical Report 📖</a> /
  <a href="https://huggingface.co/datasets/zhifeixie/Voices-in-the-Wild-2M">Voices-in-the-wild-2M 🤗</a> /
  <a href="https://huggingface.co/zhifeixie/Mega-ASR">Mega-ASR Weights 🤗</a> /
  <a href="https://github.com/xzf-thu/Voices-in-the-Wild-Bench">Voices-in-the-Wild-Bench 🏆</a>
</p>

<p align="center">
  <a href="https://github.com/xzf-thu/Mega-ASR/raw/main/assets/wechat.jpg"><img src="https://img.shields.io/badge/WeChat-Join%20Group-07C160?logo=wechat&logoColor=white" alt="WeChat"></a>&nbsp;<a href="https://xzf-thu.github.io/Mega-ASR/"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page"></a>&nbsp;<a href="https://x.com/XieZhifei14110"><img src="https://img.shields.io/badge/X-@XieZhifei14110-black?logo=x&logoColor=white" alt="X"></a>
</p>

 
我们带来 **VoiceMem**，为语音模型增加最后一个组件：灵魂，让他真正越来越懂你。VoiceMem 建立在「流式双脑」架构之上，提供精准、有情感、懂人格、低延迟且最便宜的记忆服务。快速理解 VoiceMem：
 
- **左脑：** 直接地管理信息，在 top-3 限制下维持 Mem0 的满载性能。
- **右脑：** 用长短期情绪归因管理「情商」，并包含交叉节点和左脑信息联合维护。
- **低延迟：ß** 通过压缩信息、分层存储和流式查询机制，几乎不增加延迟。
- **简单实用：** 单轮查询约 300 token，架构全部解耦，全部组件包括底层记忆引擎可更换。

<p align="center">
  <img src="assets/images/teaser.png" alt="VoiceMem Logo" width="100%">
</p>

---

## Demo


<div align="center">
  <video
    src="https://private-user-images.githubusercontent.com/201621992/637588589-34d46638-20db-4943-a88b-b3826c16f156.mp4?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3ODcwNjIwNzIsIm5iZiI6MTc4NzA2MTc3MiwicGF0aCI6Ii8yMDE2MjE5OTIvNjM3NTg4NTg5LTM0ZDQ2NjM4LTIwZGItNDk0My1hODhiLWIzODI2YzE2ZjE1Ni5tcDQ_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwODE4JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDgxOFQxNDAyNTJaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT00ZTdkY2IxYzhiYjk2MzAzMGYxN2MyYjE1YTNjODk1MTMxNWY4ZmRlYzZlNTQzOWM4YzE5YjQ3M2M3MjE0OWQ0JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZyZXNwb25zZS1jb250ZW50LXR5cGU9dmlkZW8lMkZtcDQifQ.GxzWfZKRHMcGRQe4Kc__YKqJrtNSnUcPIFPnPv7n2zQ"
    width="800"
    controls>
  </video>
</div>
</div>

## 🔥News

- [Coming]: We are going to release RL code and optimize WebUI.
- [Coming]: Dataset and benchmark will be reformatted to be clearer.
- [Coming]: We will release all the data process pipeline.
- **May 28, 2026**: 🔥 We add Mega-ASR vLLM streaming inference support and a long-form streaming demo audio.
- **May 20, 2026**: 🔥 We release **Voices-in-the-Wild-Bench**, a benchmark for in-the-wild ASR robustness evaluation.
- **May 20, 2026**: 🔥 We release **Voices-in-the-Wild-2M**.
- **May 20, 2026**: 🔥 We release the **Mega-ASR Inference and Training Codebase**.
- **May 19, 2026**: 🔥 **Mega-ASR** model weights are now available on Hugging Face.
- **May 19, 2026**: 🔥 We release the **Mega-ASR Technical Report**.

## Overview


* **[Quick Start](#quick-start)**
* **[Introduction](#inference)**
* **[Inference and deployment](#inference)**
* **[Finetuning](#finetune)**
* **[Evaluation](#evaluation)**
* **[Citation and licence](#citation)**

## Quick Start

Mega-ASR is trained on a large volume of inherently high-WER data, which leads to a slight degradation in its basic recognition capability. To address this, **we equip the system with a router** that determines whether Mega-ASR should be activated for the current audio input, via deciding whether to mount the LoRA weights.


**Installation**
```bash
git clone https://github
.com/xzf-thu/Mega-ASR.git
cd Mega-ASR

conda create -n mega-asr python=3.10 -y
conda activate mega-asr
pip install -r requirements.txt
```

**Information Only**
```bash
import voicemem

vm = voicemem.VoiceMem(
    api_key="sk-...",
    mode="text_mode",)

# 存文本
vm.ingest("中午和 Alex 吃了拉面")

# 也可以直接走总搜索
result = vm.search("我中午吃了什么？")

# 取文本：可以指定左脑事实，进一步提速
hits = vm.left_brain.search("我中午吃了什么？")
```

**Dual brain, text only**
```bash
# 存音频
result = vm.ingest(               
    audio="recordings/lunch.wav",
)

# 查音频 "我中午吃了什么？[疑问]"
hits = vm.search("recordings/query.wav")
```


Mega-ASR's default Transformers backend dynamically mounts and unmounts LoRA
deltas inside one PyTorch model. vLLM manages model weights inside its own
engine, so this vLLM entrypoint materializes LoRA into a normal checkpoint
before engine startup instead of doing router-based dynamic switching.
Qwen3-ASR streaming inference is available only through the official vLLM
backend and does not support batch inference or timestamps.

Mega-ASR also provides a vLLM streaming entrypoint. It uses the same
materialized LoRA checkpoint cache as `infer_vllm.py`, then feeds audio chunks
into Qwen3-ASR's streaming API:

```bash
python infer_vllm_streaming.py \
  --audio assets/example/streaming_long_example.wav \
  --step_ms 1000 \
  --reset_interval_sec 120 \
  --overlap_sec 2 \
  --max_new_tokens 32
```

The script prints partial text after each streaming call and a final transcript
after `finish_streaming_transcribe`. For long audio, it periodically finishes
and re-initializes Qwen3-ASR's streaming state so the internally accumulated
audio does not grow without bound. Set `--reset_interval_sec 0` to disable
state resets.


## Introduction


**MEGA-ASR** is purpose-built for **full-scenario robust ASR in the wild**, especially excelling at **semantic recovery** and **local keyword reconstruction** under severe acoustic degradation. It substantially reduces common failure modes such as **hallucinations**, **empty outputs**, and **dropped utterances**, making speech recognition reliable in truly challenging real-world environments.
<p align="center">
  <img src="assets/figures/radar_results.png" alt="Results" width="100%">
</p>

### Features 
✅ **One model for the messy real world**: Covers **7 atomic acoustic conditions** and **54 compound acoustic scenarios** in a single model.

✅ **Stronger recovery under severe distortion**: Excels at **semantic recovery** and **local keyword reconstruction**, greatly reducing **hallucinations**, **empty outputs**, and **dropped utterances**.

✅ **SOTA robust ASR performance**: Achieves up to nearly **30% gains** over leading open and closed source SOTA models in challenging acoustic environments.







## Acknowledgements

We sincerely thank the creators, maintainers, and contributors of the public datasets used in this work, including MUSAN, DNS Challenge, ESC-50, UrbanSound8K, LibriSpeech, Common Voice, WenetSpeech, and AISHELL-1.

We also sincerely thank the Qwen3-ASR Team for developing such an excellent foundation model, which provides a strong backbone for this work.

## Licence, Citation and stars
This project will be released under the **Apache-2.0 License**. You can do everything with Mega-ASR 🎉


**Citation**: You can cite Mega-ASR using the following BibTeX entry. Thank you for your kindness 🙂

```bibtex
@misc{xie2026megaasrinthewild2speechrecognition,
      title={Mega-ASR: Towards In-the-wild^2 Speech Recognition via Scaling up Real-world Acoustic Simulation},
      author={Zhifei Xie and Kaiyu Pang and Haobin Zhang and Deheng Ye and Xiaobin Hu and Shuicheng Yan and Chunyan Miao},
      year={2026},
      eprint={2605.19833},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2605.19833},
}
```
<a href="https://www.star-history.com/?repos=gpt-omni%2Fmini-omi%2Cxzf-thu%2FMega-ASR&type=date&legend=bottom-right">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=gpt-omni/mini-omi%2Cxzf-thu/Mega-ASR&type=date&theme=dark&legend=bottom-right" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=gpt-omni/mini-omi%2Cxzf-thu/Mega-ASR&type=date&legend=bottom-right" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=gpt-omni/mini-omi%2Cxzf-thu/Mega-ASR&type=date&legend=bottom-right" />
 </picture>
</a>
