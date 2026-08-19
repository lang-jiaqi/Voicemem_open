# 模型 / Providers 一览

VoiceMem 每个能力都是**可插拔**的:有内置默认,一行 config 就在**本地 ↔ API** 之间切换（仿 mem0 的 `from_config`）。下面按「输入 → 记忆 → 回复」三段列全。

> **切换方式**：记忆侧用 `VoiceMem.from_config({...})`；回复侧(web demo)用 `web/run.py` 里的 `CONFIG["reply"]`。
> ```python
> vm = VoiceMem.from_config({
>     "embedding": {"provider": "local"},          # 见下表的 provider 列
>     "slots":     {"provider": "openai"},
>     "reply": {"tts": {"provider": "local"}, ...},
> })
> ```

图例：🟢 本地开源 ・ 🔵 API ・ ✅ 默认

---

## ① 输入侧（语音感知）

| 能力 | provider | 默认 | 类型 | 模型 / 来源 |
|---|---|---|---|---|
| 流式 ASR | `sherpa`(内置) | ✅ | 🟢 Apache | `sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20`（k2-fsa） |
| 流式 ASR | **外部**（`feed_partial`） | — | 任意 | FunASR / Whisper / 云 ASR —— 换 ASR 只改喂进来的一行 |
| ASR（非流式精转写） | `sensevoice` | — | 🟢 | `FunAudioLLM/SenseVoiceSmall`（锁定一轮时比流式更准） |
| VAD | `silero`(内置) | ✅ | 🟢 MIT | `silero_vad.onnx` |
| 声纹 | `3d-speaker`(内置) | ✅ | 🟢 Apache | `3dspeaker_...eres2net...16k.onnx` |
| 声学场景 | `ast` | ✅ | 🟢 | `MIT/ast-finetuned-audioset-10-10-0.4593`（HF 自动下） |
| 声学场景 | `clap` | — | 🟢 | laion `630k-audioset-best.pt`（`VOICEMEM_CLAP_CHECKPOINT`） |
| 情绪 | 声学启发式(内置) | ✅ | 🟢 无模型 | RMS/ZCR，直接算，不下载 |
| 情绪(多模态) | `qwen-omni` | — | 🟢 | `Qwen/Qwen2.5-Omni-3B` |

---

## ② 记忆侧（`VoiceMem.from_config`）

| 能力 | provider | 默认 | 类型 | 模型 |
|---|---|---|---|---|
| embedding | `openai` | ✅ | 🔵 API | `text-embedding-3-small` |
| embedding | `local` | — | 🟢 MIT | `intfloat/multilingual-e5-small` |
| slot 分类 | `openai` | ✅ | 🔵 API | LLM（单次调用分 7 slot + 抽实体） |
| slot 分类 | `local` | — | 🟢 | `LocalQueryClassifier`（E5 余弦，0 LLM） |
| 事实抽取/摘要 | `openai` | ✅ | 🔵 API | `gpt-4o-mini`（可用 `base_url` 指向本地/vLLM） |
| 记忆引擎(向量库) | `mem0` | ✅ | 🟢 | mem0 + 本地嵌入式 Qdrant（可注入换 zep 等） |

> ⚠️ **默认记忆是走 API 的**（embedding/slot/抽取 = OpenAI）。要**全本地**：`embedding` 和 `slots` 都设 `{"provider":"local"}`、`base_url` 指向本地 LLM。web demo 已默认注入本地 E5。

---

## ③ 回复侧（web demo，两种回复模式）

demo 有两条回复控制流，用 `DEMO_MODE` 切：

### 模式一 `llm_tts`：文字模型出字 → TTS 出声（两段拼）

| 能力 | provider | 默认 | 类型 | 模型 |
|---|---|---|---|---|
| 对话 LLM | `openai` | ✅ | 🔵 API | `gpt-4o`（`base_url` 可指向我们的 adapter / 任意 OpenAI 兼容） |
| 对话 LLM | 我们的 adapter | — | 🟢 | `LangJiaqi77/Voicemem-Qwen3_6-35B-A3B-QLoRA-v2`（vLLM 起服务） |
| TTS | `openai` | ✅ | 🔵 API | `gpt-4o-mini-tts` |
| TTS | `local` | — | 🟢 | piper（`VOICEMEM_TTS_MODEL`）—— 可换 kokoro / edge-tts |

### 模式二 `realtime`：原生语音，端到端

| 能力 | provider | 默认 | 类型 | 模型 |
|---|---|---|---|---|
| Realtime(原生语音) | `openai` | ✅ | 🔵 API | `gpt-realtime`（OpenAI Realtime） |

> 本地端到端语音模型（Moshi / Qwen-Omni 等）可自接,暂未内置。

---

## 一句话总结
- **语音感知(ASR/VAD/声纹/场景/情绪)= 纯本地开源**，`download_models.sh` 从 sherpa-onnx 官方拉那 3 个 `.onnx`，其余 HF 自动下。
- **记忆(embedding/slot/抽取)默认走 OpenAI API，但都有本地开源替代**（E5 / LocalQueryClassifier / 本地 LLM）。
- **回复(web demo)两种模式**：`llm_tts`（GPT-4o 出字 → TTS 出声）/ `realtime`（GPT Realtime 原生语音）；LLM、TTS 都可换本地。
