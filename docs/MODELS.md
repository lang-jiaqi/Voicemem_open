# 模型 / Providers 一览

VoiceMem 每个能力都是**可插拔**的:有内置默认,一行 config 就在**本地 ↔ API** 之间切换（仿 mem0 的 `from_config`）。下面按「输入 → 记忆 → 回复」三段列全。

> **切换方式**：记忆侧（输入 + 记忆两段）用 `VoiceMem.from_config({...})`。
> ```python
> vm = VoiceMem.from_config({
>     "embedding": {"provider": "local"},          # 见下表的 provider 列
>     "slots":     {"provider": "openai"},
> })
> ```
> 回复的 **LLM 段已进核心**：`{"reply": {"provider": "openai", "config": {"model": ...}}}`，
> 或直接 `VoiceMem(reply=my_fn)` 换成自己的模型（见 README「回复：两条路」）。
> **TTS / Realtime 仍在 demo 层**——改 `web/run.py` 的 `CONFIG["reply"]`；那份
> `{"llm","tts","realtime"}` 嵌套写法核心也认，只取其中的 `llm`。

图例：🟢 本地开源 ・ 🔵 API ・ ✅ 默认

---

## ① 输入侧（语音感知）

| 能力 | provider | 默认 | 类型 | 模型 / 来源 |
|---|---|---|---|---|
| 流式 ASR | `funasr`(内置) | ✅ | 🟢 MIT | `paraformer-zh-streaming`（FunASR/ModelScope，首次运行自动下） |
| 流式 ASR | `sherpa`(内置) | — | 🟢 Apache | `sherpa-onnx-streaming-zipformer-bilingual-zh-en`（k2-fsa，中英双语）；`VOICEMEM_ASR=sherpa` |
| 流式 ASR | **外部**（`feed_partial`） | — | 任意 | Whisper / 云 ASR —— 换 ASR 只改喂进来的一行 |
| ASR（非流式精转写） | `sensevoice` | — | 🟢 | `FunAudioLLM/SenseVoiceSmall`（锁定一轮时比流式更准） |
| VAD | `silero`(内置) | ✅ | 🟢 MIT | `silero_vad.onnx`（`VOICEMEM_SILERO_VAD` 可指路径，config 可调 `threshold`） |
| VAD | `custom` / `VoiceMem(vad=…)` | — | 任意 | 你自己的：只要有 `is_speech(frame) -> bool` |
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
| slot 分类 | `local`(内置) | ✅ | 🟢 MIT | `LocalQueryClassifier`（E5 余弦，0 LLM 0 网络）——只出 slot，不抽实体 |
| slot 分类 | `openai` | — | 🔵 API | LLM 单次调用：分 slot **+ 抽实体 + 子 slot 下钻**；`VOICEMEM_SLOTS=openai` |
| 事实抽取/摘要 | `openai` | ✅ | 🔵 API | `gpt-4o-mini`（可用 `base_url` 指向本地/vLLM） |
| 记忆引擎(向量库) | `mem0` | ✅ | 🟢 | mem0 + 本地嵌入式 Qdrant（可注入换 zep 等） |

> **slot 分类默认走本地**——它在投机预取那 0–500ms 预算里（`stream._speculate`），不能走网络。
> 代价：本地版**不抽实体**（少了检索时的实体缩窄）、**不做子 slot 下钻**。要这两样就
> `VOICEMEM_SLOTS=openai` 或 `{"slots": {"provider": "openai"}}`。
> 本地版需要 `sentence-transformers`（在 `[demo]` extra 里）；缺了会打一行提示并回落 LLM 版。
>
> ⚠️ **embedding 和事实抽取仍默认走 API**。要**全本地**：`embedding` 也设 `{"provider":"local"}`、
> `base_url` 指向本地 LLM。

---

## ③ 回复侧

**对话 LLM 在核心里**（`voicemem/reply.py`），TTS / Realtime 在 demo 层。

### 对话 LLM（核心能力，`vm.reply()` / `vm.reply_stream()`）

| 能力 | provider | 默认 | 类型 | 模型 / 来源 |
|---|---|---|---|---|
| 对话 LLM | `openai`(内置) | ✅ | 🔵 API | `OPENAI_CHAT_MODEL` → `gpt-4o-mini`；`base_url` 可指向任意 OpenAI 兼容服务 |
| 对话 LLM | `custom` / `VoiceMem(reply=fn)` | — | 任意 | 你自己的函数：同步 / 协程 / 异步生成器都收 |
| 对话 LLM | 我们的 adapter | — | 🟢 | `LangJiaqi77/Voicemem-Qwen3_6-35B-A3B-QLoRA-v2`（vLLM 起服务后当 `base_url`，或 `reply=fn` 直接本地加载） |

### 模式一 `llm_tts`（web demo）：文字模型出字 → TTS 出声（两段拼）

| 能力 | provider | 默认 | 类型 | 模型 |
|---|---|---|---|---|
| TTS | `openai` | ✅ | 🔵 API | `gpt-4o-mini-tts` |
| TTS | `local` | — | 🟢 | piper（`VOICEMEM_TTS_MODEL`）—— 可换 kokoro / edge-tts |

### 模式二 `realtime`：原生语音，端到端

| 能力 | provider | 默认 | 类型 | 模型 |
|---|---|---|---|---|
| Realtime(原生语音) | `openai` | ✅ | 🔵 API | `gpt-realtime`（OpenAI Realtime） |

> 本地端到端语音模型（Moshi / Qwen-Omni 等）可自接,暂未内置。

---

## 一句话总结
- **语音感知(ASR/VAD/声纹/场景/情绪)= 纯本地开源**，`download_models.sh` 从 sherpa-onnx 官方拉 VAD/声纹/回退 ASR 那几个 `.onnx`，默认流式 ASR(paraformer) 与其余模型首次运行自动下。
- **记忆(embedding/slot/抽取)默认走 OpenAI API，但都有本地开源替代**（E5 / LocalQueryClassifier / 本地 LLM）。
- **回复(web demo)两种模式**：`llm_tts`（GPT-4o 出字 → TTS 出声）/ `realtime`（GPT Realtime 原生语音）；LLM、TTS 都可换本地。
