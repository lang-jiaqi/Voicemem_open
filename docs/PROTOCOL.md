# 浏览器 ↔ web demo WebSocket 通信规范

连 `ws://<host>:8787/ws`（由 `web/run.py` 起的 FastAPI 服务；前端可用 `?ws=host:port` 覆盖）。
前端只管**发音频/文字、收事件画界面**；分帧、分轮（VAD 判说完）、0–500ms 投机检索、
回复生成全在后端。记忆检索走 voicemem 核心 `vm.stream()`；回复由 demo 的两条控制流
（`llm_tts` = GPT 流→TTS 流 / `realtime` = OpenAI Realtime）完成，见 `web/run.py`。

---

## 1. 客户端 → 后端

同一条连接上发两种帧：

- **二进制帧 = 原始 PCM 音频**：16-bit / mono / **24 kHz**，逐帧持续推（不用等一句说完；
  后端 `vm.stream().feed()` 内部重采样到 16k、跑流式 ASR + VAD 分轮）。
- **文本帧 = 打字轮**（可选，无麦克风时用）：JSON
  ```json
  {"type": "user_text", "text": "我中午吃了什么"}
  ```

---

## 2. 后端 → 客户端

JSON 文本帧（状态/内容）和二进制帧（回复语音 PCM）混在同一连接上，按帧类型区分：
`bytes` → 回复语音；其余 → JSON。连上先发一条 `session_ready`。

| type | 时机 | 字段 |
|---|---|---|
| `session_ready` | WebSocket 刚连上 | `mode`（`llm_tts` 或 `realtime`） |
| `partial_transcript` | 用户说话过程中的实时转写 | `text`, `replace: true` |
| `user_transcript` | VAD 判定一轮说完的最终文本 | `text` |
| `memory_hits` | 该轮**投机检索**到的记忆（0–500ms 在关键路径外已算好） | 见 2.1 |
| `answer_start` | 开始回复 | — |
| `answer_delta` | 回复文字流（逐 token 增量） | `text` |
| `answer_done` | 这轮回复结束 | — |
| （二进制帧） | 回复语音流式吐出 | 24 kHz PCM，前端直接播放 |

### 2.1 `memory_hits` 的字段

直接来自 voicemem 核心 `Search()` 的结果（`SearchResult`）：

| 字段 | 含义 |
|---|---|
| `left_brain` | 左脑事实命中：`[{text, score, attributed_to}]` |
| `right_brain_hits` | 右脑情绪/画像命中：`[{content, source, priority}]` |
| `current_scene` | 当前声学场景 tag（如 `transit`），可能为 `null` |
| `related_summaries` | 相关 slot 的一句话摘要：`{slot: summary_text}` |

---

## 3. 一轮的时序

```
前端逐帧发 24k PCM ─▶ 后端 vm.stream().feed()
                        ├─ partial_transcript      （边说边出，灰字滚动）
                        └─ VAD 500ms 判说完（这 0–500ms 内已投机检索好记忆）
                            ├─ user_transcript      （最终文本）
                            ├─ memory_hits          （驱动左右脑面板）
                            └─ answer_start → answer_delta… → answer_done + 语音二进制帧
```

> barge-in：说到一半停顿又续上，`vm.stream()` 内部取消这次投机、不误触发回合，
> 前端只会看到 `partial_transcript` 继续更新，不会提前收到 `memory_hits`/`answer_*`。
