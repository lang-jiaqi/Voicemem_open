# 浏览器 ↔ backend WebSocket 通信规范

连 `ws://<host>:8771`（`service/ws_server.py`，端口可用 `VOICEMEM_WS_PORT` 覆盖）。

> 跟旧版的差异：以前 assistant.py 还要经 HTTP 调一个独立的 `cognitive_server.py`
> 进程，那段内部协议现在不存在了（`backend/service/cognitive.py` 直接在同一进程
> 内被调用）。这份文档只保留、并补全了浏览器真正会收到的那部分——也就是做 GUI
> 真正需要对接的东西。

---

## 1. 客户端 → 后端

只发一种东西：**原始 PCM 音频**，16kHz / 16bit / mono，作为 WebSocket 二进制帧，
持续推送（不用等一句话说完再发，逐帧发即可，后端自己做分帧/分轮）。

---

## 2. 后端 → 客户端

JSON 文本帧（控制/状态消息）和二进制帧（回答语音 PCM）混在同一条连接上，
用帧类型区分：`bytes`/`bytearray` → 24kHz PCM 音频；其余 → JSON。

### 2.1 消息类型总览

| type | 时机 | 关键字段 | 用来驱动什么 |
|---|---|---|---|
| `partial` | 用户说话过程中 | `global_id`, `text` | 实时滚动文字（未锁定，灰字） |
| `result`（`locked:false`） | 停顿满 200ms | `text`,`speaker`,`emotion`,`entities`,`slots`,`status` | 说话人/情绪/实体标签先显示一版 |
| `result`（`locked:true`） | 停顿满 500ms | 上述字段 + `info`,`left_brain`,`right_brain`,`audiomem`,`elapsed_ms` | **大脑面板**：这轮最终定档的记忆检索结果 |
| `append` | 碎字并入上一句 | `text`,`speaker`,`emotion` | 修正上一条已显示的文字 |
| `answer_start` | 开始生成回答 | — | 提示"AI 正在想"状态 |
| `answer` | 回答文字流式吐出 | `text`（增量） | 打字机效果显示 AI 说的话 |
| （二进制帧） | 回答语音流式吐出 | 24kHz PCM | 播放音频、驱动波形动画 |
| `answer_done` | 这轮回答说完了 | — | 状态切回"倾听中" |
| `answer_interrupt` | 被打断（用户抢话 / barge-in） | — | 清空前端已排队但还没播的音频 |
| `playback` | 命中"回放原声"请求 | `memory_text`,`audio_b64` | 播放一段历史原始录音 |
| `history` | WebSocket 刚连接时 | `events` | 恢复本机服务端保存的聊天记录 |

### 2.2 `result`（`locked:true`）里的 `audiomem` 字段

这是"能看到 AI 大脑在转"这个 GUI 差异化功能的直接数据源，来自
`voicemem` 核心 `Ingest()` 的返回值：

| 字段 | 含义 | 更新频率 |
|---|---|---|
| `current_scene` | 当前声学场景（如"办公室"） | 每轮都有（只要带了音频） |
| `speaker_id` | 声纹（3D-Speaker ERes2Net）认出的跨 session person_id | 每轮都有 |
| `recognized_tune` | 识别到的熟悉背景音乐/哼唱 | 偶尔 |
| `abnormal_sounds` | 异常环境音（碎裂声/警报/尖叫） | 偶尔，罕见 |
| `recognized_place` / `familiar_place_prompt` | 熟悉地点识别 | 偶尔 |
| `new_routine` | 新发现的生活规律 | 偶尔 |
| `triggered_reminders` | 触发的场景提醒 | 偶尔 |
| `scene_trigger_created` | 本轮新设置的提醒 | 偶尔 |
| `playback` | 本轮命中的原声回放请求 | 偶尔 |

**注意**：`left_brain`/`right_brain`（文字形式的记忆摘要）和上面这些 audiomem
字段是"每轮都有内容，但内容本身有没有新东西"的关系——`current_scene`/
`speaker_id` 基本每轮都有值；后面那些"偶尔"的字段大多数轮次是空的，只有
命中时才会出现。设计 UI 时建议按"平时安静、命中时才亮起"来做，不要预期
每轮都有新东西可显示。

---

## 3. 碎字与时长规则

- 语音段**短于 1.5s**：不做说话人 / 情感 / 路由分析（相应字段为 `-` 或空）。
- 锁定时文本**不足 3 个字符单位**（中文字数 + 英文单词数）：视为碎字，**并入上一句**（走 `append`），不新开 turn，不触发回答。
