# service

实时语音服务层：处理"通话正在进行中"这种活的状态，直接调用 `../voicemem`
（进程内调用，不走 HTTP 中转，`cognitive_server.py` 那个独立 Flask 进程已经
不存在了）。

| 文件 | 内容 |
|---|---|
| `asr.py` | 流式转写（`StreamingASR`）+ 精转写（`Transcriber`），sherpa-onnx / SenseVoice |
| `live_signals.py` | 实时快速版说话人/情绪识别（`SpeakerId`/`EmotionRecognizer`）+ Silero VAD（`Vad`，"有没有人在说话"，跟 voicemem 核心的情绪 VAD 同名不同义） |
| `router.py` | 文本 -> 实体 + 簇分类（jieba + model2vec），原样保留 |
| `turn.py` | `Turn` 状态容器 + 计时常量（200ms/500ms 阈值等）+ 纯函数，没有回调依赖 |
| `cognitive.py` | `CognitiveService`：原来 `cognitive_server.py`（Flask）的逻辑搬过来，改成直接调用 `voicemem.VoiceMem`，不再序列化成 HTTP JSON |
| `session.py` | `VoiceAssistant`：真正的时序状态机（说话中/预览/预取/锁定），把上面这些东西串起来 |
| `tts.py` | Gemini Live 语音输出（原 `live_tts.py`，已去掉硬编码 key） |
| `ws_server.py` | 入口：WebSocket 服务，浏览器 ↔ `VoiceAssistant` |

**没有迁移的东西**：原 `models.py` 里的 `EcapaEncoder`（旧的跨 session ECAPA 声纹
匹配）。这个功能在当前协议里其实已经是死代码——服务端本来就不读它算出来的
`voiceprint_vec`/`confirmed_person_id`，也从没往回传过 `person_id`，这段逻辑
从没真正生效过。跨 session 认人现在完全由 `voicemem` 核心的 CAM++ 负责
（`Ingest()` 返回的 `info_detail.audiomem_info.speaker_id` 就是 CAM++ 认出的
person_id）。

浏览器 ↔ 这层的 WebSocket 协议见 `../docs/PROTOCOL.md`。

## 如何运行

安装依赖（这层需要 `sherpa-onnx`/`funasr`/`websockets`/`jieba`/`model2vec`/
`google-genai`，都在 `service` extra 里）：

```bash
# 从仓库根目录：
pip install -e .                                        # voicemem 核心，文本模式即可
pip install -e ".[audio,environment,omni,voiceprint,service]"
```

下载模型（约 570MB，`service/models/` 已 gitignore，只需下一次）：

```bash
bash scripts/download_models.sh service/models
```

必需的环境变量：

| 变量 | 用途 |
|---|---|
| `OPENAI_API_KEY` | voicemem 核心自己的抽取/分类/检索排序调用 |
| `GEMINI_API_KEY`（或 `GOOGLE_API_KEY`） | `tts.py` 的 Gemini Live 语音输出；不设置会在启动时报错退出 |

可选环境变量（有默认值，一般不用改）：`VOICEMEM_MODELS_DIR`（默认
`service/models`）、`VOICEMEM_MEMORY_ROOT`、`VOICEMEM_WS_PORT`（默认
`8771`）、`VOICEMEM_WARMUP`、`EMOTION_BACKEND`、`SILENCE_LOCK`、
`BARGE_MIN_VOICE`、`EMO_MIN_CONF` 等一批阈值调参项（见 `live_signals.py`/
`turn.py`）。

启动：

```bash
cd service
python ws_server.py                            # ws://localhost:8771
```

这层本身不带浏览器前端——见协议文档自己接一个客户端，或参考
[`../openai_voice_demo/frontend/index.html`](../openai_voice_demo/frontend/index.html)
的 WebSocket 客户端写法（协议不同，但连接/PCM播放的基本模式类似）。
