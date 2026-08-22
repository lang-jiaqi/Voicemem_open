# examples

三个能直接跑的例子，从「当记忆库用」到「做一个会说话的 agent」。

```bash
pip install -e .
export OPENAI_API_KEY=sk-...
```

| | 干什么 | 额外要什么 |
|---|---|---|
| [`01_memory.py`](01_memory.py) | 存和查 —— 最小用法 | 音频那半段要 `bash scripts/download_models.sh` |
| [`02_streaming.py`](02_streaming.py) | 流式接口：喂音频块，看每一轮算出了什么 | 同上 |
| [`03_simple_agent_with_voicemem_memory.py`](03_simple_agent_with_voicemem_memory.py) | 完整语音 agent：边听边取记忆、说话时能被打断 | `pip install openai sounddevice scipy kokoro` |

## 01 · 存和查

```bash
python examples/01_memory.py
```

两种输入，区别只在 `mode`：

```python
VoiceMem(mode="normal")           # 音频 → 双脑（ASR / 声纹 / 场景 / 情绪都跑）
VoiceMem(mode="leftbrain_only")   # 文本 → 只有左脑（事实），不做情绪归因
```

查出来的东西分两半：`result.result_leftbrain` 是事实，`result.result_rightbrain`
是画像和情绪。

## 02 · 流式接口

```bash
python examples/02_streaming.py speech.wav
```

按块喂音频，每块回一个状态；VAD 判定说完时 `state` 变成 `turn_over`，这时候
`memory_context` 早就算好了 —— 检索是在你还在说的时候后台跑完的，不占回复前面
那段时间。

`turn_over` 那一刻能拿到的全部字段：

```
result_leftbrain / result_rightbrain    这一轮检索到的记忆
speaker_id / speaker_voiceprint         谁在说
emotion / transcript                    情绪 / 转写
entity / schema / text_embedding        抽出的实体、槽位、向量
```

## 03 · 完整语音 agent

```bash
python examples/03_simple_agent_with_voicemem_memory.py
```

麦克风 → voicemem 边听边预取记忆 → OpenAI 带着记忆回答 → Kokoro 本地 TTS 出声，
**你一开口它就闭嘴**。想看 web 版（带脑图和记忆面板）用 `python web/run.py`。

## 为什么都用 `from_config` 而不是 `VoiceMem(openai_key=...)`

三个例子的 embedding 和 slots 都走本地 E5：

```python
vm = VoiceMem.from_config({
    "mode": "normal",
    "embedding": {"provider": "local"},   # 记忆向量：本地，0 网络
    "slots":     {"provider": "local"},   # 槽位分类：本地，0 LLM
    "api_key":   os.environ["OPENAI_API_KEY"],   # 只在写入侧抽事实时用
})
```

这不是可选的优化 —— **投机预取那 0–500ms 预算里不能走网络**。你说话的时候检索
就在后台跑，说完时记忆得是现成的；embedding 要是每次发一个 HTTP 去 OpenAI，
光往返就吃掉整个预算。README 里的 134ms 说的就是这套配置，实测 search 本体 ~10ms。

`VoiceMem(openai_key=...)` 这种默认构造用的是 OpenAI embedding，也能跑，只是
每轮检索都要联网。

## 顺带一个坑：库不能混用

不传 `memory_root` 时所有例子和 web demo 共用同一个库（包目录下的
`results/voice_memory`）。**向量维度不同的库不能混用** —— 本地 E5 是 384 维、
OpenAI 是 1536 维，指同一个目录会直接报：

```
ValueError: shapes (25,384) and (1536,) not aligned
```

例子现在跟 demo 用同一套本地配置，所以不会撞上。要各跑各的就显式指定
`memory_root`。
