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

## 一个坑：记忆库不能混用

不传 `memory_root` 时所有例子共用同一个库（包目录下的 `results/voice_memory`）。
想各跑各的就显式指定：

```python
VoiceMem(..., memory_root="/tmp/my_mem")
```

**embedding 换了维度就不能共用旧库** —— 默认的 OpenAI embedding 是 1536 维，
`from_config({"embedding": {"provider": "local"}})` 的本地 E5 是 384 维，
指向同一个 `memory_root` 会直接报维度不匹配。
