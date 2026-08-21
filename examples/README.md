# examples

四个能直接跑的例子，从最小到最完整。

```bash
pip install -e .
export OPENAI_API_KEY=sk-...
```

| 例子 | 干什么 | 额外要什么 |
|---|---|---|
| [`01_text.py`](01_text.py) | 文本模式：存文字、查文字，只用左脑 | — |
| [`02_audio.py`](02_audio.py) | 音频模式：喂 wav，双脑都写（ASR/声纹/场景/情绪） | `bash scripts/download_models.sh` |
| [`03_voice_chat.py`](03_voice_chat.py) | 最简单的语音对话：麦克风 → 带记忆回答 → 存 | 同上 + `pip install sounddevice` |
| [`04_custom_asr_vad.py`](04_custom_asr_vad.py) | 换成自己的 ASR / VAD | — |

```bash
python examples/01_text.py
python examples/02_audio.py speech.wav
python examples/03_voice_chat.py
python examples/04_custom_asr_vad.py
```

## 换自己的 ASR / VAD

`04` 里有两条路，按你的 ASR 是什么形态选：

**路 1 — 组件替换**。你的 ASR 能一块块吃音频、随时给出「到目前为止的文本」：

```python
vm = VoiceMem(asr=lambda: MyASR(), vad=lambda: MyVAD())   # 零参工厂，不是实例
```

要实现的协议就这几个方法，`samples` 是 np.float32 / 16kHz / 单声道 / [-1, 1]：

```python
class MyASR:
    def reset(self): ...                 # 一轮开始
    def feed(self, samples) -> str: ...  # 返回累积文本
    def flush(self) -> str: ...          # 可选，收尾补字

class MyVAD:
    def is_speech(self, samples) -> bool: ...
```

**路 2 — 只喂文本**。你的 ASR/VAD 已经在别处跑了（云端、系统自带、另一个进程），
voicemem 完全不碰音频：

```python
await stream.feed_partial("我对坚果")                  # partial 来一句喂一句
turn = (await stream.feed_partial("我对坚果过敏", ended=True)).turn
```

`ended=True` 就是你的 VAD 说「这句说完了」。这条路不会加载任何音频模型。

## 关于记忆库

不传 `memory_root` 时所有例子共用同一个库（包目录下的 `results/voice_memory`）。
想各跑各的就显式指定：

```python
VoiceMem(..., memory_root="/tmp/my_mem")
```

注意 embedding 换了维度就不能共用旧库——比如默认的 OpenAI embedding（1536 维）
和 `from_config({"embedding": {"provider": "local"}})` 的本地 E5（384 维）各自建库。

---

仓库根目录的 [`example.py`](../example.py) 是完整版：麦克风和 wav 两种输入、
partial 回显、投机预取的各个阶段都打出来。想看流式内部发生了什么看它。
