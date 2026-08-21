# finetune — 训一个自己的 Voicemem-Qwen adapter

```bash
pip install "transformers>=4.45" trl peft datasets accelerate
pip install bitsandbytes        # 4bit 量化用；bitsandbytes 要 CUDA，
                                # Mac / CPU 上装不了，那就加 --no-4bit 跑

python finetune/train.py --data data/train.jsonl
```

默认超参就是已发布的 `checkpoint-3318` 那次训练用的（直接读
`models/voicemem-qwen3.6-35b-a3b-qlora-v2/config.json`），所以**照默认跑 = 复现同一个
adapter**。跑完 `out/voicemem-qlora/` 里是 adapter 权重 + 这次实际用的参数。

先拿仓库自带的 3 条样例把流程走通，再上真数据：

```bash
# 换个小基座验证流程，几分钟跑完（不占显存，不用 CUDA）
# 注意 --target-modules：默认那条正则是按 35B 的模块命名写的，换基座必须换
python finetune/train.py --data finetune/examples/sample.jsonl \
  --base Qwen/Qwen2.5-0.5B-Instruct --out out/smoke --target-modules all-linear \
  --epochs 1 --batch-size 1 --grad-accum 1 --max-len 512 --no-4bit
```

## 数据长什么样

一行一条多轮对话的 JSONL，`messages` 是 OpenAI 那套写法：

```json
{"messages": [
  {"role": "system",    "content": "你是用户的语音助手，简短自然地回答。\n\n以下是你记得的关于这个用户的事：\n- [2023-05-08] 用户养了一只英短猫，名字叫墨墨，今年三岁。"},
  {"role": "user",      "content": "我的猫叫什么名字来着"},
  {"role": "assistant", "content": "叫墨墨呀，三岁的英短。"}]}
```

**system 里放的就是 voicemem 检索出来的记忆**——格式跟运行时
`build_memory_context()` 吐出来的一致（含 `[YYYY-MM-DD]` 日期前缀）。训练目标是让模型
学会「拿到这样一段记忆，自然地用上」，而不是背下具体某条记忆。

所以造数据的路子是：拿真实对话跑一遍 voicemem，把每轮检索到的记忆当 system，
用户那句当 user，理想回复当 assistant。**记忆里没有的信息就该答不知道**——样例第 3 条
就是这种负例，不放的话模型学会的是编。

字段名跟你的数据对不上，改 `train.py` 里的 `load_jsonl()`，别的地方不用动。

## 常用参数

| 参数 | 干什么 | 默认（= 已发布 checkpoint） |
|---|---|---|
| `--data` / `--eval-data` | 训练集 / 验证集 | 必填 / 无 |
| `--base` | 基座模型 | `Qwen/Qwen3.6-35B-A3B` |
| `--epochs` | 训几轮 | 2 |
| `--lr` | 学习率 | 2e-4（cosine，warmup 3%） |
| `--batch-size` / `--grad-accum` | 单卡批大小 / 梯度累积 | 8 / 2 |
| `--max-len` | 最大序列长度 | 2048 |
| `--rank` / `--alpha` | LoRA 秩 / alpha | 32 / 64 |
| `--target-modules` | LoRA 挂哪些模块。默认那条正则**绑死 35B 的模块命名**，换基座必须换（不确定就 `all-linear`） | config.json 里的正则 |
| `--no-4bit` | 不量化（显存够才用） | 默认开 4bit |
| `--resume` | 从 checkpoint 续训 | 无 |

**显存**：35B 全量 bf16 约需 70GB。默认开 4bit QLoRA（nf4 + 双重量化 + bf16 计算），
单卡 40G 能跑。显存不够就调小 `--batch-size`、同步调大 `--grad-accum` 保持等效批大小。

## 训练完怎么用

adapter 挂回基座就行，跟已发布那个用法完全一样：

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-35B-A3B", trust_remote_code=True)
model = PeftModel.from_pretrained(base, "out/voicemem-qlora")
```

接进语音对话：`scripts/realtime_funasr_qwen.py` 里把 `ADAPTER` 改成你的输出目录。
那个脚本走的是回复层的「路 B」——把模型包成 `(text, memory_context) -> str` 传给
`VoiceMem(reply=fn)`，其余（ASR / VAD / 投机预取 / 记忆检索）核心全包了。

评测新 adapter：

```bash
python evaluation/run.py --dataset locomo --data data/locomo.json --answer-model <你的模型端点>
```

## 说明

- 训练数据**不在这个仓库**。公开前需要补齐来源、许可、同意状态和预处理说明
  （见 `models/voicemem-qwen3.6-35b-a3b-qlora-v2/config.json` 的 `data` 段）。
- LoRA 只训语言模型侧的注意力和 FFN 投影（含 MoE 的 `shared_expert_gate`），
  具体正则在 `config.json` 的 `target_modules_pattern`。
- 基座模型的许可和获取条件请自行确认；adapter 不能当独立模型分发。
