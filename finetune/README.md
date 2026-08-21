# finetune

训一个自己的 VoiceMem 回复 adapter。

```bash
pip install -e ".[finetune]"        # ms-swift + bitsandbytes；torch 按自己的平台装
python finetune/train.py            # 先拿自带的 5 条样例把流程走通
python finetune/train.py --data data/train.jsonl
```

默认超参直接读已发布 adapter 的清单
`models/voicemem-qwen3.6-35b-a3b-qlora-v2/training_config.json`，**照默认跑 =
复现同一次训练**（base `Qwen/Qwen3.6-35B-A3B`，LoRA rank 32 / alpha 64，4bit）。

## 文件

| | |
|---|---|
| `train.py` | 训练入口，ms-swift 的 `sft_main` |
| `eval.py` | 拿训好的 adapter 在数据上跑一遍，逐条打印 `ref` / `pred` |
| `dataset.py` | 读 JSONL 并**逐条校验**格式，不合规直接报第几条第几句 |
| `utils.py` | 读清单里的超参、四种 system prompt、warmup 换算 |
| `data/sample.jsonl` | 5 条样例，四个 category 都有 |

## 数据格式

一行一条对话，`messages` 是 OpenAI 那套写法：

```json
{
  "messages": [
    {"role": "system",    "content": "<按 category/lang 选，见下>"},
    {"role": "user",      "content": "我的猫叫什么名字来着\n\nMEMORY CONTEXT (things you remember about the user):\n- [2023-05-08] 用户养了一只英短猫，名字叫墨墨，今年三岁。"},
    {"role": "assistant", "content": "叫墨墨呀，三岁的英短。"}
  ],
  "meta": {"lang": "zh", "category": "knowledge", "session_id": "s_0001", "turn": 1}
}
```

`dataset.py` 会强制这几条，任何一条不满足都会中断并指出是第几条：

- 第一句必须是 `system`，最后一句必须是 `assistant`，中间严格 user/assistant 交替
- 历史最多 6 轮（`MAX_HISTORY`）
- `meta.lang` ∈ `zh` / `en`，`meta.category` ∈ `knowledge` / `emotion` / `persona` / `casual`
- **system 必须和 `category`+`lang` 对得上**——四种 prompt 写死在 `utils.py` 的 `SYSTEM` 里，
  `knowledge`/`emotion`/`persona` 共用「带记忆」那版，`casual` 用「闲聊」那版

记忆块拼在**最后一轮的 user 里**，历史轮不带记忆。只有最后一句 assistant 算 loss
（`loss_scale="last_round"`），历史轮不算。

## 常用参数

```bash
python finetune/train.py --data data/train.jsonl \
    --out out/my-adapter --epochs 3 --lr 1e-4 --no-4bit
```

| | | 默认 |
|---|---|---|
| `--data` | 训练数据 | `finetune/data/sample.jsonl` |
| `--out` | 输出目录 | `out/voicemem-qlora` |
| `--base` | 基座模型 | 清单里的 `Qwen/Qwen3.6-35B-A3B` |
| `--rank` / `--alpha` | LoRA 秩 / alpha | 32 / 64 |
| `--epochs` / `--lr` | 轮数 / 学习率 | 清单里的值 |
| `--max-len` | 最大序列长度 | 清单里的值 |
| `--no-4bit` | 不做 4bit 量化（显存够才用） | 默认开 4bit |

**换基座必须改 `target_regex`**——它在清单的 `adapter.target_modules` 里，按
Qwen3.6-35B-A3B 的模块命名写死的。不确定就改成 `all-linear`。

## 评估

```bash
python finetune/eval.py --adapter out/voicemem-qlora --out preds.jsonl
```

逐条打印 `question` / `ref` / `pred` / `meta`，`--out` 存成 JSONL。
温度固定 0，方便复现比较。

跑记忆检索本身的指标见 [`evaluation/`](../evaluation/)。

## 说明

- **训练数据不在这个仓库**。公开前要补齐来源、许可、同意状态和预处理说明。
- adapter 不能当独立模型分发，基座的许可和获取条件请自行确认。
- 多卡训练时 `utils.warmup_steps()` 是按单进程算的，要再除以卡数。
