# evaluation — 跑一条命令，出一个数字

```bash
export OPENAI_API_KEY=sk-...

# 先拿仓库自带的小样例验证环境（2 段对话 5 个问题，几十秒）
python evaluation/run.py --dataset locomo --data evaluation/examples/locomo_sample.json

# 换成真数据集
python evaluation/run.py --dataset locomo --data data/locomo.json
```

跑完直接打印，结果同时写进 `results/locomo.json`：

```text
locomo  10 段对话 · 152 题
得分 139/152  =  91.4%
   multi_hop                88.2%
   temporal                 85.7%
   single_hop               95.1%
检索中位数 12ms · 记忆中位数 298 tokens
结果已存 results/locomo.json
```

**先跑 `--inspect`**：不花钱、不调模型，只把数据集解析结果打出来给你看。字段对不上
就改 `datasets/locomo.py` 里那几行——比跑了两小时才发现全错强。

```bash
python evaluation/run.py --dataset locomo --data data/locomo.json --inspect
```

## 它到底在测什么

```
① 读数据集       →  list[Conversation]
② 逐轮 ingest    →  每段对话一个独立记忆库
③ 每题 search    →  拿到这题该用的记忆（top-k 条）
④ 让模型作答     →  只给它检索到的记忆，不给原始对话
⑤ 判分           →  数据集自己的口径
```

**第 ④ 步只给记忆、不给原文，是这个评测的关键**——给了原文就变成阅读理解，测的
是答题模型而不是记忆系统。所以这里的分数衡量的是「voicemem 有没有把该记的记住、
该找的找出来」。

## 常用参数

| 参数 | 干什么 | 默认 |
|---|---|---|
| `--dataset` / `--data` | 用哪个适配器 / 数据文件 | 必填 |
| `--answer-model` | 拿记忆作答的模型 | `gpt-4o-mini` |
| `--judge` | 判分的裁判模型 | `gpt-4o-mini` |
| `--top-k` | 每题检索几条记忆 | `5` |
| `--mode` | `left_brain_single`=只测事实记忆；`text_mode`=连右脑一起 | `left_brain_single` |
| `--workers` | 并发跑几段对话 | `4` |
| `--limit` | 只跑前 N 段（调试用） | 全部 |
| `--resume` | 接着上次跑，跳过已完成的对话 | 关 |
| `--save-memory` | 把每题检索到的记忆也存进结果，便于人工复核 | 关 |
| `--inspect` | 只解析数据集并打印，不跑评测 | 关 |

结果每跑完一段就落盘，所以跑几小时的评测中途挂了，加 `--resume` 接着跑即可。

模型走 OpenAI 兼容接口；换自建端点设 `OPENAI_BASE_URL` 就行。

## 评测协议（报数字时请一并说明）

这几条直接影响分数高低，换一条数字就不可比：

- **答题模型只看检索结果**，看不到原始对话（见上）。
- **每段对话一个独立记忆库**。混在一起等于把别的对话的答案也喂了进去。
- **时间戳要带上**。`Turn.observed_at` 传的是对话真实发生的日期，不是跑评测那天；
  不传的话时序类问题（"这事发生在哪次之前"）会全错。
- **判分口径**：先字面包含匹配，不中再交给裁判模型（提示词在 `datasets/locomo.py`
  的 `JUDGE` 里，宽松判定——意思对就算对）。
- **没有标准答案的题不计入分母**（`Score(total=0)`）。
- 报告时写清楚：答题模型、裁判模型、`--top-k`、`--mode`。

## 加一个新 benchmark

一个文件、两个函数，主流程一行不用动。

**1. 复制 `datasets/locomo.py` 改成 `datasets/你的数据集.py`**，实现两个函数：

```python
def load(path: str) -> list[Conversation]:
    """读你的数据文件，转成统一结构。
    Conversation(id, turns=[Turn(speaker, text, observed_at)], questions=[Question(...)])
    """

def score(q: Question, answer: str, judge) -> Score:
    """判这道题对不对。judge(system, user) -> str 是注入进来的裁判模型。
    Score(correct=1.0, total=1.0, note="判分理由")
    rubric 类的评分：correct=满足的要点数, total=总要点数
    """
```

数据结构定义都在 `datasets/__init__.py`，一共二十来行，打开就看完了。

**2. 登记到 `datasets/__init__.py` 的 `get()`**：

```python
table = {"locomo": locomo, "你的数据集": 你的模块}
```

**3. 跑**：

```bash
python evaluation/run.py --dataset 你的数据集 --data data/xxx.json --inspect   # 先验证解析
python evaluation/run.py --dataset 你的数据集 --data data/xxx.json
```

只有「怎么读」和「怎么判分」是数据集自己的事，中间那段（ingest → search → 作答）
所有数据集共用同一份代码——这样不同 benchmark 的数字才有可比性。
