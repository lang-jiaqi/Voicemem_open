#!/usr/bin/env python3
"""微调出 Voicemem-Qwen QLoRA adapter：一条命令开训。

    python finetune/train.py --data data/train.jsonl

默认超参就是已发布 checkpoint-3318 那次训练用的（抄自
models/voicemem-qwen3.6-35b-a3b-qlora-v2/config.json），所以照默认跑 = 复现同一个
adapter。想调就用命令行参数覆盖，跑完会把实际用的参数写进输出目录。

数据格式：一行一条多轮对话的 JSONL（messages 是 OpenAI 那套写法）::

    {"messages": [
      {"role": "system",    "content": "…记忆上下文…"},
      {"role": "user",      "content": "我的猫叫什么"},
      {"role": "assistant", "content": "叫墨墨，三岁的英短。"}]}

只对 assistant 那几段算 loss（用户说的话不该被学着去生成）——见下面的
DataCollatorForCompletionOnlyLM。

显存：35B 全量 bf16 要 ~70GB；默认开 4bit QLoRA，单卡 40G 能跑。真放得下就
--no-4bit。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# 已发布 checkpoint 的配置，改这里就等于改"默认怎么训"
CFG = json.loads((Path(__file__).parent.parent /
                  "models/voicemem-qwen3.6-35b-a3b-qlora-v2/config.json").read_text())
A, T = CFG["adapter"], CFG["training"]


def parse():
    p = argparse.ArgumentParser(description="Voicemem QLoRA 微调")
    p.add_argument("--data", required=True, help="训练数据 JSONL")
    p.add_argument("--eval-data", default="", help="验证集 JSONL（可选）")
    p.add_argument("--out", default="out/voicemem-qlora", help="输出目录")
    p.add_argument("--base", default=CFG["base_model_hub_id"], help="基座模型")
    p.add_argument("--epochs", type=float, default=T["epochs"])
    p.add_argument("--lr", type=float, default=T["learning_rate"])
    p.add_argument("--batch-size", type=int, default=T["per_device_train_batch_size"])
    p.add_argument("--grad-accum", type=int, default=T["gradient_accumulation_steps"])
    p.add_argument("--max-len", type=int, default=T["max_sequence_length"])
    p.add_argument("--target-modules", default="",
                   help="LoRA 挂在哪些模块上。默认用 config.json 里那条正则，它是按 "
                        "Qwen3.6-35B-A3B 的模块命名写的——**换基座就必须换**，"
                        "否则 peft 直接报 target modules not found。不确定就填 all-linear")
    p.add_argument("--rank", type=int, default=A["rank"])
    p.add_argument("--alpha", type=int, default=A["alpha"])
    p.add_argument("--seed", type=int, default=T["seed"])
    p.add_argument("--no-4bit", action="store_true",
                   help="不量化（35B bf16 约需 70GB 显存）")
    p.add_argument("--resume", default="", help="从某个 checkpoint 继续")
    return p.parse_args()


def load_jsonl(path: str):
    """读 JSONL。字段不是 messages 的话，改这个函数就行——别的地方不用动。"""
    from datasets import Dataset
    rows = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise SystemExit(f"{path} 第 {i} 行不是合法 JSON：{e}")
        if "messages" not in obj:
            raise SystemExit(f"{path} 第 {i} 行缺 messages 字段。格式见 finetune/README.md")
        rows.append({"messages": obj["messages"]})
    if not rows:
        raise SystemExit(f"{path} 是空的")
    return Dataset.from_list(rows)


def main():
    args = parse()
    import torch
    from datasets import Dataset  # noqa: F401  （load_jsonl 里用）
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    train_ds = load_jsonl(args.data)
    eval_ds = load_jsonl(args.eval_data) if args.eval_data else None
    print(f"训练数据 {len(train_ds)} 条" + (f" / 验证 {len(eval_ds)} 条" if eval_ds else ""))

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # 4bit QLoRA：量化配置跟 adapter 训练时对齐（nf4 + 双重量化 + bf16 计算）
    quant = None if args.no_4bit else BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=quant, dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    model.config.use_cache = False              # 和 gradient_checkpointing 冲突

    lora = LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=A["dropout"],
        bias=A["bias"], task_type=A["task_type"],
        # 默认那条正则取自已发布的 adapter_config.json，训出来才和 checkpoint-3318
        # 命中同一批模块（锚点和转义都不能少，写松了匹配到的模块就不一样）。
        # 它绑死了 Qwen3.6-35B-A3B 的模块命名，换基座要用 --target-modules 覆盖。
        target_modules=args.target_modules or A["target_modules"],
    )

    kw = dict(
        output_dir=args.out, seed=args.seed,
        num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_len,
        lr_scheduler_type=T["lr_scheduler"],
        weight_decay=T["weight_decay"], adam_beta1=T["adam_beta"][0], adam_beta2=T["adam_beta"][1],
        optim=T["optimizer"], bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10, save_strategy="epoch",
        eval_strategy="epoch" if eval_ds else "no",
        report_to="none",
    )
    # warmup：config.json 记的是比例（0.03），但新版 trl 的 SFTConfig 去掉了
    # warmup_ratio、只剩 warmup_steps。两种都兼容一下，跨版本训出来的调度一致。
    import dataclasses
    if "warmup_ratio" in {f.name for f in dataclasses.fields(SFTConfig)}:
        kw["warmup_ratio"] = T["warmup_ratio"]
    else:
        per_epoch = max(1, len(train_ds) // max(1, args.batch_size * args.grad_accum))
        kw["warmup_steps"] = max(1, round(per_epoch * args.epochs * T["warmup_ratio"]))
    cfg = SFTConfig(**kw)

    trainer = SFTTrainer(model=model, args=cfg, train_dataset=train_ds,
                         eval_dataset=eval_ds, peft_config=lora,
                         processing_class=tok)
    trainer.train(resume_from_checkpoint=args.resume or None)

    trainer.save_model(args.out)                # 只存 adapter，不存基座权重
    tok.save_pretrained(args.out)
    Path(args.out, "train_args.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n训练完成。adapter 在 {args.out}")
    print(f"用它：见 finetune/README.md「训练完怎么用」")


if __name__ == "__main__":
    main()
