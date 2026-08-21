import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "models/voicemem-qwen3.6-35b-a3b-qlora-v2/training_config.json"
DATA = Path(__file__).parent / "data" / "sample.jsonl"

CFG = json.loads(MANIFEST.read_text())
BASE = CFG["base_model_hub_id"]
ADAPTER = CFG["adapter"]
TRAIN = CFG["training"]

MEMORY_CATEGORIES = ("knowledge", "emotion", "persona")

SYSTEM = {
    ("memory", "zh"): "你是VoiceMem，一个有记忆的个人AI伴侣。用下面检索到的记忆和用户画像自然地"
                      "回应，不要把记忆原样念给用户。",
    ("memory", "en"): "You are VoiceMem, a personal AI companion with memory. Reply naturally "
                      "using the retrieved memories and user profile below; do not recite them "
                      "verbatim to the user.",
    ("casual", "zh"): "你是VoiceMem，一个聪明有温度的AI伴侣。基于当下对话自然地回应。",
    ("casual", "en"): "You are VoiceMem, a warm and thoughtful AI companion. Reply naturally "
                      "based on the current conversation.",
}


def system_prompt(category, lang):
    kind = "memory" if category in MEMORY_CATEGORIES else "casual"
    return SYSTEM[(kind, lang)]


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows, path):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def warmup_steps(n_rows, epochs):
    # transformers 5.x 删了 warmup_ratio，只剩 warmup_steps，这里自己换算。
    # 按单进程算，多卡要再除以卡数。
    per_step = TRAIN["per_device_train_batch_size"] * TRAIN["gradient_accumulation_steps"]
    return max(1, round(n_rows * epochs / per_step * TRAIN["warmup_ratio"]))
