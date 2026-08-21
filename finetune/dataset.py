from utils import read_jsonl, system_prompt

MAX_HISTORY = 6
LANGS = ("zh", "en")
CATEGORIES = ("knowledge", "emotion", "persona", "casual")


def load(path):
    rows = read_jsonl(path)
    for i, row in enumerate(rows):
        check(row, i)
    return rows


def check(row, i=0):
    msgs = row["messages"]
    assert msgs[0]["role"] == "system", f"第 {i} 条第一句不是 system"
    assert msgs[-1]["role"] == "assistant", f"第 {i} 条最后一句不是 assistant"
    assert len(msgs) % 2 == 1, f"第 {i} 条 system 之后没有成对的 user/assistant"
    for j, m in enumerate(msgs[1:]):
        want = "user" if j % 2 == 0 else "assistant"
        assert m["role"] == want, f"第 {i} 条第 {j + 1} 句该是 {want}"
        assert isinstance(m["content"], str), f"第 {i} 条第 {j + 1} 句 content 不是字符串"
    assert history_turns(row) <= MAX_HISTORY, f"第 {i} 条历史超过 {MAX_HISTORY} 轮"
    meta = row.get("meta", {})
    assert meta.get("lang") in LANGS, f"第 {i} 条 lang 非法: {meta.get('lang')}"
    assert meta.get("category") in CATEGORIES, f"第 {i} 条 category 非法: {meta.get('category')}"
    want = system_prompt(meta["category"], meta["lang"])
    assert msgs[0]["content"] == want, f"第 {i} 条 system 和 category/lang 对不上"


def history_turns(row):
    return (len(row["messages"]) - 3) // 2


def question(row):
    # 当前轮，记忆块就拼在这一句里；历史轮不带记忆。
    return row["messages"][-2]["content"]


def answer(row):
    # 只有这一句算 loss，靠 train.py 的 loss_scale="last_round"。
    return row["messages"][-1]["content"]


def prompt(row):
    return row["messages"][:-1]
