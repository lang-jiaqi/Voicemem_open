import argparse
import json

from swift import InferRequest, RequestConfig, TransformersEngine

from dataset import answer, load, prompt, question
from utils import BASE, DATA

p = argparse.ArgumentParser()
p.add_argument("--adapter", required=True)
p.add_argument("--data", default=str(DATA))
p.add_argument("--base", default=BASE)
p.add_argument("--max-tokens", type=int, default=256)
p.add_argument("--out", default="")
args = p.parse_args()

rows = load(args.data)
engine = TransformersEngine(args.base, adapters=[args.adapter])
config = RequestConfig(max_tokens=args.max_tokens, temperature=0.0)
resps = engine.infer([InferRequest(messages=prompt(r)) for r in rows], config)

out = []
for row, resp in zip(rows, resps):
    out.append({
        "question": question(row),
        "ref": answer(row),
        "pred": resp.choices[0].message.content,
        "meta": row["meta"],
    })
    print(json.dumps(out[-1], ensure_ascii=False))

if args.out:
    from utils import write_jsonl
    write_jsonl(out, args.out)
