"""Dump what actually landed in each user's memory after ingest.

    python case_study/inspect_memory.py            # both users
    python case_study/inspect_memory.py --users maya

Prints left-brain fact counts, right-brain memories grouped by class and
emotion, the synthesized persona document, and the pre-stimulus block -- i.e.
the state the four cases retrieve from.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MEM0_TELEMETRY", "False")

from case_study.corpus import USERS  # noqa: E402
from case_study.run_cases import DEFAULT_MEMORY_ROOT, make_vm  # noqa: E402


def dump_user(user_key: str, memory_root: Path) -> None:
    root = memory_root / user_key
    print(f"\n{'=' * 72}\n{user_key}  ({root})\n{'=' * 72}")
    if not root.exists():
        print("  (not ingested)")
        return

    rb_db = root / "right_brain.sqlite"
    if rb_db.exists():
        con = sqlite3.connect(rb_db)
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT memory_class, content, metadata, created_at "
            "FROM right_brain_memories WHERE user_id = ? ORDER BY created_at",
            (user_key,),
        ).fetchall()
        by_class = Counter(r["memory_class"] for r in rows)
        emotions = Counter(
            (json.loads(r["metadata"] or "{}") or {}).get("emotion", "") for r in rows
        )
        print(f"\nRIGHT BRAIN: {len(rows)} memories  {dict(by_class)}")
        print(f"  emotions: {dict(emotions)}")
        for r in rows:
            meta = json.loads(r["metadata"] or "{}") or {}
            inner = meta.get("inner_os") or ""
            print(f"  - [{r['memory_class']}/{meta.get('emotion', '?')}] "
                  f"{(r['created_at'] or '')[:10]} {r['content'][:78]}")
            if inner:
                print(f"      inner_os: {inner[:78]}")
        con.close()
    else:
        print("\nRIGHT BRAIN: (no right_brain.sqlite)")

    vm = make_vm(user_key, memory_root)
    try:
        store = vm._get_repo()._cognitive_store
        ents = store.list_entities(user_key) if hasattr(store, "list_entities") else []
        print(f"\nLEFT BRAIN entities: {len(ents)}")
    except Exception as e:
        print(f"\nLEFT BRAIN entities: (unavailable: {e})")
    print(f"corpus utterances: {len(USERS[user_key]['memories'])}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", default=",".join(USERS))
    ap.add_argument("--memory-root", default=str(DEFAULT_MEMORY_ROOT))
    args = ap.parse_args()
    for u in [x.strip() for x in args.users.split(",") if x.strip()]:
        dump_user(u, Path(args.memory_root))


if __name__ == "__main__":
    main()
