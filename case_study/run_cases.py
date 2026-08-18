"""Run the four paper cases against the two-user corpus.

    python case_study/run_cases.py ingest          # load the corpus (slow, LLM-bound)
    python case_study/run_cases.py query           # run the 4 inputs for both users
    python case_study/run_cases.py all --reset     # wipe, ingest, query

Each (user, case) produces three panels, which is exactly what the paper needs:

    LEFT BRAIN   SearchResult.hits            -- retrieved factual memories
    RIGHT BRAIN  rb_hits / rb_directive       -- heartnotes, response experience,
                 + prestimulus_text              persona and preference layer
    REPLY        the reply model's output, conditioned on both panels

Outputs land in case_study/out/ as results.json (machine-readable) and
cases.md (paper-ready side-by-side tables).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# mem0's telemetry opens a *global* qdrant collection under ~/.mem0 and takes an
# exclusive file lock on it, so a second mem0-backed process (a running demo
# server, or this script twice) dies with "already accessed by another instance
# of Qdrant client". Nothing here needs that collection. Must be set before
# voicemem_core -> mem0 is imported, which is why every voicemem import in this
# file is deferred into a function.
os.environ.setdefault("MEM0_TELEMETRY", "False")

from case_study.corpus import CASES, TODAY, USERS  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
DEFAULT_MEMORY_ROOT = ROOT / "memory"

SYSTEM_PROMPT = """You are a voice assistant with long-term memory of this user.

Today is {today}. The user just said: "{utterance}"

Below is what your memory system retrieved for this turn. The LEFT BRAIN block
is factual episodic memory. The RIGHT BRAIN block is affective and stylistic
memory: how this specific user tends to feel in situations like this one, and
what kinds of responses have worked or failed with them before.

Speak in one short spoken-voice turn (1-3 sentences, no lists, no markdown).
Use the memory the way a person who was there would: reference the specific
thing you remember rather than saying "I remember". Follow the right brain's
guidance on tone even when it contradicts a generically "supportive" reply.

Return JSON only:
{{"reply": "<what you say out loud>",
  "action": null | {{"type": "play_music", "track": "...", "artist": "...", "source": "spotify"}}}}

Emit an action only when the user's request implies one should be performed now.

--- RETRIEVED MEMORY ---
{memory}
--- END RETRIEVED MEMORY ---"""

BASELINE_PROMPT = """You are a voice assistant with no memory of this user.
Today is {today}. Reply in one short spoken-voice turn (1-3 sentences).
Return JSON only: {{"reply": "...", "action": null}}"""


# ── memory system ─────────────────────────────────────────────────────────────

def make_vm(user_key: str, memory_root: Path):
    """A text-only VoiceMem: the five audio perception heads are off, so this
    runs without the acoustic model stack. Emotion labels come from the corpus
    (standing in for the acoustic head) and are passed to Ingest directly."""
    from voicemem_core import VoiceMem
    return VoiceMem(
        memory_root=memory_root / user_key,
        user_id=user_key,
        enable_scene=False,
        enable_music=False,
        enable_abnormal_sound=False,
        enable_voiceprint=False,
        enable_emotion=False,
    )


def ingest_user(user_key: str, memory_root: Path) -> dict:
    entry = USERS[user_key]
    name = entry["profile"]["display_name"]
    vm = make_vm(user_key, memory_root)
    t0 = time.time()
    facts = 0
    for i, (date, text, emotion, ents) in enumerate(entry["memories"], 1):
        r = vm.Ingest(text, speaker=name, emotion=emotion, entities=list(ents),
                      observed_at=date)
        n = r.get("facts_count") or 0
        facts += n
        print(f"  [{user_key} {i:>2}/{len(entry['memories'])}] {date[:10]}  "
              f"facts={n}  {text[:64]}", flush=True)
    print(f"  [{user_key}] Flush() -- right-brain attribution + persona synthesis", flush=True)
    vm.Flush()
    dt = time.time() - t0
    print(f"  [{user_key}] done: {len(entry['memories'])} utterances, "
          f"{facts} facts, {dt:.1f}s", flush=True)
    return {"utterances": len(entry["memories"]), "facts": facts, "seconds": round(dt, 1)}


# ── one turn ──────────────────────────────────────────────────────────────────

def build_case_context(result) -> str:
    """Relabel one SearchResult into the paper's three panels.

    Same fields the library's build_memory_context() renders, but with the
    left/right split made explicit (and rb_hits kept at the full top-5 rather
    than the latency-motivated cap of 3), because the whole claim of these
    cases is that the two channels contribute different things.
    """
    blocks: list[str] = []
    if result.hits:
        blocks.append("LEFT BRAIN — what happened (episodic facts):\n" +
                      "\n".join(f"- {h.text}" for h in result.hits))
    if result.rb_hits:
        blocks.append("RIGHT BRAIN — how this user works (affect, style, experience):\n" +
                      "\n".join(f"- [{h.source}] {h.content}" for h in result.rb_hits))
    # The low-confidence abstention hint lives only on rb_directive, never on
    # rb_hits -- carry it across so the reply model still sees it.
    extra = [ln for ln in (result.rb_directive or "").splitlines()
             if ln.strip() and all(ln.strip() not in h.content for h in result.rb_hits)]
    if extra:
        blocks.append("\n".join(extra))
    if result.prestimulus_text:
        blocks.append("PRE-STIMULUS — persona and standing preferences:\n" +
                      result.prestimulus_text)
    return "\n\n".join(blocks)


def run_turn(vm, case: dict) -> dict:
    from voicemem_core import build_memory_context
    utterance = case["input"]
    t0 = time.time()
    c = vm.Classify(utterance)
    result = vm.Search(
        utterance,
        slots=list(c.slots),
        entities=list(c.entities),
        emotion=case.get("emotion_hint") or None,
    )
    retrieval_ms = round((time.time() - t0) * 1000)

    return {
        "classification": {"slots": list(c.slots), "entities": list(c.entities)},
        "left_brain": [
            {"text": h.text, "score": round(h.score, 3), "memory_id": h.memory_id}
            for h in result.hits
        ],
        "right_brain": [
            {"content": h.content, "source": h.source, "priority": round(h.priority, 3),
             "emotion": (h.metadata or {}).get("emotion", "")}
            for h in result.rb_hits
        ],
        "rb_directive": result.rb_directive,
        "prestimulus": result.prestimulus_text,
        "memory_context": build_case_context(result),
        "library_context": build_memory_context(result),
        "retrieval_ms": retrieval_ms,
        "timing": result.timing,
    }


# ── reply model ───────────────────────────────────────────────────────────────

def _client():
    from openai import OpenAI
    return OpenAI(base_url=os.environ.get("OPENAI_BASE_URL") or None)


def generate_reply(memory_context: str, utterance: str, model: str) -> dict:
    prompt = SYSTEM_PROMPT.format(today=TODAY, utterance=utterance,
                                  memory=memory_context or "(nothing retrieved)")
    return _chat(prompt, utterance, model)


def generate_baseline(utterance: str, model: str) -> dict:
    return _chat(BASELINE_PROMPT.format(today=TODAY), utterance, model)


def _chat(system: str, utterance: str, model: str) -> dict:
    resp = _client().chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": utterance}],
        temperature=0.6,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"reply": raw.strip(), "action": None}
    return {"reply": str(data.get("reply", "")).strip(), "action": data.get("action")}


# ── rendering ─────────────────────────────────────────────────────────────────

def render_markdown(results: dict) -> str:
    L = ["# VoiceMem case study", "",
         f"Corpus: {sum(len(USERS[u]['memories']) for u in USERS)} utterances across "
         f"{len(USERS)} users. Today = {TODAY}.", ""]
    for u in results["users"]:
        p = USERS[u]["profile"]
        L += [f"- **{p['display_name']}** (`{u}`) — {p['blurb']}"]
    L += [""]

    for case in CASES:
        cid = case["id"]
        L += [f"## Case {cid} — {case['name']}", "",
              f"**Input:** “{case['input']}”  ",
              f"**Core insight:** {case['insight']}", ""]
        for u in results["users"]:
            r = results["cases"][cid][u]
            p = USERS[u]["profile"]
            L += [f"### {p['display_name']}", "",
                  "**Left brain (retrieved facts)**", ""]
            L += [f"- `{h['score']:.3f}` {h['text']}" for h in r["left_brain"]] or ["- (none)"]
            L += ["", "**Right brain (affect / style / experience)**", ""]
            L += [f"- `{h['source']}` {h['content']}" for h in r["right_brain"]] or ["- (none)"]
            if r["rb_directive"]:
                L += ["", "> " + r["rb_directive"].replace("\n", "\n> ")]
            if r["prestimulus"]:
                L += ["", "**Pre-stimulus (persona + preferences)**", "",
                      "```", r["prestimulus"], "```"]
            L += ["", "**Reply**", "", f"> {r['reply']['reply']}"]
            if r["reply"].get("action"):
                L += ["", f"**Action:** `{json.dumps(r['reply']['action'], ensure_ascii=False)}`"]
            if r.get("baseline"):
                L += ["", f"*No-memory baseline:* {r['baseline']['reply']}"]
            L += [f"", f"<sub>retrieval {r['retrieval_ms']} ms</sub>", ""]
    return "\n".join(L)


# ── stages ────────────────────────────────────────────────────────────────────

def stage_ingest(users: list[str], memory_root: Path) -> dict:
    stats = {}
    for u in users:
        print(f"[ingest] {u}", flush=True)
        stats[u] = ingest_user(u, memory_root)
    return stats


def stage_query(users: list[str], memory_root: Path, model: str, baseline: bool) -> dict:
    out = {"users": users, "today": TODAY, "model": model,
           "cases": {c["id"]: {} for c in CASES}}
    for u in users:
        vm = make_vm(u, memory_root)
        for case in CASES:
            print(f"[query] {u} case {case['id']}: {case['input']}", flush=True)
            turn = run_turn(vm, case)
            turn["reply"] = generate_reply(turn["memory_context"], case["input"], model)
            if baseline:
                turn["baseline"] = generate_baseline(case["input"], model)
            out["cases"][case["id"]][u] = turn
            print(f"         left={len(turn['left_brain'])} right={len(turn['right_brain'])} "
                  f"{turn['retrieval_ms']}ms", flush=True)
            print(f"         > {turn['reply']['reply']}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["ingest", "query", "all"])
    ap.add_argument("--users", default=",".join(USERS))
    ap.add_argument("--memory-root", default=str(DEFAULT_MEMORY_ROOT))
    ap.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    ap.add_argument("--reset", action="store_true",
                    help="delete the memory root before ingesting")
    ap.add_argument("--baseline", action="store_true",
                    help="also generate a no-memory reply for contrast")
    args = ap.parse_args()

    users = [u.strip() for u in args.users.split(",") if u.strip()]
    unknown = [u for u in users if u not in USERS]
    if unknown:
        sys.exit(f"unknown users: {unknown}; known: {list(USERS)}")

    memory_root = Path(args.memory_root)
    OUT.mkdir(parents=True, exist_ok=True)

    if args.reset and memory_root.exists():
        print(f"[reset] removing {memory_root}", flush=True)
        shutil.rmtree(memory_root)

    if args.stage in ("ingest", "all"):
        stats = stage_ingest(users, memory_root)
        (OUT / "ingest_stats.json").write_text(
            json.dumps(stats, indent=2), encoding="utf-8")

    if args.stage in ("query", "all"):
        results = stage_query(users, memory_root, args.model, args.baseline)
        (OUT / "results.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        (OUT / "cases.md").write_text(render_markdown(results), encoding="utf-8")
        print(f"\n[done] {OUT / 'results.json'}\n       {OUT / 'cases.md'}", flush=True)


if __name__ == "__main__":
    main()
