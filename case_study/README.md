# case_study — the four paper cases

Two users with contrasting personalities live through the same kind of month.
The four case inputs are then replayed for both, so any difference in the
output comes from memory alone.

| # | Input | Core insight |
|---|---|---|
| 01 | "I'm feeling tired today." | Recalls past events to interpret the current state (last week's interview loop). |
| 02 | "I'm fine." | Infers hidden emotion beyond the literal text — same words, different meaning per user. |
| 03 | "My boss criticized my proposal today." | Adapts tone to personality — acknowledge-first vs. fix-first. |
| 04 | "What song was I listening to yesterday?" | Recalls a preference and emits the action that follows from it. |

| User | Personality | Why it matters |
|---|---|---|
| `maya` | Reserved; understates distress; "I'm fine" means "not yet"; shuts down when handed a fix before acknowledgement. | Drives cases 01/02/04 and the acknowledge-first half of 03. |
| `ryan` | Blunt and literal; "I'm fine" means fine; wants ranked next steps; irritated by emotional validation. | The contrast half of 02 and 03. |

## Files

| File | Role |
|---|---|
| `corpus.py` | 62 dated utterances (33 + 29) with emotion labels and entity hints, plus the four case definitions. |
| `run_cases.py` | Ingests the corpus, then runs the four inputs and captures left brain / right brain / reply. |
| `out/results.json` | Machine-readable output for every (user, case). |
| `out/cases.md` | Paper-ready side-by-side write-up. |

## Run

```bash
export OPENAI_API_KEY=sk-...        # extraction, classification, embeddings, reply
export OPENAI_MODEL=gpt-4o-mini     # optional
python case_study/run_cases.py all --reset
```

Stages can be run separately — `ingest` is the slow, LLM-bound one and only
needs to happen once per corpus change:

```bash
python case_study/run_cases.py ingest --reset
python case_study/run_cases.py query --baseline   # --baseline adds a no-memory reply
```

## What each panel is

- **Left brain** — `SearchResult.hits`: episodic facts retrieved through slot
  routing → entity narrowing → graph expansion → vector rank.
- **Right brain** — `SearchResult.rb_hits` / `rb_directive`: heartnotes
  (situation→emotion regularities), response experience (what has worked with
  this user), and relation/trait nodes. Written during `Ingest` whenever an
  emotion label is present, consolidated at `Flush`.
- **Pre-stimulus** — `SearchResult.prestimulus_text`: the slow-moving persona
  document and stable preferences, injected unconditionally.
- **Reply** — the reply model conditioned on all of the above, returning
  `{reply, action}` so case 04 can end in an actual `play_music` call.

Emotion labels in the corpus stand in for the acoustic emotion head; the audio
perception stack is switched off (`enable_scene/music/voiceprint/... = False`)
so the case study runs without the model downloads.
