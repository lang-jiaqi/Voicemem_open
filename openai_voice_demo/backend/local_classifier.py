"""Local embedding-based slot classification -- ONLY used by this demo
(openai_voice_demo), as a fast local alternative to Classify()'s LLM call.
voicemem's own core Classify() (voicemem/leftbrain/cognitive_graph/
query_slot_classifier.py) is left completely untouched and remains the
canonical/standard implementation -- this file does not replace or modify
it, it's an opt-in speed path scoped to this demo only.

Real numbers from this session's own testing (classify_embedding_test.py,
16 real queries against voicemem's own few-shot ground truth): embedding-
based slot picking agreed with the LLM's own answers on 15/16 (94%, top-2),
but using OpenAI's *remote* embedding API for it was NOT meaningfully
faster than the LLM call (536ms mean) -- the real bottleneck was always the
network round trip, not "LLM reasoning" specifically. A LOCAL model removes
that round trip entirely.

Model choice, also tested for real: the first local model tried
(all-MiniLM-L6-v2, English-only) only hit 57% (8/14) on a real bilingual
test set -- too lossy, this demo gets real Chinese queries. Switched to
intfloat/multilingual-e5-small (already cached on this machine), which hit
93% (13/14) on the same test, at ~4.7ms per query on this machine's GPU --
matching the remote-embedding accuracy while being ~100x faster and using
zero network calls. E5 models need the documented "query: " / "passage: "
prefix convention to perform well -- this is not optional decoration, tested
both with and without it.

Real limitation, not glossed over: Classify() ALSO extracts open-vocabulary
entities (named people/places/things) from the query text, which cosine
similarity fundamentally cannot do (there's no fixed target list to compare
against). This module only ever returns slots=[...] -- callers fall back to
voicemem's existing, already-supported "slot-only" narrowing mode (visible
in core.py's own Search() logs as `mode=slot-only`), not a new/degraded
state.
"""
from __future__ import annotations

import numpy as np

# Same canonical slot descriptions voicemem's own LLM classifier uses (see
# voicemem/leftbrain/cognitive_graph/query_slot_classifier.py's _BASE_SLOTS)
# -- kept in sync deliberately, this approximates the SAME classifier's
# slot decision locally, not a different taxonomy.
_SLOT_DESCRIPTIONS = {
    "work": "career, job, company, projects, colleagues, workplace, 工作, 职业, 辞职, 升职",
    "finance": "money, salary, income, expenses, investments, savings, 财务, 薪资, 投资",
    "relationships": "friends, family, romantic, social connections, 朋友, 家人, 感情",
    "health": "physical health, exercise, diet, sleep, medical, 健康, 运动, 生病",
    "goals": "future plans, dreams, aspirations, self-improvement, 目标, 计划, 梦想",
    "daily_life": "daily routines, hobbies, leisure, lifestyle, 日常, 爱好, 习惯",
    "knowledge": "learning, concepts, skills, facts, technology, 知识, 技能, 学习",
}


class LocalSlotClassifier:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-small", model=None) -> None:
        # model=: reuse an already-loaded SentenceTransformer (e.g. shared
        # with local_embedder.py's LocalMemoryEmbedder) instead of loading a
        # second copy of the same ~470MB model into memory.
        if model is not None:
            self._model = model
        else:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
        self._slot_names = list(_SLOT_DESCRIPTIONS.keys())
        slot_texts = [f"passage: {k}: {v}" for k, v in _SLOT_DESCRIPTIONS.items()]
        embs = self._model.encode(slot_texts, normalize_embeddings=True)
        self._slot_embs = np.asarray(embs)

    def classify(self, query: str, top_k: int = 2) -> list[str]:
        q = self._model.encode([f"query: {query}"], normalize_embeddings=True)[0]
        sims = self._slot_embs @ np.asarray(q)
        order = np.argsort(-sims)[:top_k]
        return [self._slot_names[i] for i in order]
