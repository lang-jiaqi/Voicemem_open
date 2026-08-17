"""Local embedding-based emotion classification -- replaces the LLM call
(`gpt-4o-mini`, see memory_bridge.py's old `_classify_emotion`) that used to
be the only real network round trip left in `memory_bridge.search()` once
slot classification (local_classifier.py) and memory ranking
(local_embedder.py) were both made local.

Unlike those two, there was never an "official core voicemem" version of
this to preserve as a fallback: `_classify_emotion` was itself a
demo-only bridge invented to unblock the right brain when
`audio_native=False` (voicemem's own real emotion detection only runs on
actual audio, via `PaperAlignedEmotionDetector`) -- see the comment that
used to sit next to it in memory_bridge.py. So this isn't a "keep OpenAI as
standard, add a local option" split like the other two; there is nothing
else to keep, this local classifier IS now the only implementation, in both
the demo and (nothing else currently calls text-only emotion classification
at all, so "core" doesn't have a competing version here).

The right brain's emotion anchors are a FIXED set of 8 Chinese labels
(``voicemem/rightbrain/anchor_router.py``'s ``_CANONICAL_EMOTIONS`` /
``graph_store.py``'s ``SEED_SLOTS`` "情绪" entities) -- structurally the
exact same "pick the closest of N fixed categories" problem
local_classifier.py already solved for slots, so the same approach applies
directly: cosine similarity against each label's description, using the
same E5 model (shared instance, see memory_bridge.py) with the "query: "/
"passage: " prefix convention.

Anchor text per label is hand-copied from anchor_router.py's own
``_EMOTION_KEYWORDS``/``_EMOTION_KEYWORDS_EN`` (not imported -- those are
private module names, and this mirrors local_classifier.py's own precedent
of hand-copying voicemem's canonical descriptions rather than reaching into
another module's private state) -- kept in sync deliberately, this is the
SAME taxonomy voicemem's own keyword-matching fallback (``normalize_emotion``)
uses, not a parallel one. Output is one of the 8 canonical Chinese labels
directly, which `normalize_emotion()` passes through unchanged (it checks
canonical membership first before falling back to keyword search), so this
stays fully compatible with the existing right-brain anchor matching.
"""
from __future__ import annotations

import numpy as np

# Mirrors anchor_router.py's _EMOTION_KEYWORDS + _EMOTION_KEYWORDS_EN,
# grouped by canonical label instead of flattened to (keyword, canonical)
# pairs -- same words, just reshaped into one description string per label
# for embedding.
_EMOTION_DESCRIPTIONS = {
    "焦虑": "焦虑 压力 紧张 担忧 恐惧 害怕 不安 慌 "
            "anxious anxiety nervous worried worry stressed stress tense "
            "fearful afraid scared panicked panic uneasy apprehensive",
    "悲伤": "悲伤 难过 失落 沮丧 伤心 绝望 崩溃 "
            "sad sadness upset depressed disappointed heartbroken miserable "
            "dejected despair sorrowful grief",
    "委屈": "委屈 愤怒 气愤 不满 憋屈 不公平 "
            "wronged angry anger mad furious irritated annoyed frustrated "
            "resentful indignant unfair bitter",
    "孤独": "孤独 空虚 寂寞 "
            "lonely loneliness isolated empty alone",
    "纠结": "纠结 矛盾 迷茫 犹豫 "
            "conflicted torn confused uncertain hesitant ambivalent indecisive "
            "perplexed lost",
    "平静": "平静 淡然 冷静 释然 坦然 "
            "calm relaxed peaceful composed serene settled neutral",
    "开心": "开心 高兴 兴奋 期待 愉快 满足 自豪 憧憬 感激 轻松 坚定 "
            "happy happiness joy joyful excited excitement glad pleased "
            "delighted proud grateful thankful relieved hopeful cheerful "
            "satisfied content amused",
    "疲惫": "疲惫 疲倦 困倦 无力 累 "
            "tired exhausted fatigue fatigued weary drained sleepy worn out",
}


class LocalEmotionClassifier:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-small", model=None) -> None:
        if model is not None:
            self._model = model
        else:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
        self._labels = list(_EMOTION_DESCRIPTIONS.keys())
        passage_texts = [f"passage: {label}: {desc}" for label, desc in _EMOTION_DESCRIPTIONS.items()]
        embs = self._model.encode(passage_texts, normalize_embeddings=True)
        self._label_embs = np.asarray(embs)

    def classify(self, text: str) -> str:
        """Returns one of the 8 canonical Chinese emotion labels."""
        q = self._model.encode([f"query: {text}"], normalize_embeddings=True)[0]
        sims = self._label_embs @ np.asarray(q)
        return self._labels[int(np.argmax(sims))]
