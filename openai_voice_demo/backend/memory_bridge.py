"""The ONLY module that touches `voicemem`. Both providers call through here —
neither imports `voicemem` directly, so there is exactly one place that knows
about VoiceMem's constructor flags, Ingest()/Search() signatures, and the
audio-native on/off toggle.
"""
from __future__ import annotations

import asyncio
import uuid

from config import Config

from voicemem.engine import VoiceMem, SearchResult

from audio_utils import write_wav_file
from local_classifier import LocalSlotClassifier
from local_embedder import LocalMemoryEmbedder
from local_emotion_classifier import LocalEmotionClassifier

# core.py only *auto*-detects emotion when the caller passes no emotion at
# all (`if self._enable_emotion and not emotion:`) -- an explicitly-passed
# emotion is used regardless of `enable_emotion` (tied to `audio_native`,
# which this demo defaults to False/text-only), so a cheap text-only
# classifier here is enough to unblock the right brain without needing the
# heavier audio-native install. See local_emotion_classifier.py.


class MemoryBridge:
    def __init__(self, config: Config) -> None:
        self._config = config
        # One SentenceTransformer load (~8s incl. model weights) shared by
        # the slot classifier, the memory embedder, and the emotion
        # classifier below -- same underlying model, no reason to load it
        # three times.
        from sentence_transformers import SentenceTransformer
        shared_model = SentenceTransformer("intfloat/multilingual-e5-small")
        # See local_classifier.py for what this replaces (Classify()'s LLM
        # call) and the real accuracy numbers measured for it.
        self._local_classifier = LocalSlotClassifier(model=shared_model)
        # Real-time-conversation speed path: local embedding instead of an
        # OpenAI embeddings API round trip for every Ingest()/Search()'s
        # Rank() step. voicemem's own official default remains OpenAI
        # embeddings (see local_embedder.py for the real trade-off and why
        # this is opt-in, demo-only). Real, disclosed cost: this changes
        # vector dimensions (384 vs. OpenAI's 1536), so it is NOT compatible
        # with a memory store that was built using OpenAI embeddings -- an
        # existing `config.memory_root` from before this switch must be
        # reset, not just reused.
        local_embedder = LocalMemoryEmbedder(model=shared_model)
        # Was an LLM call (gpt-4o-mini) -- the last remaining real network
        # round trip in search()/ingest() once the two above were made
        # local. See local_emotion_classifier.py: unlike the other two,
        # there was no pre-existing "official core voicemem" version of
        # text-only emotion classification to preserve here, so this is now
        # simply the implementation, not a demo-only opt-in path.
        self._emotion_classifier = LocalEmotionClassifier(model=shared_model)
        # The five enable_* flags ARE the entire audio-native on/off switch —
        # voicemem already has exactly this knob built in, nothing new to write.
        self._vm = VoiceMem(
            memory_root=config.memory_root,
            user_id=config.user_id,
            enable_scene=config.audio_native,
            enable_music=config.audio_native,
            enable_abnormal_sound=config.audio_native,
            enable_voiceprint=config.audio_native,
            enable_emotion=config.audio_native,
            embedder=local_embedder,
        )

    @property
    def vm(self) -> VoiceMem:
        """底层 VoiceMem 实例（供启动自检 startup_check 复用同一实例测速）。"""
        return self._vm

    async def search(self, query: str, **kwargs) -> SearchResult:
        # voicemem's own canonical pattern (see VoiceMem.PrimeSubgraphFromQuery
        # in voicemem/core.py) is Classify() -> Search(slots=, entities=) --
        # skipping Classify() was a real gap in this demo: without it, slots/
        # entities are always empty, Search() has nothing to narrow against,
        # and silently falls back to an unfiltered full-library search.
        # Search() itself now auto-records Algorithm 1's query-activation
        # bookkeeping as a side effect (see core.py::_record_subgraph_activation),
        # so this Classify()->Search() pair already IS the complete pattern --
        # calling PrimeSubgraphFromQuery() separately would just redundantly
        # repeat both calls, not add anything PrimeSubgraphFromQuery does that
        # Search() doesn't already do on its own.
        #
        # slots come from the LOCAL classifier (local_classifier.py), not
        # voicemem's own Classify() (an LLM call, ~1.2s mean measured this
        # session) -- see that module for the real speed/accuracy numbers
        # behind this swap. voicemem's own Classify() is untouched and
        # remains the standard implementation; this is a demo-only speed
        # path. Real, disclosed cost: no entities (cosine similarity can't
        # do open-vocabulary NER), so Search() runs in its existing
        # "slot-only" narrowing mode rather than slot+entity.
        #
        # Real gap found via live testing: Search() takes an `emotion=` that
        # feeds the right brain's emotion-trait lookup (core.py's
        # _rb_emotion_trait_hit, matched against the SAME normalized-emotion
        # anchor set at ingest time -- see local_emotion_classifier.py),
        # but this was never passed here, so that specific right-brain
        # source could never surface no matter what was in the store.
        #
        # Both are independent local model calls now (neither is a network
        # round trip anymore), so run them concurrently -- costs nothing,
        # and there's no reason to serialize two unrelated calls.
        slots, emotion = await asyncio.gather(
            asyncio.to_thread(self._local_classifier.classify, query),
            asyncio.to_thread(self._emotion_classifier.classify, query),
        )
        kwargs.setdefault("emotion", emotion)
        return await asyncio.to_thread(
            self._vm.Search, query, slots=slots, entities=[], **kwargs)

    async def flush(self) -> None:
        """Call once when a conversation/connection formally ends -- runs the
        session-boundary batch voicemem itself needs (Algorithm 1 subgraph
        checkpoint + right-brain long/short-term attribution) that otherwise
        only fires on the *next* session's Ingest() detecting a session_id
        change. Without this, the last real session of a demo run never gets
        its batch processing done at all. See VoiceMem.Flush() in core.py.
        """
        await asyncio.to_thread(self._vm.Flush)

    async def ingest(self, text: str, *, pcm: bytes | None = None, turn_id: str | None = None,
                      **kwargs) -> dict:
        audio_path = None
        if self._config.audio_native and pcm:
            turn_id = turn_id or uuid.uuid4().hex
            wav_path = self._config.raw_audio_dir / f"{turn_id}.wav"
            write_wav_file(wav_path, pcm)
            audio_path = str(wav_path)
        if not self._config.audio_native and not kwargs.get("emotion"):
            kwargs["emotion"] = await asyncio.to_thread(self._emotion_classifier.classify, text)
        return await asyncio.to_thread(self._vm.Ingest, text, audio_path=audio_path, **kwargs)

    def build_instructions(self, base_prompt: str, result: SearchResult) -> str:
        """Turn a SearchResult into a system-prompt string. Shared by both
        providers so neither reimplements memory-context formatting.
        """
        lines = [base_prompt]
        if result.hits:
            lines.append("\n你记得用户的这些事(这是你脑子里的背景,自然地织进对话,"
                         "一次带一两个细节就够,别罗列、别复述原文):")
            for hit in result.hits:
                lines.append(f"- {hit.text}")
        if getattr(result, "prestimulus_text", None):
            lines.append(f"\n{result.prestimulus_text}")
        # Top 3, not all 5 (result.rb_hits is already priority-sorted, see
        # core.py's Search()) -- real, measured cost found via live testing:
        # once the right-brain bug fix (cognitive_graph/store.py) made this
        # actually return real content instead of silently crashing to
        # empty, the full 5-item rb_directive block added a real few hundred
        # characters to every system prompt, on the direct path to how long
        # GPT-4o/gpt-realtime take to start replying. Trading a little
        # persona/emotional context depth for real, felt response speed.
        rb_hits = getattr(result, "rb_hits", None) or []
        if rb_hits:
            lines.append("\n你感受到的用户内心状态(只用来共情和把握语气,不要念出来):")
            lines.append("\n".join(h.content for h in rb_hits[:3]))
        return "\n".join(lines)

    @staticmethod
    def build_rag_items(result: SearchResult) -> list[dict]:
        """Same SearchResult, rendered as external-RAG items ({title, content})
        instead of a prompt string -- the shape 豆包's ChatRAGText event wants
        (a JSON array of titled snippets, not one instructions blob). Lives
        here for the same reason build_instructions does: one place knows how
        to turn a SearchResult into model-facing context. Top-3 rb_hits for
        the same measured latency reason documented in build_instructions.
        """
        items = [{"title": "用户记忆", "content": h.text} for h in result.hits]
        if getattr(result, "prestimulus_text", None):
            items.append({"title": "当下情境", "content": result.prestimulus_text})
        items.extend({"title": "用户画像", "content": h.content}
                     for h in (getattr(result, "rb_hits", None) or [])[:3])
        return items

    @staticmethod
    def search_payload(result: SearchResult) -> dict:
        """Left-brain hits and right-brain results, kept as separate fields
        so the client can actually show voicemem's two-hemisphere split
        instead of a single flattened list.

        ``right_brain_hits`` is voicemem's structured top-5 right-brain list
        (each item tagged with its source: response_experience/
        situation_pattern/relation/emotion_trait/profile/current_signal) --
        ``right_brain`` is kept alongside it as the same list pre-rendered to
        a single text block, for callers that just want a prompt string.
        """
        return {
            "left_brain": [
                {"text": h.text, "score": h.score, "attributed_to": h.attributed_to}
                for h in result.hits
            ],
            "right_brain": getattr(result, "rb_directive", None) or None,
            "right_brain_hits": [
                {"content": h.content, "source": h.source, "priority": h.priority}
                for h in getattr(result, "rb_hits", None) or []
            ],
            "current_scene": getattr(result, "current_scene", None) or None,
            # 每个 slot 的会话级概括性摘要（voicemem 维护的 slot_summaries）——
            # 前端用作簇标签的悬停描述。{slot: summary_text}
            "related_summaries": getattr(result, "related_summaries", None) or {},
        }
