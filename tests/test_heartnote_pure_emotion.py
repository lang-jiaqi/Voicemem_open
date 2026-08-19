"""Real bug this session: core.py's ``_finish_ingest`` gated the right-brain
heartnote write on ``if emotion and result.memory_ids:`` -- wrongly coupling
two unrelated things: "was there an emotion signal" and "did THIS turn's own
text also produce an extractable left-brain fact". Real repro: "I'm so
excited, I just got promoted at work today!" -- a pure emotion-expressing
sentence with no new fact to extract (``facts_count=0``, ``memory_ids=[]``,
too colloquial/nothing new to store), but the emotion classifier correctly
returned "excited". Under the old logic the heartnote was silently skipped
entirely, so the right-brain panel stayed permanently empty for exactly the
turns most worth capturing emotionally. Fixed to ``if emotion:`` alone
gating the write, with the evidence memory id (``mid``) independently
nullable. This test drives ``VoiceMem._finish_ingest`` directly (mocking out
the LLM-backed fact extraction and inner-OS generation, matching the
``test_clap_environment_fallback.py`` pattern for exercising core.py's
internal methods without live network calls) and asserts the heartnote is
still written when memory_ids is empty.
"""
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from voicemem import VoiceMem
from voicemem.utils.common.voice_input import VoiceIngestResult


def _ctx(**overrides) -> dict:
    base = dict(
        text="I'm so excited, I just got promoted at work today!",
        speaker="Speaker 0", emotion="excited", entities=[], session_id="sess1",
        audio_path=None, observed_at=None, ts="12:00:00",
        environment="", scene_tag=None, scene_raw_labels=[], person_id=None,
        tune_result=None, abnormal_hits=[], place_result=None, new_routine=None,
        detection=None, environment_hint="",
    )
    base.update(overrides)
    return base


class HeartnotePureEmotionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.vm = VoiceMem(memory_root=self.tmp, user_id="u1")
        self.fake_rb_repo = MagicMock()
        self.fake_rb_repo._store.upsert_memory.return_value = MagicMock(id="rb_mem_1")
        self.vm._cache["rb_repo"] = self.fake_rb_repo
        # _get_repo()/_get_extractor() eagerly construct real OpenAI clients
        # (fail without a real OPENAI_API_KEY) -- ingest_voice_input() itself
        # is mocked below so these values are never actually used, they just
        # need to exist so evaluating them as call arguments doesn't crash.
        self.vm._cache["repo"] = MagicMock()
        self.vm._cache["extractor"] = MagicMock()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_heartnote_written_when_emotion_present_but_no_extractable_fact(self):
        # This is the exact real repro: emotion detected, but nothing new for
        # the left brain to store this turn (no fact extracted).
        no_fact_result = VoiceIngestResult(
            voice_id="v1", memory_ids=[], facts_count=0,
            begin_time="12:00:00", end_time="12:00:00", slots=[], messages_count=1,
        )
        with patch("voicemem.utils.common.voice_input.ingest_voice_input", return_value=no_fact_result), \
             patch.object(self.vm, "_generate_inner_os", return_value=""):
            self.vm._finish_ingest(_ctx())

        self.fake_rb_repo._store.upsert_memory.assert_called_once()
        _, kwargs = self.fake_rb_repo._store.upsert_memory.call_args
        self.assertEqual(kwargs["memory_class"], "heartnote")
        self.assertEqual(kwargs["metadata"]["emotion"], "excited")
        # No fact this turn -> no evidence memory id, but the write still happens.
        self.assertEqual(kwargs["evidence_memory_ids"], [])

    def test_heartnote_written_and_linked_when_fact_also_extracted(self):
        with_fact_result = VoiceIngestResult(
            voice_id="v1", memory_ids=["mem_123"], facts_count=1,
            begin_time="12:00:00", end_time="12:00:00", slots=[], messages_count=1,
        )
        with patch("voicemem.utils.common.voice_input.ingest_voice_input", return_value=with_fact_result), \
             patch.object(self.vm, "_generate_inner_os", return_value=""):
            self.vm._finish_ingest(_ctx())

        self.fake_rb_repo._store.upsert_memory.assert_called_once()
        _, kwargs = self.fake_rb_repo._store.upsert_memory.call_args
        self.assertEqual(kwargs["evidence_memory_ids"], ["mem_123"])

    def test_no_heartnote_when_no_emotion(self):
        no_fact_result = VoiceIngestResult(
            voice_id="v1", memory_ids=[], facts_count=0,
            begin_time="12:00:00", end_time="12:00:00", slots=[], messages_count=1,
        )
        with patch("voicemem.utils.common.voice_input.ingest_voice_input", return_value=no_fact_result), \
             patch.object(self.vm, "_generate_inner_os", return_value=""):
            self.vm._finish_ingest(_ctx(emotion=""))

        self.fake_rb_repo._store.upsert_memory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
