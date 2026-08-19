"""右脑检索/排序/演化这一轮优化的机制级测试（确定性，不调 LLM）。

覆盖的改动（每条对应 benchmark 上的一个弱项）：
  1. anchor_score 从 SQL 一路带回 RightBrainMemory，并混进最终 hit 排序
     ——命中查询锚点的具体 heartnote 不再被静态 priority 压在泛化画像下面
     （ES-MemEval IE / UM）。
  2. 渲染层每条记忆带 [YYYY-MM-DD] 日期（temporal reasoning）。
  3. 矛盾旧况打 superseded 标记保留 + 渲染标注 + 降权，而不是删除
     （PersonaMem Evol / conflict detection）。
  4. heartnote content 存原话，inner_os 移到 metadata 作补充渲染（IE）。
  5. normalize_emotion 不再把未识别情绪兜底成"平静"高权重锚点。
  6. 归因精炼每条记忆只做一次（refined 标记），防止反复有损重写。
"""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from voicemem.rightbrain.anchor_router import (
    AnchorRouter,
    normalize_emotion,
    normalize_emotion_strict,
)
from voicemem.rightbrain.store import RightBrainStore
from voicemem.rightbrain.types import (
    CurrentSignals,
    MemoryAnchor,
    RightBrainContext,
    RightBrainMemory,
)


def _mk_mem(**overrides) -> RightBrainMemory:
    base = dict(
        id="m1", user_id="u1", memory_class="heartnote",
        content="和 Lisa 吵架了，很难受", condition=None,
        priority=0.5, confidence=1.0, ttl="long_term",
        metadata={}, evidence_turn_ids=[], evidence_memory_ids=[],
        created_at="2026-08-01T10:00:00+00:00",
        updated_at="2026-08-01T10:00:00+00:00",
    )
    base.update(overrides)
    return RightBrainMemory(**base)


class AnchorScorePropagationTests(unittest.TestCase):
    """anchor_score 必须从 search_by_anchors 的 SQL 带回 dataclass。"""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.store = RightBrainStore(self.tmp / "rb.sqlite")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_search_by_anchors_carries_anchor_score(self):
        m = self.store.upsert_memory("u1", "heartnote", "跟 Lisa 吵架了")
        self.store.link_anchor(m.id, "u1", MemoryAnchor(
            anchor_type="entity", anchor_id="lisa", role="subject",
            weight=1.0, confidence=1.0,
        ))
        got = self.store.search_by_anchors("u1", [MemoryAnchor(
            anchor_type="entity", anchor_id="lisa", role="subject",
        )])
        self.assertEqual(len(got), 1)
        self.assertGreater(got[0].anchor_score, 0.0)

    def test_get_all_defaults_anchor_score_zero(self):
        self.store.upsert_memory("u1", "heartnote", "随手一条")
        got = self.store.get_all("u1")
        self.assertEqual(got[0].anchor_score, 0.0)

    def test_multi_anchor_hit_scores_higher(self):
        strong = self.store.upsert_memory("u1", "heartnote", "命中两个锚点")
        weak   = self.store.upsert_memory("u1", "heartnote", "只命中一个")
        for aid in ("lisa", "焦虑"):
            self.store.link_anchor(strong.id, "u1", MemoryAnchor(
                anchor_type="entity" if aid == "lisa" else "emotion",
                anchor_id=aid, role="subject", weight=1.0, confidence=1.0,
            ))
        self.store.link_anchor(weak.id, "u1", MemoryAnchor(
            anchor_type="entity", anchor_id="lisa", role="subject",
            weight=1.0, confidence=1.0,
        ))
        got = self.store.search_by_anchors("u1", [
            MemoryAnchor(anchor_type="entity", anchor_id="lisa", role="subject"),
            MemoryAnchor(anchor_type="emotion", anchor_id="焦虑", role="trigger"),
        ])
        self.assertEqual(got[0].id, strong.id)
        self.assertGreater(got[0].anchor_score, got[1].anchor_score)

    def test_merge_metadata_roundtrip(self):
        m = self.store.upsert_memory("u1", "heartnote", "旧偏好", metadata={"emotion": "开心"})
        self.store.merge_metadata(m.id, {"superseded_by": "new-id"})
        got = self.store.get_memory(m.id)
        self.assertEqual(got.metadata["superseded_by"], "new-id")
        self.assertEqual(got.metadata["emotion"], "开心")  # 原有 key 不丢


class HitRankingAndRenderingTests(unittest.TestCase):
    """_rb_ctx_to_hits：混合排序、日期、inner_os、superseded 标注。"""

    def test_blended_priority_lets_specific_heartnote_beat_static_profiles(self):
        from voicemem.engine import _rb_blended_priority
        hot = _mk_mem(anchor_score=2.0)          # 强锚点命中的 heartnote
        cold = _mk_mem(anchor_score=0.0)
        self.assertAlmostEqual(_rb_blended_priority(cold), 0.5)
        self.assertGreater(_rb_blended_priority(hot), 0.75)   # 超过 emotion_trait
        self.assertLess(_rb_blended_priority(hot), 1.0)

    def test_hit_content_has_date_prefix(self):
        from voicemem.engine import _rb_ctx_to_hits
        ctx = RightBrainContext(situation_patterns=[_mk_mem()])
        hits = _rb_ctx_to_hits(ctx)
        self.assertIn("[2026-08-01]", hits[0].content)

    def test_inner_os_rendered_as_supplement_not_replacement(self):
        from voicemem.engine import _rb_ctx_to_hits
        m = _mk_mem(metadata={"inner_os": "【难过】TA 和好友起了冲突"})
        hits = _rb_ctx_to_hits(RightBrainContext(situation_patterns=[m]))
        self.assertIn("和 Lisa 吵架了", hits[0].content)          # 原话在
        self.assertIn("【难过】TA 和好友起了冲突", hits[0].content)  # 内心OS也在

    def test_superseded_memory_labeled_and_demoted_but_kept(self):
        from voicemem.engine import _rb_ctx_to_hits
        old = _mk_mem(metadata={
            "superseded_by": "new-id",
            "superseded_at": "2026-08-10T00:00:00+00:00",
        })
        hits = _rb_ctx_to_hits(RightBrainContext(situation_patterns=[old]))
        self.assertEqual(len(hits), 1)                       # 保留，不是删除
        self.assertIn("旧况", hits[0].content)
        self.assertIn("2026-08-10", hits[0].content)          # 变化时间可见
        self.assertLess(hits[0].priority, 0.5)                # 降权

    def test_current_signals_localized_english(self):
        from voicemem.engine import _rb_ctx_to_hits
        m = _mk_mem(content="Had a fight with Lisa, feeling awful about it")
        ctx = RightBrainContext(
            situation_patterns=[m],
            current_signals=CurrentSignals(dissatisfaction_signal=True),
        )
        contents = [h.content for h in _rb_ctx_to_hits(ctx)]
        signal = next(c for c in contents if "Current signals" in c or "当前信号" in c)
        self.assertIn("Current signals", signal)              # 英文上下文 → 英文信号

    def test_to_prompt_block_includes_dates(self):
        ctx = RightBrainContext(situation_patterns=[_mk_mem()])
        self.assertIn("[2026-08-01]", ctx.to_prompt_block())

    def test_long_raw_content_falls_back_to_inner_os(self):
        # 批量 ingest（20行对话合并）的超长原话不该整段进 prompt
        from voicemem.engine import _rb_ctx_to_hits
        long_text = "今天发生了很多事。" * 100
        m = _mk_mem(content=long_text, metadata={"inner_os": "【疲惫】TA 这一天很累"})
        hits = _rb_ctx_to_hits(RightBrainContext(situation_patterns=[m]))
        self.assertIn("【疲惫】TA 这一天很累", hits[0].content)
        self.assertLess(len(hits[0].content), 200)   # 原话没有整段带进来


class TextEmotionFallbackTests(unittest.TestCase):
    """纯文本 ingest 的情绪兜底：右脑写入不再依赖音频情绪检测。

    背景：三个 QA benchmark 的 ingest 都是纯文本且 emotion=""，而情绪检测
    只挂在音频分支——右脑写入路径 gate 在 `if emotion:` 上，等于整个基准里
    右脑从未写入。文本兜底用关键词表匹配（零 LLM 成本）。
    """

    def setUp(self):
        from voicemem.engine import VoiceMem
        self.tmp = Path(tempfile.mkdtemp())
        self.vm = VoiceMem(memory_root=self.tmp, user_id="u1")
        self.fake_rb_repo = MagicMock()
        self.fake_rb_repo._store.upsert_memory.return_value = MagicMock(id="rb1")
        self.vm._cache["rb_repo"] = self.fake_rb_repo
        self.vm._cache["repo"] = MagicMock()
        self.vm._cache["extractor"] = MagicMock()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _ingest(self, text: str):
        from voicemem.utils.common.voice_input import VoiceIngestResult
        no_fact = VoiceIngestResult(
            voice_id="v1", memory_ids=[], facts_count=0,
            begin_time="12:00:00", end_time="12:00:00", slots=[], messages_count=1,
        )
        with patch("voicemem.utils.common.voice_input.ingest_voice_input", return_value=no_fact), \
             patch.object(self.vm, "_generate_inner_os", return_value=""), \
             patch.object(self.vm, "_extract_rb_traits", return_value=[]):
            return self.vm.Ingest(
                text, speaker="user", emotion="", entities=[],
                session_id="s1", observed_at=None,
            )

    def test_emotional_text_writes_heartnote_without_audio(self):
        self._ingest("I'm so anxious about the deadline next week, I can't sleep")
        self.fake_rb_repo._store.upsert_memory.assert_called_once()
        _, kwargs = self.fake_rb_repo._store.upsert_memory.call_args
        self.assertEqual(kwargs["memory_class"], "heartnote")
        self.assertEqual(kwargs["metadata"]["emotion"], "焦虑")

    def test_neutral_text_writes_no_heartnote(self):
        self._ingest("The meeting is scheduled for 3pm in room 204")
        self.fake_rb_repo._store.upsert_memory.assert_not_called()

    def test_fallback_disabled_by_flag(self):
        import os as _os
        _os.environ["VOICEMEM_TEXT_EMOTION"] = "0"
        try:
            self._ingest("I'm so anxious about the deadline next week")
            self.fake_rb_repo._store.upsert_memory.assert_not_called()
        finally:
            _os.environ.pop("VOICEMEM_TEXT_EMOTION", None)


class EmotionNormalizationTests(unittest.TestCase):
    def test_strict_returns_none_for_unknown(self):
        self.assertIsNone(normalize_emotion_strict("guilty"))
        self.assertIsNone(normalize_emotion_strict(""))

    def test_strict_matches_known(self):
        self.assertEqual(normalize_emotion_strict("anxious"), "焦虑")
        self.assertEqual(normalize_emotion_strict("开心"), "开心")
        self.assertEqual(normalize_emotion_strict("有点难过"), "悲伤")

    def test_compat_wrapper_still_defaults(self):
        self.assertEqual(normalize_emotion("guilty"), "平静")

    def test_query_plan_skips_unknown_emotion_anchor(self):
        router = AnchorRouter(cognitive_store=None)
        plan = router.build_query_plan("what happened", "u1", emotion="guilty")
        self.assertFalse(any(a.anchor_type == "emotion" for a in plan.anchors))
        plan2 = router.build_query_plan("what happened", "u1", emotion="anxious")
        emo = [a for a in plan2.anchors if a.anchor_type == "emotion"]
        self.assertEqual(len(emo), 1)
        self.assertEqual(emo[0].anchor_id, "焦虑")


class AttributionRefineOnceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.store = RightBrainStore(self.tmp / "rb.sqlite")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_refine_runs_once_then_skips(self):
        from voicemem.rightbrain.attribution_manager import AttributionManager
        mem = self.store.upsert_memory("u1", "heartnote", "今天真的非常非常难受啊")
        graph = MagicMock()
        graph.get_entity.return_value = MagicMock(name="悲伤")
        graph.get_memories_for_entity.return_value = [mem.id]
        llm = MagicMock(return_value="今天很难受")
        mgr = AttributionManager(graph, self.store, llm)

        mgr.run_short_term("u1", ["e1"])
        first_calls = llm.call_count            # summarize + refine
        self.assertEqual(self.store.get_memory(mem.id).content, "今天很难受")
        self.assertTrue(self.store.get_memory(mem.id).metadata.get("refined"))

        mgr.run_short_term("u1", ["e1"])        # 第二轮：只 summarize，不再 refine
        self.assertEqual(llm.call_count, first_calls + 1)


class CleanupSupersedeTests(unittest.TestCase):
    """_run_cleanup：矛盾 → 标记 supersede 保留；重复 → 删除。"""

    def setUp(self):
        from voicemem.engine import VoiceMem
        self.tmp = Path(tempfile.mkdtemp())
        self.vm = VoiceMem(memory_root=self.tmp, user_id="u1")
        self.rb_store = self.vm._get_rb_repo()._store

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _fake_openai(self, payload: dict):
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content=json.dumps(payload)))]
        client = MagicMock()
        client.chat.completions.create.return_value = resp
        return MagicMock(return_value=client)

    def test_contradiction_marked_not_deleted(self):
        mems = [
            self.rb_store.upsert_memory("u1", "heartnote", f"记录 {i}")
            for i in range(10)
        ]
        old = self.rb_store.upsert_memory("u1", "heartnote", "以前很喜欢喝咖啡")
        new = self.rb_store.upsert_memory("u1", "heartnote", "现在戒咖啡了，改喝茶")
        payload = {
            "delete_ids": [mems[0].id[:8], mems[1].id[:8]],   # 假设前两条是重复
            "supersede": [{"old_id": old.id[:8], "new_id": new.id[:8]}],
        }
        with patch("openai.OpenAI", self._fake_openai(payload)):
            self.vm._run_cleanup()

        self.assertIsNone(self.rb_store.get_memory(mems[0].id))     # 重复被删
        kept_old = self.rb_store.get_memory(old.id)
        self.assertIsNotNone(kept_old)                              # 矛盾旧况保留
        self.assertEqual(kept_old.metadata.get("superseded_by"), new.id)
        self.assertTrue(kept_old.metadata.get("superseded_at"))
        new_mem = self.rb_store.get_memory(new.id)
        self.assertNotIn("superseded_by", new_mem.metadata)          # 新况不受影响


if __name__ == "__main__":
    unittest.main()
