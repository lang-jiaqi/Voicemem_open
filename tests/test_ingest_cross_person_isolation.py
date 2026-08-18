"""ingest_voice_input()：existing_memories 候选池按 user_id（整个家庭）搜出来，
不按说话人过滤——真实复现过的 bug：Nancy 和 Jennifer 都说了近似模板的话
（"我今年 donation drive 选哪个慈善机构"），Jennifer 那轮抽取时 Nancy 的旧
记忆被搜进候选，ConflictResolver 判成 UPDATE，产出"Nancy...是 Jennifer
的答案"这种缝合内容。system prompt 里虽然写了"不同人名不能 UPDATE"，但那只是
软提示，模型不保证遵守。这里测的是代码层面的硬过滤：候选文本点了别人的名字、
又没提当前说话人自己的名字，必须在传给 ConflictResolver 之前就被剔除。
"""
import shutil
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from voicemem_core.leftbrain.extract_facts_openai import ExtractedAdditiveMemory
from voicemem_core.voice_input import (
    VoiceContent,
    VoiceInput,
    VoiceprintRegistry,
    ingest_voice_input,
)


@dataclass
class _Hit:
    memory_id: str
    text: str


class _FakeRepo:
    def __init__(self, hits):
        self._hits = hits
        self.appended: list = []

    def search(self, query, *, user_id, top_k=5, threshold=None):
        return self._hits[:top_k]

    def append_extracted(self, memories, *, user_id, extra_metadata=None):
        self.appended.extend(memories)
        return [f"mem_{i}" for i in range(len(memories))]


class _FakeExtractor:
    def __init__(self, facts):
        self._facts = facts

    def extract(self, **kwargs):
        return self._facts


class IngestCrossPersonIsolationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.registry = VoiceprintRegistry(self.tmp / "voiceprint_registry.json")
        self.registry.bind("person_nancy", name="Nancy")
        self.registry.bind("person_jennifer", name="Jennifer")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _jennifer_vi(self):
        return VoiceInput(
            id="vi1", time_stamp={"begin": "", "end": ""}, slots=[],
            contents=[VoiceContent(
                sub_id="0", time_start="", time_end="",
                sentence="My preferred local charity is the downtown botanical society.",
                voiceprint_id="person_jennifer",
            )],
        )

    def test_other_persons_memory_never_reaches_resolver(self):
        nancy_hit = _Hit(
            memory_id="mem_nancy_charity",
            text=(
                "Nancy has decided that her preferred local charity for the "
                "annual donation drive is the local animal shelter."
            ),
        )
        repo = _FakeRepo([nancy_hit])
        extractor = _FakeExtractor([
            ExtractedAdditiveMemory(
                local_id="0",
                text=(
                    "Jennifer has decided that her preferred local charity for "
                    "the annual donation drive is the downtown botanical society."
                ),
                attributed_to="user",
            )
        ])

        with patch(
            "voicemem_core.leftbrain.extract_facts_openai.ConflictResolver.resolve",
            return_value=[],
        ) as mock_resolve:
            result = ingest_voice_input(
                self._jennifer_vi(), "household_1",
                registry=self.registry, repo=repo, extractor=extractor,
            )

        # Nancy 的旧记忆被过滤掉后 existing 变空，resolve() 压根不会被调用——
        # 没有候选可看，也就没有把 Jennifer 的事实错判成 UPDATE 目标的机会。
        mock_resolve.assert_not_called()
        self.assertEqual(len(repo.appended), 1)
        self.assertIn("Jennifer", repo.appended[0].text)
        self.assertIn("downtown botanical society", repo.appended[0].text)
        self.assertEqual(result.facts_count, 1)

    def test_own_memory_still_reaches_resolver(self):
        """过滤只排除"点了别人名字、没点自己名字"的候选；提到当前说话人自己的
        候选（哪怕也提到了别人，比如关系类记忆）要正常留着走 UPDATE/DELETE。"""
        jennifer_hit = _Hit(
            memory_id="mem_jennifer_old_charity",
            text=(
                "Jennifer has decided that her preferred local charity for the "
                "annual donation drive is the wrong old answer."
            ),
        )
        repo = _FakeRepo([jennifer_hit])
        extractor = _FakeExtractor([
            ExtractedAdditiveMemory(
                local_id="0",
                text=(
                    "Jennifer has decided that her preferred local charity for "
                    "the annual donation drive is the downtown botanical society."
                ),
                attributed_to="user",
            )
        ])

        with patch(
            "voicemem_core.leftbrain.extract_facts_openai.ConflictResolver.resolve",
            return_value=[],
        ) as mock_resolve:
            ingest_voice_input(
                self._jennifer_vi(), "household_1",
                registry=self.registry, repo=repo, extractor=extractor,
            )

        mock_resolve.assert_called_once()
        args, kwargs = mock_resolve.call_args
        passed_existing = args[1]
        self.assertEqual(len(passed_existing), 1)
        self.assertEqual(passed_existing[0]["id"], "mem_jennifer_old_charity")


if __name__ == "__main__":
    unittest.main()
