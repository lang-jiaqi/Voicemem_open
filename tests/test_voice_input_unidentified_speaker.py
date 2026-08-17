"""voice_input_to_messages()：未绑定姓名的声纹不能被喂给抽取模型时看起来像"账号主人"。

真实 bug（用真实 OpenAI 调用验证过，见 core.py/voice_input.py 里的注释）：一段
"person_f7b840f9: ..." 这样的内部 id 前缀，mem0 抽取模型基本当噪声忽略掉，
一旦 existing memories 里账号主人的真名已经出现过，就会把这句话直接安到
主人名下——哪怕说话人其实是完全不同的另一个人。这里只测 message 拼装这一层
的行为改动：未绑定声纹要显式标成"未识别说话人"，且不同的未识别声纹不能被
拼成同一个标签（否则会把两个不同的陌生人也搅到一起）。
"""
import shutil
import tempfile
import unittest
from pathlib import Path

from voicemem.voice_input import (
    VoiceContent,
    VoiceInput,
    VoiceprintRegistry,
    voice_input_to_messages,
)


class VoiceInputUnidentifiedSpeakerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.registry = VoiceprintRegistry(self.tmp / "voiceprint_registry.json")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _vi(self, *pairs):
        contents = [
            VoiceContent(sub_id=str(i), time_start="", time_end="", sentence=s, voiceprint_id=vpid)
            for i, (vpid, s) in enumerate(pairs)
        ]
        return VoiceInput(id="vi1", time_stamp={"begin": "", "end": ""}, slots=[], contents=contents)

    def test_bound_speaker_uses_real_name(self):
        self.registry.bind("person_abc", name="William")
        vi = self._vi(("person_abc", "Hello there."))
        msgs = voice_input_to_messages(vi, self.registry)
        self.assertEqual(msgs, [{"role": "user", "content": "William: Hello there."}])

    def test_unbound_speaker_is_labeled_unidentified_not_raw_id(self):
        vi = self._vi(("person_f7b840f9", "I have decided that my preferred charity is the local animal shelter."))
        msgs = voice_input_to_messages(vi, self.registry)
        self.assertEqual(len(msgs), 1)
        content = msgs[0]["content"]
        self.assertNotRegex(
            content, r"^person_f7b840f9:",
            "未绑定声纹不该把内部 id 原样当人名前缀喂给抽取模型",
        )
        self.assertIn("Unidentified speaker", content)
        self.assertIn("person_f7b840f9", content, "id 还是要保留，避免多个陌生人被混成一个标签")

    def test_two_different_unbound_speakers_get_distinct_labels(self):
        vi = self._vi(
            ("person_aaa", "First stranger talking."),
            ("person_bbb", "Second, different stranger talking."),
        )
        msgs = voice_input_to_messages(vi, self.registry)
        self.assertEqual(len(msgs), 2)
        self.assertNotEqual(msgs[0]["content"], msgs[1]["content"])
        self.assertIn("person_aaa", msgs[0]["content"])
        self.assertIn("person_bbb", msgs[1]["content"])


if __name__ == "__main__":
    unittest.main()
