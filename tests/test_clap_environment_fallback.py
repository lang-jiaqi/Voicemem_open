"""_finish_clap_environment()：CLAP 的候选词表是写死的固定集合（见
environment_detector_clap.py::PROMPTS），benchmark 实测过有真实环境音类别
不在这个表里（car_horn/helicopter/cat 等），这种情况下 CLAP 无论音质多好
都不可能给出信心过阈值的结果，detect_full() 会返回空 pairs。以前遇到这种
情况直接放弃、什么都不写——而 core.py 切到 CLAP 模式时已经把 AST 那条
覆盖面更广的即时 hint 从记忆里清空了，两边都不写等于这条环境记忆彻底丢失，
比换 CLAP 之前还差。这里测的是新加的兜底行为：CLAP 没有信心结果时，退回
用调用方传入的 AST hint 写记忆，而不是静默丢弃。
"""
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from voicemem.engine import VoiceMem


class ClapEnvironmentFallbackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.vm = VoiceMem(memory_root=self.tmp, user_id="u1")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_falls_back_to_ast_hint_when_clap_has_no_confident_match(self):
        fake_detector = MagicMock()
        fake_detector.detect_full.return_value = {"pairs": [], "music": None, "abnormal": [], "embedding": None}
        self.vm._cache["clap_env_detector"] = fake_detector

        with patch.object(self.vm, "IngestEnv") as mock_ingest_env:
            self.vm._finish_clap_environment(
                "/fake/audio.wav", "some text", "sess1",
                environment_hint="background sounds: cat(0.42)",
            )

        mock_ingest_env.assert_called_once()
        _, kwargs = mock_ingest_env.call_args
        self.assertEqual(kwargs["environment_override"], "background sounds: cat(0.42)")

    def test_no_write_when_clap_empty_and_no_ast_hint(self):
        fake_detector = MagicMock()
        fake_detector.detect_full.return_value = {"pairs": [], "music": None, "abnormal": [], "embedding": None}
        self.vm._cache["clap_env_detector"] = fake_detector

        with patch.object(self.vm, "IngestEnv") as mock_ingest_env:
            self.vm._finish_clap_environment("/fake/audio.wav", "some text", "sess1", environment_hint="")

        mock_ingest_env.assert_not_called()

    def test_clap_result_used_directly_when_confident(self):
        fake_detector = MagicMock()
        fake_detector.detect_full.return_value = {
            "pairs": [("siren", 0.91)], "music": None, "abnormal": [], "embedding": None,
        }
        self.vm._cache["clap_env_detector"] = fake_detector

        with patch.object(self.vm, "IngestEnv") as mock_ingest_env:
            self.vm._finish_clap_environment(
                "/fake/audio.wav", "some text", "sess1",
                environment_hint="background sounds: wind(0.30)",
            )

        mock_ingest_env.assert_called_once()
        _, kwargs = mock_ingest_env.call_args
        self.assertIn("siren(0.91)", kwargs["environment_override"])


if __name__ == "__main__":
    unittest.main()
