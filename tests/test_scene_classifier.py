import unittest

from voicemem.utils.audio.environment.scene_classifier import SceneTag, classify_scene


class SceneClassifierTests(unittest.TestCase):
    def test_rejects_single_weak_mapped_label(self):
        result = classify_scene([("Washing machine", 0.16)])
        self.assertEqual(result.tag, SceneTag.UNKNOWN)

    def test_accepts_strong_unambiguous_evidence(self):
        result = classify_scene([("Washing machine", 0.72), ("Mechanical fan", 0.22)])
        self.assertEqual(result.tag, SceneTag.HOME)
        self.assertGreater(result.confidence, 0.5)

    def test_rejects_conflicting_scenes(self):
        result = classify_scene([("Bus", 0.51), ("Computer keyboard", 0.47)])
        self.assertEqual(result.tag, SceneTag.UNKNOWN)


if __name__ == "__main__":
    unittest.main()
