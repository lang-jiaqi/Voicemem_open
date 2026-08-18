import unittest

from voicemem_core.speaker_identity import parse_self_identification


class SpeakerIdentityTests(unittest.TestCase):
    def test_accepts_explicit_names(self):
        self.assertEqual(parse_self_identification("My name is Deborah."), "Deborah")
        self.assertEqual(parse_self_identification("I'm Mark."), "Mark")
        self.assertEqual(parse_self_identification("Hi, I am Thomas"), "Thomas")
        self.assertEqual(parse_self_identification("我叫小明。"), "小明")

    def test_rejects_states_and_actions(self):
        for text in (
            "I'm worried.",
            "I'm heading out.",
            "I am worried.",
            "I am heading out.",
            "I'm actually heading out the door.",
            "我是担心。",
            "我是出门。",
        ):
            with self.subTest(text=text):
                self.assertIsNone(parse_self_identification(text))


if __name__ == "__main__":
    unittest.main()
