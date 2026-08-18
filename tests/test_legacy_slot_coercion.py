"""Real bug this session: four real entities in a demo's own store had a
legacy slot value (``slot='people'``) from before the SlotV2 taxonomy
unification (old taxonomy: people/projects/knowledge/tasks/places/routines/
assets). ``SlotV2(row["slot"])`` raised ``ValueError: 'people' is not a
valid SlotV2`` on every single row-hydration call, and that crash was
silently swallowed by core.py's ``_run_rb()``'s blanket
``except Exception: return []`` -- meaning the right brain returned zero
hits on every search, for an entire session, misleadingly presenting as "no
relevant memories" rather than "the code is crashing". Fixed via
``_coerce_slot_v2()`` in cognitive_graph/store.py. This test inserts a row
with a legacy slot value directly (bypassing ``upsert_entity``, which only
accepts real ``SlotV2`` values) to simulate exactly the pre-existing data
that caused the crash, and asserts hydration succeeds and maps sensibly.
"""
import shutil
import tempfile
import unittest
import uuid
from pathlib import Path

from voicemem_core.leftbrain.cognitive_graph.slot_v2 import SlotV2
from voicemem_core.leftbrain.cognitive_graph.store import CognitiveGraphStore


class LegacySlotCoercionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.store = CognitiveGraphStore(self.tmp / "cognitive_graph.sqlite")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _insert_legacy_row(self, slot: str) -> str:
        entity_id = f"person_{uuid.uuid4().hex[:6]}"
        with self.store._conn() as c:
            c.execute(
                "INSERT INTO entities "
                "(id, user_id, entity_type, name, name_norm, slot, created_at, updated_at) "
                "VALUES (?, 'user1', 'person', 'Jiaqi Lang', 'jiaqi lang', ?, '2024-01-01', '2024-01-01')",
                (entity_id, slot),
            )
        return entity_id

    def test_legacy_people_slot_does_not_crash_find_entities(self):
        self._insert_legacy_row("people")
        # This is the exact call chain that crashed silently in production
        # (core.py's _run_rb -> AnchorRouter._build_anchors -> find_entities).
        entities = self.store.find_entities("user1")
        self.assertEqual(len(entities), 1)
        # "people" (old taxonomy) maps to "relationships" (new SlotV2), per
        # _LEGACY_SLOT_TO_SLOTV2 -- not knowledge, and not a crash.
        self.assertEqual(entities[0].slot, SlotV2.RELATIONSHIPS)

    def test_other_legacy_slots_map_as_expected(self):
        cases = {
            "projects": SlotV2.WORK,
            "tasks": SlotV2.WORK,
            "places": SlotV2.DAILY_LIFE,
            "routines": SlotV2.DAILY_LIFE,
            "assets": SlotV2.DAILY_LIFE,
        }
        for legacy, expected in cases.items():
            with self.subTest(legacy=legacy):
                eid = self._insert_legacy_row(legacy)
                found = self.store.find_entities("user1", entity_ids=[eid])
                self.assertEqual(len(found), 1)
                self.assertEqual(found[0].slot, expected)

    def test_unrecognized_legacy_value_falls_back_to_knowledge_not_crash(self):
        self._insert_legacy_row("some_totally_unknown_old_value")
        entities = self.store.find_entities("user1")
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0].slot, SlotV2.KNOWLEDGE)

    def test_current_slotv2_values_pass_through_unchanged(self):
        eid = self._insert_legacy_row("work")
        found = self.store.find_entities("user1", entity_ids=[eid])
        self.assertEqual(found[0].slot, SlotV2.WORK)


if __name__ == "__main__":
    unittest.main()
