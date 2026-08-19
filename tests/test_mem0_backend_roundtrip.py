"""Real integration test for the mem0/Qdrant migration (voicemem/leftbrain/
mem0_backend_store.py::Mem0BackendStore), which this session replaced the
old hand-rolled SQLite+numpy LocalMemoryStore with. This has zero test
coverage otherwise -- a mock of ``mem0.Memory`` would only prove the
plumbing/argument-passing is right, not the actual integration (which is
exactly where real regressions were found and fixed this session: the local/
embedded Qdrant mode's one-client-per-process constraint, requiring the
module-level ``_MEM0_CLIENT_CACHE`` keyed by resolved qdrant path so
multiple ``Mem0BackendStore`` instances pointed at the same ``memory_root``
share one underlying client instead of colliding on the storage folder
lock -- see that module's docstring for the real repro).

``Mem0BackendStore`` always uses mem0's own real OpenAI-backed embedder
internally (the ``embedder`` constructor arg is kept only for interface
parity, per that class's own comment) -- there is no way to run this
against a fully local/mocked embedder, so this test is a real network call
and is skipped when no ``OPENAI_API_KEY`` is configured, same convention
real API-backed integration tests use elsewhere in this ecosystem.
"""
import os
import shutil
import tempfile
import unittest
from pathlib import Path

# mem0's telemetry writes to a single global ~/.mem0/migrations_qdrant path
# regardless of memory_root, which collides with any other voicemem process
# on this machine holding that same lock -- unrelated to what this test
# actually checks, so disable it rather than have the test flake based on
# what else happens to be running. Must be set before Mem0BackendStore (and
# therefore mem0.Memory) is constructed.
os.environ.setdefault("MEM0_TELEMETRY", "False")

from voicemem.leftbrain.local_memory_store import OpenAILocalEmbedder, OpenAILocalEmbedderConfig
from voicemem.leftbrain.mem0_backend_store import Mem0BackendStore


@unittest.skipUnless(
    os.environ.get("OPENAI_API_KEY"),
    "requires a live OPENAI_API_KEY -- mem0's embedder is always real OpenAI, no local/mock path",
)
class Mem0BackendRoundtripTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        embedder = OpenAILocalEmbedder(OpenAILocalEmbedderConfig())
        self.store = Mem0BackendStore(embedder, memory_root=self.tmp)
        self.user_id = "test_user_mem0_roundtrip"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_add_then_search_finds_the_fact(self):
        ids = self.store.add_records_with_ids(
            self.user_id,
            [("ignored_local_id", "My favorite noodle dish is tonkotsu ramen.", "user", {})],
        )
        self.assertEqual(len(ids), 1)
        self.assertTrue(ids[0])  # mem0 generated a real id, not the ignored local one

        hits = self.store.search("what noodles do I like", user_id=self.user_id, top_k=5)
        self.assertTrue(hits, "expected the just-added fact to be retrievable")
        self.assertTrue(any("ramen" in h.text.lower() for h in hits))

    def test_two_stores_same_memory_root_share_one_qdrant_client(self):
        """The real regression this guards: local/embedded Qdrant only allows
        one client per storage folder ('Storage folder ... is already
        accessed by another instance of Qdrant client'). Two Mem0BackendStore
        instances pointed at the same memory_root must not both try to open
        their own Qdrant client."""
        embedder = OpenAILocalEmbedder(OpenAILocalEmbedderConfig())
        second_store = Mem0BackendStore(embedder, memory_root=self.tmp)
        # No exception on construction -- and they must be the exact same
        # cached client (module-level _MEM0_CLIENT_CACHE keyed by qdrant
        # path), not two independent ones that happen not to have collided
        # yet.
        self.assertIs(second_store._mem0, self.store._mem0)


if __name__ == "__main__":
    unittest.main()
