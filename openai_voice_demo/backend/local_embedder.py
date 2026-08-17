"""Local embedding-based memory storage/ranking -- ONLY used by this demo
(openai_voice_demo), as a fast local alternative to the OpenAI embeddings
API call voicemem's mem0/Qdrant backend otherwise makes on every Ingest()
(storing a fact) and every Search() (Rank()'s vector ranking step).

voicemem's own default (``OpenAILocalEmbedder`` in
``voicemem/leftbrain/local_memory_store.py``) remains the official,
standard implementation -- this is a demo-only opt-in, wired in by passing
``embedder=LocalMemoryEmbedder(...)`` to ``VoiceMem()``'s constructor (see
``memory_bridge.py``). Real, disclosed cost: switching embedders changes
vector *dimensions* (this model is 384-dim; OpenAI's default
``text-embedding-3-small`` is 1536-dim), so an existing memory store built
with OpenAI embeddings is NOT compatible with this -- the Qdrant collection
must be rebuilt from empty when switching. Not something this module works
around; the caller needs to reset the memory store when flipping this
setting for an existing installation.

Same model as local_classifier.py (``intfloat/multilingual-e5-small``,
93% top-2 slot accuracy measured this session, ~4.7ms/query on this
machine's GPU) -- pass the same already-loaded ``SentenceTransformer``
instance in via ``model=`` to avoid loading it twice.

E5's "query: "/"passage: " prefix convention matters here more than it did
for slot classification: this embedder's output IS the actual memory
content vector space (what gets compared for real fact retrieval), not just
a 7-way category pick. voicemem/leftbrain/mem0_backend_store.py's
``_Mem0EmbedderAdapter`` is what makes the query/passage distinction
actually take effect -- see that file's docstring: mem0's ``Memory.add()``/
``.search()`` call ``embedding_model.embed(text, "add"|"search")`` with the
real action, and this class exposes ``embed_query_text()`` specifically so
that adapter can route "add" (storing a fact -- passage) and "search"
(a query) to different prefixes instead of treating them identically.
"""
from __future__ import annotations

import numpy as np


class LocalMemoryEmbedder:
    """Conforms to voicemem's ``TextEmbedder`` protocol (``model_name``,
    ``dimensions``, ``embed_texts()``) plus the optional ``embed_query_text()``
    extension ``_Mem0EmbedderAdapter`` looks for.
    """

    def __init__(self, model_name: str = "intfloat/multilingual-e5-small", model=None) -> None:
        if model is not None:
            self._model = model
        else:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
        self._model_name = model_name
        self._dims = self._model.get_embedding_dimension()

    @property
    def model_name(self) -> str:
        return f"{self._model_name} (local)"

    @property
    def dimensions(self) -> int:
        return self._dims

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generic path -- used for storing facts (mem0's "add" action) and
        anywhere else in voicemem that just wants text embedded (e.g. slot
        description embeddings), so it prefixes as "passage: " (an
        indexed/stored piece of text), matching E5's convention.
        """
        if not texts:
            return []
        prefixed = [f"passage: {t}" for t in texts]
        embs = self._model.encode(prefixed, normalize_embeddings=True)
        return np.asarray(embs).tolist()

    def embed_query_text(self, text: str) -> list[float]:
        """Search-query path (mem0's "search" action) -- "query: " prefix."""
        emb = self._model.encode([f"query: {text}"], normalize_embeddings=True)[0]
        return np.asarray(emb).tolist()
