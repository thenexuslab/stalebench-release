"""Sentence-embedding wrapper with a disk cache.

Embeds each text list once per (tag, model) and caches to
``results/emb_cache/<tag>__<model>__<hash>.npz``, so the full experiment grid
re-uses a single embedding pass per condition. Embeddings are L2-normalized
float32, so dot product == cosine.
"""

from __future__ import annotations

import hashlib
import os
from typing import List

import numpy as np

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "emb_cache")

MINILM = "sentence-transformers/all-MiniLM-L6-v2"
MPNET = "sentence-transformers/all-mpnet-base-v2"


class Embedder:
    def __init__(self, model_name: str = MINILM, cache_dir: str = _CACHE_DIR):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: List[str], tag: str) -> np.ndarray:
        digest = hashlib.md5("\x1f".join(texts).encode()).hexdigest()[:12]
        slug = self.model_name.split("/")[-1]
        os.makedirs(self.cache_dir, exist_ok=True)
        path = os.path.join(self.cache_dir, f"{tag}__{slug}__{digest}.npz")
        if os.path.exists(path):
            return np.load(path)["emb"]
        emb = self._load().encode(texts, batch_size=128, show_progress_bar=False,
                                  convert_to_numpy=True, normalize_embeddings=True)
        emb = emb.astype(np.float32)
        np.savez_compressed(path, emb=emb)
        return emb
