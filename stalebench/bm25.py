"""Self-contained Okapi BM25 (no external dependency).

Small corpora only (StaleBench stores are ~10^3 items); the index is rebuilt
per query for simplicity — O(n * |q|) per call, negligible here.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import List, Optional, Sequence

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN.findall(text.lower())


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def fit(self, corpus: Sequence[Optional[str]]) -> "BM25":
        self._docs = [tokenize(d or "") for d in corpus]
        self._tf = [Counter(d) for d in self._docs]
        self._len = np.array([len(d) for d in self._docs], dtype=float)
        self._avglen = float(self._len.mean()) if len(self._docs) else 0.0
        df: Counter = Counter()
        for d in self._docs:
            df.update(set(d))
        n = len(self._docs)
        self._idf = {w: math.log((n - c + 0.5) / (c + 0.5) + 1.0)
                     for w, c in df.items()}
        return self

    def score(self, query: str) -> np.ndarray:
        q = tokenize(query)
        out = np.zeros(len(self._docs), dtype=float)
        for w in q:
            idf = self._idf.get(w)
            if idf is None:
                continue
            for j, tf in enumerate(self._tf):
                f = tf.get(w, 0)
                if f:
                    denom = f + self.k1 * (1 - self.b + self.b * self._len[j] / self._avglen)
                    out[j] += idf * f * (self.k1 + 1) / denom
        return out
