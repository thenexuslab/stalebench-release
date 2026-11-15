"""Retrieval / retention policies for StaleBench.

Four families, so the comparison is honest:

* **Store-all similarity** --- ``CosineRAG`` (dense), ``BM25RAG`` (lexical):
  keep everything, rank by similarity. The staleness victims.
* **Time-aware store-all** --- ``TimeRerank`` (similarity top-k, answer the most
  recent candidate) and ``RecencyWeighted`` (score = sim + lam * decay**age):
  the cheap metadata fixes a reviewer will immediately propose. They repair
  top-1 freshness in easy regimes but keep every stale version in the store,
  so they cannot repair contamination@k or footprint.
* **Retention by overwrite** (``KeyedRetention``) --- one slot per fact,
  destructive.
* **Retention by invalidation** (``TemporalKG``) --- the Zep/Graphiti design:
  a superseded statement is marked invalid rather than deleted, and retrieval
  prefers valid edges. Staleness and contamination fall as with overwrite,
  false merges stay recoverable, footprint does not fall.

Both retention mechanisms take the *same* pluggable **keyer**, which is the one
component that decides whether two statements describe the same fact. The
keying ladder is therefore an axis, not a set of unrelated policies:

===================  =======================================================
``ThresholdKeyer``   cosine to nearest slot > tau (what vector stores do)
``LexicalKeyer``     longest verbatim gazetteer subject in the text
``AliasKeyer``       gazetteer + alias inventory: an idealized entity linker
``LLMKeyer``         a language model adjudicates candidate pairs
``OracleKeyer``      the true fact key --- the bound for any keyer
===================  =======================================================

Every policy sees only opaque item ids at ``observe`` time (the entity/version
truth is decoded by the metric, never by the policy) --- except
``OracleKeyer``, which is granted the true key to define the upper bound.

Storage note: embeddings live in a doubling ``(n, d)`` buffer rather than a
list, so a same-fact scan is one matrix-vector product against a view. The
full-corpus track streams ~10k statements, where re-stacking a list per event
is quadratic in wall-clock.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set

import numpy as np

import hashlib
import re

from stalebench.bm25 import BM25

_MISSING = object()


class SurfaceMatcher:
    """Finds the longest inventory surface form occurring in a text as a mention.

    Plain substring matching (``form in text``) is wrong in two ways that both
    inflate keying quality in opposite directions. It matches inside words, and
    it matches inside *other names*: on the full corpus it linked 46% of
    distractor statements into a queried fact through forms like ``Democratic``,
    ``States`` or ``John``, which made an incomplete inventory look about three
    times more costly than it is. A mention therefore must

    * align to token boundaries, and
    * not continue into another capitalized token on either side --- ``Richard``
      does not refer to *Richard Blumenthal* in ``Richard Cresswell plays for
      ...``.

    Matching is done over token n-grams rather than by scanning the inventory,
    so cost is independent of inventory size (a few hundred dict lookups per
    statement instead of a regex sweep over thousands of forms).
    """

    _TOKEN = re.compile(r"\S+")

    def __init__(self, forms: Sequence[str]):
        self._by_ngram: Dict[tuple, str] = {}
        for f in sorted(set(forms), key=lambda x: (-len(x), x)):
            key = tuple(self._norm(t) for t in f.split())
            self._by_ngram.setdefault(key, f)
        self._max_len = max((len(k) for k in self._by_ngram), default=1)

    @staticmethod
    def _norm(tok: str) -> str:
        """Strip sentence punctuation while keeping name-internal marks.

        Kept: initials (``M.``) and dotted abbreviations (``F.C.``). Removed:
        the sentence period that follows them (``F.C...`` -> ``F.C.``), the
        possessive (``Baton Rouge's`` -> ``Baton Rouge``), and quotes/brackets.
        Without this the templates that end on an abbreviation or take a
        possessive never match their own subject.
        """
        tok = tok.rstrip(",;:!?)\u201d\"'").lstrip("(\u201c\"")
        for suf in ("\u2019s", "'s"):
            if tok.endswith(suf):
                tok = tok[:-len(suf)]
        while tok.endswith(".."):
            tok = tok[:-1]
        if (tok.endswith(".") and tok.count(".") == 1
                and not (len(tok) == 2 and tok[0].isalpha())):
            tok = tok[:-1]
        return tok

    def match(self, text: str) -> Optional[str]:
        raw = [m.group(0) for m in self._TOKEN.finditer(text or "")]
        toks = [self._norm(r) for r in raw]
        if not toks:
            return None
        for length in range(min(self._max_len, len(toks)), 0, -1):
            for i in range(len(toks) - length + 1):
                form = self._by_ngram.get(tuple(toks[i:i + length]))
                if form is None:
                    continue
                if i + length < len(toks) and toks[i + length][:1].isupper():
                    continue                  # mention continues to the right
                if i:
                    prev_raw = raw[i - 1]
                    # a capitalized word immediately before is part of the same
                    # mention -- unless it closed a clause ("Politically, X ..."),
                    # where the capital is sentence-initial rather than nominal
                    if (prev_raw[:1].isupper() and prev_raw[0].isalpha()
                            and prev_raw[-1] not in ",;:."):
                        continue
                return form
        return None


class Policy:
    """Base store: append-only with optional recency eviction, cosine ranking."""

    name = "base"

    def __init__(self, budget: Optional[int] = None):
        self.budget = budget
        self.reset()

    # -- storage ----------------------------------------------------------- #

    def reset(self) -> None:
        self._ids: List[str] = []
        self._t: List[float] = []
        self._text: List[Optional[str]] = []
        self._keys: List[Optional[str]] = []
        self._M: Optional[np.ndarray] = None       # (cap, d) buffer
        self._n = 0
        self._rev = 0                              # bumped on every mutation

    @property
    def _emb(self) -> np.ndarray:
        """Live embedding rows, insertion-ordered."""
        if self._M is None:
            return np.zeros((0, 0), np.float32)
        return self._M[:self._n]

    def _push(self, emb: np.ndarray, item_id: str, t: float,
              text: Optional[str], key: Optional[str] = None) -> None:
        if self._M is None:
            self._M = np.zeros((16, emb.shape[0]), np.float32)
        elif self._n == self._M.shape[0]:
            self._M = np.concatenate([self._M, np.zeros_like(self._M)], axis=0)
        self._M[self._n] = emb
        self._n += 1
        self._ids.append(item_id); self._t.append(t)
        self._text.append(text); self._keys.append(key)
        self._rev += 1

    def _overwrite(self, j: int, emb: np.ndarray, item_id: str, t: float,
                   text: Optional[str]) -> None:
        self._M[j] = emb
        self._ids[j] = item_id; self._t[j] = t; self._text[j] = text
        self._rev += 1

    def _drop(self, j: int) -> None:
        self._M[j:self._n - 1] = self._M[j + 1:self._n]
        self._n -= 1
        for lst in (self._ids, self._t, self._text, self._keys):
            del lst[j]
        self._rev += 1

    def observe(self, emb: np.ndarray, item_id: str, t: float,
                text: Optional[str] = None) -> None:
        self._push(emb, item_id, t, text)
        self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        if self.budget is not None and self._n > self.budget:
            self._drop(int(np.argmin(self._t)))     # drop oldest

    @property
    def slots(self) -> int:
        """Stored items (memory footprint)."""
        return self._n

    @property
    def active_slots(self) -> int:
        """Items a retriever will surface (== slots unless invalidating)."""
        return self._n

    # -- ranking ----------------------------------------------------------- #

    def _scores(self, query: np.ndarray, query_text: Optional[str]) -> np.ndarray:
        return self._emb @ query

    def rank(self, query: np.ndarray, query_text: Optional[str] = None) -> List[str]:
        """Full ranking of stored item ids, best first."""
        if not self._n:
            return []
        order = np.argsort(-self._scores(query, query_text), kind="stable")
        return [self._ids[i] for i in order]

    def answer(self, query: np.ndarray,
               query_text: Optional[str] = None) -> Optional[str]:
        r = self.rank(query, query_text)
        return r[0] if r else None


class CosineRAG(Policy):
    def __init__(self):
        super().__init__(budget=None)
        self.name = "cosine-rag"


class Recency(Policy):
    def __init__(self, budget: int):
        super().__init__(budget=budget)
        self.name = f"recency(B={budget})"


class TimeRerank(Policy):
    """Similarity top-k, then answer the most recent candidate.

    The candidate *set* is unchanged from cosine ranking, so contamination@j
    for j <= k is identical to CosineRAG's --- reranking cannot remove stale
    versions from the context, only demote them.
    """

    def __init__(self, k: int = 5, budget: Optional[int] = None):
        self.k = k
        super().__init__(budget=budget)
        self.name = f"time-rerank(k={k}" + (f",B={budget})" if budget else ")")

    def rank(self, query: np.ndarray, query_text: Optional[str] = None) -> List[str]:
        if not self._n:
            return []
        order = list(np.argsort(-self._scores(query, query_text), kind="stable"))
        top = sorted(order[:self.k], key=lambda j: -self._t[j])
        return [self._ids[j] for j in top + order[self.k:]]


class RecencyWeighted(Policy):
    """Generative-Agents-style additive freshness prior: sim + lam*decay**age."""

    def __init__(self, lam: float = 0.5, decay: float = 0.99,
                 budget: Optional[int] = None):
        self.lam = lam
        self.decay = decay
        super().__init__(budget=budget)
        self.name = f"recency-weighted(lam={lam},decay={decay})"

    def _scores(self, query: np.ndarray, query_text: Optional[str]) -> np.ndarray:
        sims = self._emb @ query
        t = np.asarray(self._t, dtype=float)
        return sims + self.lam * (self.decay ** (t.max() - t))


class BM25RAG(Policy):
    """Lexical store-all baseline (Okapi BM25 over stored texts).

    ``rerank_k`` turns it into the lexical analogue of ``TimeRerank``.
    """

    def __init__(self, rerank_k: Optional[int] = None):
        self.rerank_k = rerank_k
        super().__init__(budget=None)
        self.name = "bm25" + (f"+time-rerank(k={rerank_k})" if rerank_k else "")

    def reset(self) -> None:
        super().reset()
        self._index = None
        self._index_rev = -1

    def _scores(self, query: np.ndarray, query_text: Optional[str]) -> np.ndarray:
        assert query_text is not None, "BM25RAG needs query_text"
        if self._index is None or self._index_rev != self._rev:
            # the store is static during a query batch; refitting per query is
            # O(corpus) and dominates the full-corpus track (~7k statements)
            self._index = BM25().fit(list(self._text))
            self._index_rev = self._rev
        return self._index.score(query_text)

    def rank(self, query: np.ndarray, query_text: Optional[str] = None) -> List[str]:
        if not self._n:
            return []
        order = list(np.argsort(-self._scores(query, query_text), kind="stable"))
        if self.rerank_k:
            top = sorted(order[:self.rerank_k], key=lambda j: -self._t[j])
            order = top + order[self.rerank_k:]
        return [self._ids[j] for j in order]


# --------------------------------------------------------------------------- #
# Keyers: the same-fact decision, isolated
# --------------------------------------------------------------------------- #


class Keyer:
    """Decides the fact identity of an incoming statement.

    ``key_of`` returns a key string shared by statements about the same fact,
    or ``None`` for "cannot key this" (the store then appends an unmergeable
    slot). ``store`` exposes the current contents (``_emb``, ``_ids``,
    ``_text``, ``_t``, ``_keys``, ``_n``) for keyers that need context.
    """

    name = "keyer"

    def reset(self) -> None:
        """Clear per-run state."""

    def key_of(self, store: "Policy", emb: np.ndarray, item_id: str,
               text: Optional[str]) -> Optional[str]:
        raise NotImplementedError


class ThresholdKeyer(Keyer):
    """Cosine-threshold keying: the mechanism vector-store memories actually use.

    The nearest stored slot within cosine ``tau`` is judged the same fact.
    Keys are synthetic (slot identities), so this never sees fact truth.
    """

    def __init__(self, tau: float = 0.9):
        self.tau = tau
        self.name = f"tau={tau}"
        self.reset()

    def reset(self) -> None:
        self._next = 0

    def key_of(self, store, emb, item_id, text):
        if store._n:
            sims = store._emb @ emb
            j = int(np.argmax(sims))
            if sims[j] > self.tau and store._keys[j] is not None:
                return store._keys[j]
        self._next += 1
        return f"k{self._next}"


class LexicalKeyer(Keyer):
    """Gazetteer/NER-idealized keyer: longest verbatim subject in the text.

    Idealizes an off-the-shelf entity linker with a clean public subject
    inventory and no aliasing. It holds exactly as long as statements carry the
    subject's surface form verbatim --- the assumption the anchor-free track
    removes.
    """

    name = "lexical"

    def __init__(self, gazetteer: Sequence[str]):
        # longest first, ties broken lexicographically: `set` iteration order
        # depends on the interpreter's per-process string hash seed, so keying
        # must not inherit it (v0.2 did, and drifted by one query across runs).
        self._matcher = SurfaceMatcher(gazetteer)

    def key_of(self, store, emb, item_id, text):
        return self._matcher.match(text or "")


class FuzzyKeyer(Keyer):
    """Entity resolution with a subject inventory but **no alias table**.

    The gazetteer and alias rungs bracket a range no deployed system occupies:
    exact matching collapses when the anchor goes, while the alias rung is
    handed the very inventory that generated the aliases and so has no
    out-of-vocabulary mass. This rung is what a system actually has --- a list
    of canonical subject names and nothing else --- and must resolve short forms
    itself.

    It detects candidate mentions as maximal runs of capitalized tokens, then
    links each to the inventory entry with the highest character-trigram
    Jaccard similarity, accepting above ``tau``. That is deliberately ordinary
    machinery: no learning, no embeddings, no curated aliases.
    """

    def __init__(self, gazetteer: Sequence[str], tau: float = 0.34):
        self.tau = tau
        self.name = f"fuzzy(tau={tau})"
        self._canon = sorted(set(gazetteer))
        self._grams = [self._tri(c) for c in self._canon]
        # inverted index over trigrams so scoring touches few candidates
        self._index: Dict[str, List[int]] = {}
        for i, gs in enumerate(self._grams):
            for g in gs:
                self._index.setdefault(g, []).append(i)
        # The link itself does not depend on tau, so the best (match, score)
        # per statement is cached across instances sharing an inventory:
        # threshold tuning then costs one pass over the corpus, not one per
        # grid point.
        key = hashlib.md5("\x1f".join(self._canon).encode()).hexdigest()
        self._memo = FuzzyKeyer._CACHE.setdefault(key, {})
        self.reset()

    @staticmethod
    def _tri(text: str) -> Set[str]:
        t = f"  {text.lower()} "
        return {t[i:i + 3] for i in range(len(t) - 2)}

    _MENTION = re.compile(r"(?:[A-Z\u00C0-\u024F][\w.\u00C0-\u024F'-]*"
                          r"(?:\s+(?:of|the|de|van|von))?\s*)+")
    _CACHE: Dict[tuple, Dict[str, tuple]] = {}

    def reset(self) -> None:
        self.n_unlinked = 0
        self.n_linked = 0

    def key_of(self, store, emb, item_id, text):
        # the decision depends only on the statement text, so it is memoized:
        # threshold tuning replays the same corpus once per grid point
        hit = self._memo.get(text, _MISSING)
        if hit is _MISSING:
            hit = self._link(text)
            self._memo[text] = hit
        best, sim = hit
        if best is not None and sim >= self.tau:
            self.n_linked += 1
            return best
        self.n_unlinked += 1
        return None

    def _link(self, text) -> tuple:
        """Best (inventory entry, similarity) for any mention in the text."""
        best, best_sim = None, 0.0
        for m in self._MENTION.finditer(text or ""):
            span = m.group(0).strip(" .,;:")
            if len(span) < 3:
                continue
            q = self._tri(span)
            cand: Dict[int, int] = {}
            # iterate the trigram set in sorted order and break similarity ties
            # lexicographically: iterating a set of strings inherits the
            # interpreter's per-process hash seed, which is exactly the defect
            # recorded in the appendix for the gazetteer keyer
            for g in sorted(q):
                for i in self._index.get(g, ()):
                    cand[i] = cand.get(i, 0) + 1
            for i in sorted(cand):
                shared = cand[i]
                sim = shared / (len(q) + len(self._grams[i]) - shared)
                if sim > best_sim or (sim == best_sim and best is not None
                                      and self._canon[i] < best):
                    best, best_sim = self._canon[i], sim
        return best, best_sim


class AliasKeyer(Keyer):
    """Idealized *entity linker*: resolves alias surface forms to canonical keys.

    Holds the same subject inventory as :class:`LexicalKeyer` plus every alias
    derivable from it (:mod:`stalebench.alias`). The longest matching surface
    form is resolved to its canonical subject. An alias owned by more than one
    subject is genuinely ambiguous from the statement alone; ``ambiguous``
    selects the linker's policy:

    * ``"skip"`` --- refuse to link, append a new slot (conservative);
    * ``"recent"`` --- link to the most recently written candidate (the
      discourse-recency heuristic real linkers use).

    This rung prices entity resolution itself: the alias table is handed over
    for free, so any residual gap to the oracle is caused by ambiguity and by
    statements that carry no resolvable form at all.
    """

    def __init__(self, gazetteer: Sequence[str],
                 alias_map: Dict[str, Set[str]], ambiguous: str = "skip"):
        assert ambiguous in ("skip", "recent")
        self.ambiguous = ambiguous
        self.name = f"alias/{ambiguous}"
        canon = set(gazetteer)
        forms: Dict[str, Optional[str]] = {c: c for c in canon}
        self._ambig_owners: Dict[str, Set[str]] = {}
        for a, owners in alias_map.items():
            own = owners & canon
            if not own or a in canon:
                continue
            if len(own) == 1:
                forms[a] = next(iter(own))
            else:
                forms[a] = None
                self._ambig_owners[a] = own
        self._canon_of = forms
        self._matcher = SurfaceMatcher(forms)
        self.reset()

    def reset(self) -> None:
        self.n_ambiguous_hits = 0
        self.n_unlinked = 0

    def key_of(self, store, emb, item_id, text):
        f = self._matcher.match(text or "")
        if f is None:
            self.n_unlinked += 1
            return None
        canon = self._canon_of[f]
        if canon is not None:
            return canon
        self.n_ambiguous_hits += 1
        if self.ambiguous == "skip":
            self.n_unlinked += 1
            return None
        owners = self._ambig_owners[f]
        best, best_t = None, -np.inf
        for j in range(store._n):
            if store._keys[j] in owners and store._t[j] > best_t:
                best, best_t = store._keys[j], store._t[j]
        if best is None:
            self.n_unlinked += 1
        return best


class LLMKeyer(Keyer):
    """Same-fact detection delegated to a language model.

    The top-``n_cand`` stored items by cosine are offered to the model, which
    returns the 1-based index of the one describing the same fact, or 0 for
    "none". Keys are synthetic slot identities, so the model never sees a fact
    key --- only statement text.

    This rung answers the question the ladder raises: *is anchor-free same-fact
    detection simply solved by asking an LLM?*
    """

    def __init__(self, client, n_cand: int = 4):
        self.client = client
        self.n_cand = n_cand
        self.name = f"llm/{getattr(client, 'name', 'llm')}"
        self.reset()

    def reset(self) -> None:
        self._next = 0
        self.n_calls = 0
        self.n_linked = 0

    def key_of(self, store, emb, item_id, text):
        if store._n:
            sims = store._emb @ emb
            cand = [int(j) for j in np.argsort(-sims, kind="stable")[:self.n_cand]]
            choice = self.client.same_fact(text or "",
                                           [store._text[j] or "" for j in cand])
            self.n_calls += 1
            if choice is not None and 0 < choice <= len(cand):
                k = store._keys[cand[choice - 1]]
                if k is not None:
                    self.n_linked += 1
                    return k
        self._next += 1
        return f"k{self._next}"


class OracleKeyer(Keyer):
    """The true fact key (item-id prefix): the bound for any keyer."""

    name = "oracle"

    def key_of(self, store, emb, item_id, text):
        return item_id.split(":")[0]


# --------------------------------------------------------------------------- #
# Retention mechanisms
# --------------------------------------------------------------------------- #


class KeyedRetention(Policy):
    """Retention by overwrite: one slot per key, previous content destroyed."""

    def __init__(self, keyer: Keyer, budget: Optional[int] = None,
                 name: Optional[str] = None):
        self.keyer = keyer
        super().__init__(budget=budget)
        self.name = name or f"overwrite[{keyer.name}]"

    def reset(self) -> None:
        super().reset()
        self._slot_of: Dict[str, int] = {}
        if hasattr(self, "keyer"):
            self.keyer.reset()

    def observe(self, emb: np.ndarray, item_id: str, t: float,
                text: Optional[str] = None) -> None:
        key = self.keyer.key_of(self, emb, item_id, text)
        j = self._slot_of.get(key) if key is not None else None
        if j is None:
            self._push(emb, item_id, t, text, key)
            if key is not None:
                self._slot_of[key] = self._n - 1
            self._evict_if_needed()
        else:
            self._overwrite(j, emb, item_id, t, text)

    def _drop(self, j: int) -> None:
        k = self._keys[j]
        super()._drop(j)
        if k is not None:
            self._slot_of.pop(k, None)
        for kk, jj in list(self._slot_of.items()):      # reindex after shift
            if jj > j:
                self._slot_of[kk] = jj - 1


class TemporalKG(Policy):
    """Zep/Graphiti-style bi-temporal store: supersede, don't delete.

    When a statement keys to an existing slot, that slot is marked *invalid*
    (its edge closes at the new timestamp) and the new statement is appended as
    the valid edge. Retrieval returns valid items first, then invalid ones, so
    a reader sees the current version unless the store holds fewer than k valid
    matches.

    Relative to overwrite this trades footprint for recoverability --- the
    design the false-merge audit argues production systems should adopt --- and
    makes the trade measurable: same current@1 and contamination, honest
    ``slots``, smaller ``active_slots``.
    """

    def __init__(self, keyer: Keyer, budget: Optional[int] = None,
                 name: Optional[str] = None):
        self.keyer = keyer
        super().__init__(budget=budget)
        self.name = name or f"temporal-kg[{keyer.name}]"

    def reset(self) -> None:
        super().reset()
        self._valid: List[bool] = []
        self._slot_of: Dict[str, int] = {}
        if hasattr(self, "keyer"):
            self.keyer.reset()

    def observe(self, emb: np.ndarray, item_id: str, t: float,
                text: Optional[str] = None) -> None:
        key = self.keyer.key_of(self, emb, item_id, text)
        if key is not None and key in self._slot_of:
            self._valid[self._slot_of[key]] = False
        self._push(emb, item_id, t, text, key)
        self._valid.append(True)
        if key is not None:
            self._slot_of[key] = self._n - 1
        self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        """Under a budget, shed invalidated edges before valid ones.

        Invalidation trades footprint for recoverability only while the
        footprint is affordable; under a slot budget the tombstones are what a
        real system gives up first.
        """
        while self.budget is not None and self._n > self.budget:
            invalid = [j for j in range(self._n) if not self._valid[j]]
            self._drop(min(invalid, key=lambda j: self._t[j]) if invalid
                       else int(np.argmin(self._t)))

    def _drop(self, j: int) -> None:
        k = self._keys[j]
        super()._drop(j)
        del self._valid[j]
        if k is not None and self._slot_of.get(k) == j:
            self._slot_of.pop(k, None)
        for kk, jj in list(self._slot_of.items()):
            if jj > j:
                self._slot_of[kk] = jj - 1

    @property
    def active_slots(self) -> int:
        return int(sum(self._valid))

    def rank(self, query: np.ndarray, query_text: Optional[str] = None) -> List[str]:
        if not self._n:
            return []
        s = self._scores(query, query_text)
        valid = np.asarray(self._valid, dtype=bool)
        # valid edges rank above invalidated ones; similarity orders within tier
        order = np.argsort(np.where(valid, -s, -s + 1e6), kind="stable")
        return [self._ids[i] for i in order]


class ReadKeyedRetention(KeyedRetention):
    """Run the same keyer on the *read* path as on the write path.

    Every other rung keys on write and then ranks by cosine, so a keyer that
    correctly merges two statements can still lose the query to a similarity
    accident. This policy resolves the query text through the same keyer and
    ranks items sharing that key first, most recent among them at the top,
    falling back to similarity for the remainder.

    It isolates a question the ladder otherwise cannot ask: how much of the
    anchor-free deficit is a *keying* failure, and how much is an asymmetry in
    where keying is applied? Queries always name the subject in full, so the
    read side is the easy direction --- which is precisely why the comparison
    is informative.
    """

    def __init__(self, keyer: Keyer, budget: Optional[int] = None,
                 name: Optional[str] = None, write_keys: bool = True):
        self.write_keys = write_keys
        super().__init__(keyer, budget=budget,
                         name=name or f"read+write[{keyer.name}]")

    def observe(self, emb, item_id, t, text=None):
        if self.write_keys:
            super().observe(emb, item_id, t, text)
        else:                                    # store-all, key only on read
            key = self.keyer.key_of(self, emb, item_id, text)
            self._push(emb, item_id, t, text, key)
            self._evict_if_needed()

    def rank(self, query: np.ndarray,
             query_text: Optional[str] = None) -> List[str]:
        if not self._n:
            return []
        qkey = self.keyer.key_of(self, query, "", query_text or "")
        sims = self._scores(query, query_text)
        order = list(np.argsort(-sims, kind="stable"))
        if qkey is None:
            return [self._ids[j] for j in order]
        hit = [j for j in order if self._keys[j] == qkey]
        hit.sort(key=lambda j: -self._t[j])       # freshest of the right fact
        rest = [j for j in order if self._keys[j] != qkey]
        return [self._ids[j] for j in hit + rest]


# --------------------------------------------------------------------------- #
# Named rungs (thin constructors over keyer x mechanism)
# --------------------------------------------------------------------------- #


def Reconsolidation(tau: float = 0.9, budget: Optional[int] = None) -> KeyedRetention:
    """Embedding-threshold overwrite --- the realistic freshness-aware method."""
    p = KeyedRetention(ThresholdKeyer(tau), budget=budget)
    p.name = f"reconsolidation(tau={tau}" + (f",B={budget})" if budget else ")")
    p.tau = tau
    return p


def LexicalKey(gazetteer: Sequence[str]) -> KeyedRetention:
    """Verbatim-gazetteer overwrite --- upper bound of the lexical family."""
    return KeyedRetention(LexicalKeyer(gazetteer), name="lexical-key")


def FuzzyKey(gazetteer: Sequence[str], tau: float = 0.34) -> KeyedRetention:
    """Overwrite keyed by fuzzy mention linking --- no alias table."""
    return KeyedRetention(FuzzyKeyer(gazetteer, tau), name=f"fuzzy-key(tau={tau})")


def AliasKey(gazetteer: Sequence[str], alias_map: Dict[str, Set[str]],
             ambiguous: str = "skip") -> KeyedRetention:
    """Entity-linker overwrite --- gazetteer plus the derivable alias inventory."""
    return KeyedRetention(AliasKeyer(gazetteer, alias_map, ambiguous),
                          name=f"alias-key({ambiguous})")


def LLMKey(client, n_cand: int = 4) -> KeyedRetention:
    """LLM-adjudicated overwrite."""
    return KeyedRetention(LLMKeyer(client, n_cand),
                          name=f"llm-key({getattr(client, 'name', 'llm')})")


def ReadWriteFuzzyKey(gazetteer: Sequence[str], tau: float = 0.34,
                      write_keys: bool = True) -> ReadKeyedRetention:
    """Fuzzy entity resolution applied symmetrically, on both paths."""
    return ReadKeyedRetention(
        FuzzyKeyer(gazetteer, tau), write_keys=write_keys,
        name=f"fuzzy-{'rw' if write_keys else 'read'}-key")


def KeepLatestOracle() -> KeyedRetention:
    """Upper bound: keep exactly the latest item per TRUE key."""
    return KeyedRetention(OracleKeyer(), name="keep-latest-oracle")


class TimestampAwareRetention(KeyedRetention):
    """Overwrite keyed retention that keeps the version with the later
    timestamp rather than the one that arrived last.

    ``KeyedRetention`` never reads ``t``: last write wins, which makes it
    immune to corrupted time metadata and hostage to arrival order. This
    variant reads ``t`` and nothing about order, so a stale report that
    arrives late is dropped instead of written over the current value --- and
    a permuted clock now misleads it exactly as it misleads time-rerank. The
    two are one mechanism keyed to opposite temporal signals, which is what
    the corruption grid in ``run_real`` measures.
    """

    def observe(self, emb: np.ndarray, item_id: str, t: float,
                text: Optional[str] = None) -> None:
        key = self.keyer.key_of(self, emb, item_id, text)
        j = self._slot_of.get(key) if key is not None else None
        if j is None:
            self._push(emb, item_id, t, text, key)
            if key is not None:
                self._slot_of[key] = self._n - 1
            self._evict_if_needed()
        elif t >= self._t[j]:
            self._overwrite(j, emb, item_id, t, text)


def NewerOnlyLexicalKey(gazetteer: Sequence[str]) -> TimestampAwareRetention:
    """Gazetteer overwrite that keeps the newer-stamped version."""
    return TimestampAwareRetention(LexicalKeyer(gazetteer),
                                   name="lexical-key-newer")


def NewerOnlyOracle() -> TimestampAwareRetention:
    """True-key overwrite that keeps the newer-stamped version."""
    return TimestampAwareRetention(OracleKeyer(), name="keep-newest-oracle")
