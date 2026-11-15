"""Reader track: does retrieval-level staleness actually cost answers?

StaleBench v0.2 measured *contamination@k* --- the top-k holding both a current
and a stale version of the queried fact --- and argued from the
knowledge-conflict literature that a reader handed such a context is unreliable.
That was an inherited assumption. This module closes the loop by putting a
reader at the end of the pipeline and measuring answer accuracy directly.

Two experiments use it:

1. **Policy track** --- each policy's own top-k becomes the reader's context.
   Gives end-task accuracy per policy, and (since a store-all policy produces
   both clean and contaminated contexts) the *within-policy* contrast between
   them.
2. **Dose--response probe** --- contexts are constructed rather than retrieved:
   the current statement plus 0, 1, or 2 stale versions of the same fact, with
   controlled ordering. This isolates the causal effect of contamination from
   everything a retriever might co-vary with it.

Both run with and without the timestamp surfaced in the context, which answers
the reader-level form of the cheapest objection: *why not just show the model
the dates and let it sort them out?*

Answers are scored by normalized containment against the source-defined current
value and against the fact's stale values, so the reader's failure is decomposed
the same way retrieval's is: correct / stale / other.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional, Sequence, Tuple

READER_PROMPT = """Memories:
{context}

Question: {question}
Answer using only the memories above. Reply with the answer only, no explanation."""

# closed-book control: no store at all, so the reader can only use what it
# already knows. Without this the context-conditioned numbers are confounded by
# parametric knowledge of these (real, public) Wikidata facts.
CLOSED_BOOK_PROMPT = """Question: {question}
Reply with the answer only, no explanation."""

_ARTICLES = {"the", "a", "an", "of", "for"}


def _norm(s: str) -> str:
    """Casefold, strip accents/punctuation/articles: a lenient short-answer key."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^\w\s]", " ", s.lower())
    return " ".join(w for w in s.split() if w not in _ARTICLES)


def _matches(answer: str, value: str) -> bool:
    """Containment either way, with a length guard against degenerate hits."""
    a, v = _norm(answer), _norm(value)
    if not a or not v:
        return False
    if v in a:
        return True
    # a short answer may name the value's head ("Yamaha" for "Yamaha Motor
    # Racing"); require a whole-token prefix of >=4 chars, so "Yamah" and
    # "Real" do not silently satisfy "Yamaha ..." and "Real Madrid"
    return (len(a) >= 4 and v.startswith(a)
            and v[len(a):len(a) + 1] in ("", " "))


def format_context(texts: Sequence[str], years: Optional[Sequence[int]] = None,
                   ) -> str:
    """Numbered memory list; ``years`` surfaces the stored timestamp metadata."""
    if years is None:
        return "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    return "\n".join(f"{i + 1}. [{y}] {t}" for i, (t, y) in
                     enumerate(zip(texts, years)))


def build_prompt(texts: Sequence[str], question: str,
                 years: Optional[Sequence[int]] = None) -> str:
    if not texts:
        return CLOSED_BOOK_PROMPT.format(question=question)
    return READER_PROMPT.format(context=format_context(texts, years),
                                question=question)


def score_answer(answer: str, gold: str,
                 stale_values: Sequence[str]) -> Dict[str, float]:
    """Decompose a reader answer the way retrieval is decomposed.

    ``correct``  the current value is named;
    ``stale``    a superseded value is named and the current one is not ---
                 the reader-level image of the retrieval failure;
    ``other``    neither (refusal, hallucination, a different entity);
    ``hedged``   both are named (counted as not correct, reported separately).
    """
    hit_gold = _matches(answer, gold)
    hit_stale = any(_matches(answer, v) for v in stale_values if v != gold)
    return {"correct": float(hit_gold and not hit_stale),
            "stale": float(hit_stale and not hit_gold),
            "hedged": float(hit_gold and hit_stale),
            "other": float(not hit_gold and not hit_stale),
            "abstain": float(_is_abstention(answer))}


_ABSTAIN = re.compile(
    r"\b(not (?:enough|specified|mentioned|available|stated|provided)|"
    r"unknown|unclear|cannot (?:be )?determin|can(?:no|')t tell|"
    r"no (?:information|memory|memories|mention|answer)|"
    r"(?:do|does|did|is|are) not (?:say|specify|mention|state|contain|include))",
    re.I)


def _is_abstention(answer: str) -> bool:
    return bool(_ABSTAIN.search(answer or ""))


def stale_values_of(runs: Sequence[Tuple[int, str]], current: str) -> List[str]:
    """Distinct superseded values of a fact."""
    seen, out = set(), []
    for _, v in runs:
        if v != current and v not in seen:
            seen.add(v)
            out.append(v)
    return out
