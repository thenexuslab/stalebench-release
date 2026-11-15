"""Anchor-free verbalization: alias surface forms for fact subjects.

v0.2 of StaleBench measured a *three-tier keying ladder* and found that a
gazetteer string keyer nearly matched the true-key oracle, because every
templated statement carried the subject's surface form verbatim. That made
"same-fact detection" look solved and pushed the real open problem out of
reach of the instrument: detection **without a lexical anchor**.

This module supplies the missing stress. After a fact's first (introducing)
statement, later statements may refer to the subject by an *alias* --- the
short forms real logs actually use:

* head token: ``Marcelo Ebrard`` -> ``Ebrard``, ``Malmo FF`` -> ``Malmo``
* leading token: ``Marcelo Ebrard`` -> ``Marcelo``
* initialed: ``Marcelo Ebrard`` -> ``M. Ebrard``
* token-dropped phrase (3+ tokens): ``Los Angeles Rams`` -> ``Los Angeles``

Two design rules keep the track fair and diagnostic:

1. **Realism filter on generation.** A candidate alias is rejected if it is a
   bare type/stopword token (``FC``, ``Line``, ``Ministry``, ``of``), which no
   writer would use as a standalone reference. Without this the track would be
   an unfair strawman rather than a measurement.
2. **No ambiguity filter.** Aliases that collide across subjects (real
   surnames: ``Harris``, ``Sidhu``) are *kept*, because entity-resolution
   ambiguity is part of the problem being priced. :func:`alias_stats` reports
   how much of it there is, and the alias-aware keyer must live with it.

Every alias is derivable from the subject string alone, so an entity linker
holding the same public subject inventory can in principle resolve it --- the
gap a keyer then pays is the cost of *doing* entity resolution, not of missing
information.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Set

# Tokens that are organizational type markers or function words: valid inside a
# name, never a standalone reference to it.
_TYPE_TOKENS: Set[str] = {
    "fc", "f.c.", "ff", "cf", "sc", "ac", "afc", "bc", "sk", "if", "ik",
    "line", "society", "club", "university", "college", "school", "institute",
    "metropolis", "metropole", "métropole", "municipality", "ministry",
    "department", "party", "company", "corporation", "inc.", "inc", "ltd.",
    "ltd", "llc", "group", "team", "council", "board", "commission", "office",
    "of", "the", "and", "de", "del", "della", "di", "da", "la", "le", "el",
    "los", "las", "van", "von", "der", "den", "bin", "al", "ibn", "st.",
    "jr.", "sr.", "ii", "iii", "iv",
    # generic category words that head many organization names
    "embassy", "commission", "court", "service", "resort", "casino", "hotel",
    "airport", "station", "museum", "theatre", "theater", "hospital", "bank",
    "union", "federation", "association", "academy", "institute", "centre",
    "center", "trust", "authority", "agency", "basketball", "football",
    "baseball", "hockey", "cricket", "rugby", "team", "squad", "district",
    "region", "province", "county", "city", "town", "village", "state",
    "republic", "kingdom", "ministry", "office", "school", "college",
}


def _usable(tok: str) -> bool:
    """Is a single token a plausible standalone reference to its subject?"""
    t = tok.strip().lower()
    return len(t) >= 3 and t not in _TYPE_TOKENS and any(c.isalpha() for c in t)


def _usable_phrase(phrase: str) -> bool:
    toks = phrase.split()
    return bool(toks) and any(_usable(t) for t in toks)


def alias_forms(name: str, person: bool = True) -> List[str]:
    """Deterministic alias surface forms for a subject name (may be empty).

    People and organizations shorten differently, and conflating them produces
    forms no writer uses. A personal name is *Given Family*: the head token is
    the distinctive part (``Ebrard``), the leading token is a usable given name
    (``Marcelo``), and the initialed form is idiomatic (``M. Ebrard``). An
    organization or place name is *Modifier Type*: the head token is a generic
    category word (``Casino``, ``Line``, ``basketball``) that refers to nothing
    on its own, and initialing it is absurd (``W. Casino``). So organizations
    keep only the leading token and the drop-last shortening
    (``Los Angeles Rams`` -> ``Los Angeles``).
    """
    toks = name.split()
    if len(toks) < 2:
        return []                       # single-token names have no short form
    head, first = toks[-1], toks[0]
    cands: List[str] = []
    if person and _usable(head):
        cands.append(head)
    if _usable(first):
        cands.append(first)
    if len(toks) >= 3:
        # drop-last shortening; the remainder must end on a content word, so
        # "Municipality of Thessaloniki" does not shorten to "Municipality of"
        phrase = " ".join(toks[:-1])
        if (_usable_phrase(phrase) and _usable(toks[-2])
                and toks[-2][:1].isupper()):
            cands.append(phrase)
    if person and _usable(head) and _usable(first):
        cands.append(f"{first[0]}. {head}")
    seen: Set[str] = set()
    out: List[str] = []
    for c in cands:
        c = c.strip(" ,;:.\u2019'\"").strip()
        if c and c != name and c not in seen and any(ch.isalpha() for ch in c):
            seen.add(c)
            out.append(c)
    return out


def _pairs(subjects):
    """Accept either bare names or (name, is_person) pairs."""
    for s in subjects:
        yield (s, True) if isinstance(s, str) else (s[0], bool(s[1]))


def build_alias_map(subjects: Iterable) -> Dict[str, Set[str]]:
    """alias surface form -> set of canonical subjects that generate it.

    A key with more than one canonical subject is *ambiguous*: an entity linker
    seeing that alias in isolation cannot resolve it.
    """
    amap: Dict[str, Set[str]] = {}
    for name, person in _pairs(subjects):
        for a in alias_forms(name, person):
            amap.setdefault(a, set()).add(name)
    return amap


def alias_map_from_pools(pools: Dict[str, Iterable[str]]) -> Dict[str, Set[str]]:
    """The same map from an explicit ``subject -> forms`` pool.

    The mechanical map is derivable from the names alone; a pool that came from
    elsewhere (Wikidata aliases, Wikipedia redirects) has to be handed over
    explicitly, and the entity-linker rung then holds exactly the forms the
    stream drew from --- the same "given the generating inventory" reading.
    """
    amap: Dict[str, Set[str]] = {}
    for name, forms in pools.items():
        for a in forms:
            amap.setdefault(a, set()).add(name)
    return amap


def pool_stats(pools: Dict[str, Iterable[str]]) -> Dict[str, float]:
    """Coverage and ambiguity of an explicit pool (:func:`alias_stats` without
    the class composition, which only the mechanical generator defines)."""
    per = {n: list(v) for n, v in pools.items()}
    amap = alias_map_from_pools(per)
    ambiguous = {a for a, owners in amap.items() if len(owners) > 1}
    n_with = sum(1 for v in per.values() if v)
    n_amb_subj = sum(1 for v in per.values() if any(a in ambiguous for a in v))
    return {
        "n_subjects": len(per),
        "n_with_alias": n_with,
        "frac_with_alias": n_with / max(1, len(per)),
        "n_alias_forms": len(amap),
        "n_ambiguous_forms": len(ambiguous),
        "frac_ambiguous_forms": len(ambiguous) / max(1, len(amap)),
        "n_subjects_touched_by_ambiguity": n_amb_subj,
        "frac_subjects_touched_by_ambiguity": n_amb_subj / max(1, len(per)),
        "mean_aliases_per_subject": (sum(len(v) for v in per.values())
                                     / max(1, len(per))),
    }


def alias_stats(subjects: Iterable) -> Dict[str, float]:
    """Coverage, ambiguity, and class composition of the alias inventory."""
    pairs = list(_pairs(subjects))
    subjects = [n for n, _ in pairs]
    per = {n: alias_forms(n, p) for n, p in pairs}
    amap = build_alias_map(subjects)
    ambiguous = {a for a, owners in amap.items() if len(owners) > 1}
    n_with = sum(1 for v in per.values() if v)
    n_amb_subj = sum(1 for v in per.values() if any(a in ambiguous for a in v))
    return {
        "n_subjects": len(subjects),
        "n_with_alias": n_with,
        "frac_with_alias": n_with / max(1, len(subjects)),
        "n_alias_forms": len(amap),
        "n_ambiguous_forms": len(ambiguous),
        "frac_ambiguous_forms": len(ambiguous) / max(1, len(amap)),
        "n_subjects_touched_by_ambiguity": n_amb_subj,
        "frac_subjects_touched_by_ambiguity": n_amb_subj / max(1, len(subjects)),
        "mean_aliases_per_subject": (sum(len(v) for v in per.values())
                                     / max(1, len(subjects))),
        "composition": _composition(pairs),
    }


def _composition(pairs) -> Dict[str, float]:
    """Share of generated forms by class, for the appendix."""
    counts = {"head_token": 0, "leading_token": 0, "initialed": 0,
              "drop_last": 0}
    for name, person in pairs:
        toks = name.split()
        for a in alias_forms(name, person):
            if a.startswith(f"{toks[0][0]}. ") and len(a.split()) == 2:
                counts["initialed"] += 1
            elif a == toks[-1]:
                counts["head_token"] += 1
            elif a == toks[0]:
                counts["leading_token"] += 1
            else:
                counts["drop_last"] += 1
    total = max(1, sum(counts.values()))
    return {k: v / total for k, v in counts.items()}
