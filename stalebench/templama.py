"""TempLAMA instantiation of StaleBench: real updating facts, source-defined labels.

TempLAMA (Dhingra et al., 2022) provides (subject, relation, value) triples with
*yearly validity* (2010--2020) derived from Wikidata start/end qualifiers. The
currency label therefore comes from the data source, never from any retrieval
score. That is the benchmark's central guardrail.

Construction:

1. Group records by fact key (``subject_relation``). Keep facts that are
   **single-valued** in every year (crisp top-1 scoring) and whose value
   **changes** at least once in the window (892 of 5,839 keys).
2. Collapse consecutive years with the same value into a *run*; each run is
   verbalized as one natural-language statement (canonical template = the
   TempLAMA cloze with the value filled in; with probability ``paraphrase_p``
   a relation-specific paraphrase template is used instead). Statements carry
   the year only as **metadata** ``t``, not in the text.
3. The query per fact is a **value-agnostic** current-value question naming the
   subject only. Gold = the run(s) whose value equals the value valid in the
   final year (value-based, so an A→B→A fact scores its early A as current).
4. Facts are split val/test by a hash of the fact key; all hyperparameter
   tuning (tau, k, lam) happens on val only.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from stalebench.alias import alias_forms

import numpy as np

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
_YEAR0 = 2010

# --------------------------------------------------------------------------- #
# Relation templates
# --------------------------------------------------------------------------- #
# "cloze" is the exact TempLAMA query pattern (used to recover the subject
# surface form). "statements"[0] filled with the value == the cloze statement;
# the rest are paraphrases. "questions"[0] is the canonical value-agnostic
# current-value query.

RELATIONS: Dict[str, Dict[str, object]] = {
    "P54": {
        "cloze": ("", " plays for _X_."),
        "statements": ["{s} plays for {v}.",
                       "{s} is on the roster of {v}.",
                       "{s} has signed with {v}.",
                       "{v} is the club {s} turns out for."],
        "questions": ["Which team does {s} currently play for?",
                      "What is {s}'s current club?",
                      "Who does {s} play for now?"],
    },
    "P39": {
        "cloze": ("", " holds the position of _X_."),
        "statements": ["{s} holds the position of {v}.",
                       "{s} serves as {v}.",
                       "{s} now occupies the office of {v}.",
                       "The position held by {s} is {v}."],
        "questions": ["What position does {s} currently hold?",
                      "What office does {s} hold now?",
                      "What is {s}'s current position?"],
    },
    "P108": {
        "cloze": ("", " works for _X_."),
        "statements": ["{s} works for {v}.",
                       "{s} is employed by {v}.",
                       "{s} is on the payroll of {v}.",
                       "{v} employs {s}."],
        "questions": ["Who does {s} currently work for?",
                      "What is {s}'s current employer?",
                      "Where does {s} work now?"],
    },
    "P102": {
        "cloze": ("", " is a member of the _X_."),
        "statements": ["{s} is a member of the {v}.",
                       "{s} belongs to the {v}.",
                       "{s} is affiliated with the {v}.",
                       "Politically, {s} sits with the {v}."],
        "questions": ["Which party is {s} currently a member of?",
                      "What is {s}'s current political party?",
                      "Which party does {s} belong to now?"],
    },
    "P286": {
        "cloze": ("_X_ is the head coach of ", "."),
        "statements": ["{v} is the head coach of {s}.",
                       "{s} is coached by {v}.",
                       "{v} runs the bench for {s}.",
                       "The head-coaching job at {s} belongs to {v}."],
        "questions": ["Who is currently the head coach of {s}?",
                      "Who coaches {s} now?",
                      "What is the name of {s}'s current head coach?"],
    },
    "P488": {
        "cloze": ("_X_ is the chair of ", "."),
        "statements": ["{v} is the chair of {s}.",
                       "{s} is chaired by {v}.",
                       "{v} leads the board of {s}.",
                       "The chair of {s} is {v}."],
        "questions": ["Who is currently the chair of {s}?",
                      "Who chairs {s} now?",
                      "Who currently leads the board of {s}?"],
    },
    "P6": {
        "cloze": ("_X_ is the head of the government of ", "."),
        "statements": ["{v} is the head of the government of {s}.",
                       "{s}'s government is led by {v}.",
                       "{v} heads the government of {s}.",
                       "The government of {s} is run by {v}."],
        "questions": ["Who is currently the head of the government of {s}?",
                      "Who leads {s}'s government now?",
                      "Who currently heads the government of {s}?"],
    },
    "P127": {
        "cloze": ("", " is owned by _X_."),
        "statements": ["{s} is owned by {v}.",
                       "{v} owns {s}.",
                       "{s} is a subsidiary of {v}.",
                       "Ownership of {s} rests with {v}."],
        "questions": ["Who currently owns {s}?",
                      "What is the current owner of {s}?",
                      "Who is {s} owned by now?"],
    },
    "P69": {
        "cloze": ("", " attended _X_."),
        "statements": ["{s} attended {v}.",
                       "{s} studied at {v}.",
                       "{s} was educated at {v}.",
                       "{v} is where {s} went to school."],
        "questions": ["Which institution did {s} attend most recently?",
                      "Where did {s} study most recently?",
                      "What is the latest school {s} attended?"],
    },
}


# Relations whose *subject* is a person; the rest are organizations or places.
# Alias shortening differs by class (see stalebench.alias.alias_forms).
PERSON_SUBJECT_RELATIONS = {"P54", "P39", "P108", "P102", "P69"}


def subject_is_person(relation: str) -> bool:
    return relation in PERSON_SUBJECT_RELATIONS


def subjects_with_class(facts) -> List[Tuple[str, bool]]:
    """(subject, is_person) pairs --- the inventory the alias track needs."""
    return [(f.subject, subject_is_person(f.relation)) for f in facts]


def subject_of(query: str, relation: str) -> Optional[str]:
    """Recover the subject surface form from a TempLAMA cloze query."""
    pre, post = RELATIONS[relation]["cloze"]
    if not query.startswith(pre) or not query.endswith(post):
        return None
    s = query[len(pre): len(query) - len(post)]
    return s if s else None


# --------------------------------------------------------------------------- #
# Alias sources
# --------------------------------------------------------------------------- #
# Where the anchor-free track's short forms come from is its own axis. The v1.1
# stream derives them mechanically from the name (stalebench.alias); the
# *natural* source uses forms real writers used --- Wikidata aliases and
# English Wikipedia redirects, built by v2/natural_aliases.py --- which covers
# only some subjects, hence two flavours: "natural" leaves an uncovered subject
# anchored on every statement, "natural+fallback" gives it the mechanical forms.

ALIAS_SOURCES = ("mechanical", "natural", "natural+fallback")
NATURAL_INVENTORY = os.path.join(os.path.dirname(__file__), "..", "v2", "out",
                                 "alias_inventory_natural.json")


def load_natural_inventory(path: Optional[str] = None) -> Dict[str, Dict]:
    """QID -> record holding the subject's ``natural`` forms."""
    path = path or NATURAL_INVENTORY
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"natural alias inventory missing at {os.path.abspath(path)}; build "
            f"it with `python3 v2/natural_aliases.py [--split val]`")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def natural_forms(f: Fact, inventory: Dict[str, Dict]) -> List[str]:
    """A subject's natural short forms (empty if the inventory has none)."""
    rec = inventory.get(f.key.split("_")[0]) or {}
    full = f.subject.strip().lower()
    out: List[str] = []
    for a in rec.get("natural", []):
        a = a.strip()
        # a form that is the full name would re-anchor the statement
        if a and a.lower() != full and a not in out:
            out.append(a)
    return out


def alias_pool(f: Fact, alias_source: str = "mechanical",
               inventory: Optional[Dict[str, Dict]] = None) -> List[str]:
    """Short forms a non-introducing statement about ``f`` may use."""
    if alias_source not in ALIAS_SOURCES:
        raise ValueError(f"alias_source must be one of {ALIAS_SOURCES}, "
                         f"got {alias_source!r}")
    mech = alias_forms(f.subject, subject_is_person(f.relation))
    if alias_source == "mechanical":
        return mech
    if inventory is None:
        raise ValueError(f"alias_source={alias_source!r} needs the natural "
                         f"inventory (load_natural_inventory)")
    nat = natural_forms(f, inventory)
    if nat or alias_source == "natural":
        return nat
    return mech


def alias_pools(facts: List[Fact], alias_source: str = "mechanical",
                inventory: Optional[Dict[str, Dict]] = None
                ) -> Dict[str, List[str]]:
    """subject -> pool: the inventory the entity-linker rung is handed.

    For the mechanical source this equals what ``alias.build_alias_map``
    derives from the names; for the natural sources it is the pool the stream
    actually drew from, which preserves the rung's reading of "given the
    generating inventory". Two facts sharing a subject string pool their forms.
    """
    pools: Dict[str, List[str]] = {}
    for f in facts:
        pool = pools.setdefault(f.subject, [])
        pool.extend(a for a in alias_pool(f, alias_source, inventory)
                    if a not in pool)
    return pools


# --------------------------------------------------------------------------- #
# Facts
# --------------------------------------------------------------------------- #


@dataclass
class Fact:
    key: str                              # "Qxxxx_Pyy"
    relation: str
    subject: str
    by_year: Dict[int, str]               # year -> value name (first, if several)
    runs: List[Tuple[int, str]] = field(default_factory=list)  # (start_year, value)
    # multi-valued facts hold several simultaneously-valid values per year (a
    # footballer's club *and* country). They are excluded from the v1.0 queried
    # set because they defeat crisp top-1 scoring; ``values_by_year`` carries the
    # full set so the v1.1 track can score them set-valued.
    values_by_year: Dict[int, Tuple[str, ...]] = field(default_factory=dict)

    @property
    def current_value(self) -> str:
        return self.by_year[max(self.by_year)]

    @property
    def current_values(self) -> Tuple[str, ...]:
        """Every value valid in the final year (one for single-valued facts)."""
        if self.values_by_year:
            return self.values_by_year[max(self.values_by_year)]
        return (self.current_value,)

    @property
    def is_multi_valued(self) -> bool:
        return any(len(v) > 1 for v in self.values_by_year.values())

    @property
    def current_runs(self) -> Set[int]:
        """Run indices whose value is current.

        Scored by *value*, so a fact that returns to an earlier value (A-B-A)
        counts its early A as current; for a multi-valued fact every run whose
        value is valid in the final year counts.
        """
        cur = set(self.current_values)
        return {i for i, (_, v) in enumerate(self.runs) if v in cur}


def _read_raw(data_dir: str) -> Tuple[Dict[str, Dict[int, Tuple[str, ...]]],
                                      Dict[str, Tuple[str, str]]]:
    raw: Dict[str, Dict[int, Tuple[str, ...]]] = {}
    meta: Dict[str, Tuple[str, str]] = {}
    for split in ("train", "val", "test"):
        path = os.path.join(data_dir, f"templama_{split}.json")
        with open(path) as fh:
            for line in fh:
                r = json.loads(line)
                key = r["id"].rsplit("_", 1)[0]
                year = int(r["date"])
                vals = tuple(sorted(a["name"] for a in r["answer"]))
                raw.setdefault(key, {})[year] = vals
                meta[key] = (r["relation"], r["query"])
    return raw, meta


def load_facts(data_dir: str = _DATA_DIR) -> Tuple[List[Fact], Dict[str, int]]:
    """Load TempLAMA and build the single-valued changing-fact set.

    Returns (facts, stats) where stats records n at each filter step.
    """
    raw, meta = _read_raw(data_dir)

    stats = {"fact_keys": len(raw)}
    facts: List[Fact] = []
    n_single = 0
    for key, ys in raw.items():
        if not all(len(v) == 1 for v in ys.values()):
            continue
        n_single += 1
        if len({v[0] for v in ys.values()}) < 2:
            continue
        relation, query = meta[key]
        subject = subject_of(query, relation)
        if subject is None:
            continue
        by_year = {y: v[0] for y, v in sorted(ys.items())}
        runs: List[Tuple[int, str]] = []
        for y, v in sorted(by_year.items()):
            if not runs or runs[-1][1] != v:
                runs.append((y, v))
        facts.append(Fact(key, relation, subject, by_year, runs))
    stats["single_valued"] = n_single
    stats["single_valued_changing"] = len(facts)
    return facts, stats


def load_all_changing_facts(data_dir: str = _DATA_DIR
                            ) -> Tuple[List[Fact], Dict[str, int]]:
    """Every changing fact, single- or multi-valued (the v1.1 queried set).

    v1.0 restricts to facts that are single-valued in every year, which is 892
    of 5,839 keys, because set-valued gold complicates top-1 scoring. Keeping
    the multi-valued ones raises the queried set roughly sixfold and is scored
    the same way: a top-1 is current if its value is valid in the final year.
    Each simultaneously-valid value contributes its own run, so a fact with two
    concurrent values has two current runs.
    """
    raw, meta = _read_raw(data_dir)
    stats = {"fact_keys": len(raw)}
    facts: List[Fact] = []
    n_changing = 0
    for key, ys in raw.items():
        if len({tuple(v) for v in ys.values()}) < 2:
            continue                                   # never changes
        n_changing += 1
        relation, query = meta[key]
        if relation not in RELATIONS:
            continue
        subject = subject_of(query, relation)
        if subject is None:
            continue
        vals = {y: v for y, v in sorted(ys.items())}
        by_year = {y: v[0] for y, v in vals.items()}
        # one run per (interval, value): a year holding two values emits two
        runs: List[Tuple[int, str]] = []
        prev: Tuple[str, ...] = ()
        for y, v in sorted(vals.items()):
            if v != prev:
                for one in v:
                    runs.append((y, one))
                prev = v
        facts.append(Fact(key, relation, subject, by_year, runs, vals))
    stats["changing"] = n_changing
    stats["changing_with_subject"] = len(facts)
    stats["multi_valued"] = sum(1 for f in facts if f.is_multi_valued)
    return facts, stats


def load_distractor_facts(data_dir: str = _DATA_DIR) -> List[Fact]:
    """Facts excluded from the queried set, kept as store noise.

    The 4,939 fact keys that are multi-valued in some year cannot be scored
    with crisp top-1 gold, but they are perfectly good *statements* to store.
    Including them makes the retrieval pool realistic (a store holding one
    topic's worth of facts is an easy store) and puts real pressure on
    same-fact keying, which must now avoid merging across ~6x more neighbours.
    They are never queried, so they need no currency labels.
    """
    raw, meta = _read_raw(data_dir)
    out: List[Fact] = []
    for key, ys in raw.items():
        if all(len(v) == 1 for v in ys.values()):
            continue                                   # queried-set candidate
        relation, query = meta[key]
        if relation not in RELATIONS:
            continue
        subject = subject_of(query, relation)
        if subject is None:
            continue
        by_year = {y: v[0] for y, v in sorted(ys.items())}   # first answer name
        runs: List[Tuple[int, str]] = []
        for y, v in sorted(by_year.items()):
            if not runs or runs[-1][1] != v:
                runs.append((y, v))
        out.append(Fact(key, relation, subject, by_year, runs))
    return out


def split_facts(facts: List[Fact]) -> Tuple[List[Fact], List[Fact]]:
    """Deterministic val/test split by fact-key hash (tuning happens on val)."""
    val, test = [], []
    for f in facts:
        h = int(hashlib.md5(f.key.encode()).hexdigest(), 16)
        (val if h % 2 == 0 else test).append(f)
    return val, test


# --------------------------------------------------------------------------- #
# Benchmark instance (events + queries), ready for any Policy
# --------------------------------------------------------------------------- #


@dataclass
class RealInstance:
    events: List[Tuple[str, str, float]]      # (text, item_id, t) chronological
    queries: List[Tuple[str, str, Set[int]]]  # (question, fact_key_id, current_runs)
    n_facts: int
    # reader-track and audit side tables (item_id -> ...)
    text_of: Dict[str, str] = field(default_factory=dict)
    value_of: Dict[str, str] = field(default_factory=dict)
    year_of: Dict[str, int] = field(default_factory=dict)
    gold: List[str] = field(default_factory=list)      # current value per query
    subjects: List[str] = field(default_factory=list)  # subject per query
    n_alias_events: int = 0
    n_distractor_events: int = 0
    # anchor-free coverage: which source supplied the short forms, how many
    # alias-eligible statements were actually aliased, how many subjects the
    # natural inventory reached, and which inventory file that was
    alias_coverage: Dict[str, object] = field(default_factory=dict)


def build_instance(facts: List[Fact], paraphrase_p: float = 0.0,
                   seed: int = 0,
                   query_template_seed: Optional[int] = None,
                   alias_p: float = 0.0,
                   distractors: Optional[List[Fact]] = None,
                   alias_source: str = "mechanical",
                   alias_inventory: Optional[str] = None) -> RealInstance:
    """Verbalize facts into a chronological statement stream + queries.

    ``paraphrase_p``: probability a statement uses a non-canonical template —
    the surface-variation stress on similarity-based same-key detection.
    ``query_template_seed``: if set, each query samples uniformly from the
    relation's question templates (query-side robustness check); default uses
    the canonical question only.
    ``alias_p``: probability that a *non-introducing* statement (run index
    >= 1) refers to its subject by an alias instead of the full name — the
    anchor-free track. The first statement about a fact always introduces the
    full name, so every fact remains resolvable in principle; the query always
    uses the full name. At ``alias_p=1`` a verbatim-gazetteer keyer can key
    only the introducing statement of each fact.
    ``alias_source``: where the short forms come from — ``"mechanical"``
    (derived from the name, the v1.1 stream), ``"natural"`` (Wikidata aliases
    and Wikipedia redirects; a subject the inventory does not cover keeps its
    full name on every statement) or ``"natural+fallback"`` (an uncovered
    subject uses the mechanical forms instead). Every source draws from the
    same alias RNG stream, so the configurations differ only in the pool.
    ``alias_inventory``: path of the natural inventory (default
    ``v2/out/alias_inventory_natural.json``); required to exist for the natural
    sources.
    ``distractors``: extra facts streamed into the store but never queried
    (realistic retrieval pool; ids are prefixed ``d`` so the metric counts them
    as wrong-fact misses).

    Alias and distractor draws use their own RNG streams, so ``alias_p=0`` with
    no distractors reproduces the v0.2 stream exactly.
    """
    if alias_source not in ALIAS_SOURCES:
        raise ValueError(f"alias_source must be one of {ALIAS_SOURCES}, "
                         f"got {alias_source!r}")
    inventory = (load_natural_inventory(alias_inventory)
                 if alias_source != "mechanical" else None)
    pool_of = [alias_pool(f, alias_source, inventory) for f in facts]
    rng = np.random.default_rng(seed)
    arng = np.random.default_rng(seed + 10_000)          # alias stream
    qrng = (np.random.default_rng(query_template_seed)
            if query_template_seed is not None else None)
    tagged: List[Tuple[int, int, str, str]] = []   # (year, jitter, text, id)
    queries: List[Tuple[str, str, Set[int]]] = []
    value_of: Dict[str, str] = {}
    gold: List[str] = []
    subjects: List[str] = []
    n_alias = n_eligible = 0
    for fi, f in enumerate(facts):
        st = RELATIONS[f.relation]["statements"]
        aliases = pool_of[fi] if alias_p > 0 else []
        for ri, (year, value) in enumerate(f.runs):
            if paraphrase_p > 0 and rng.random() < paraphrase_p:
                tpl = st[1 + int(rng.integers(0, len(st) - 1))]
            else:
                tpl = st[0]
            surface = f.subject
            n_eligible += ri > 0
            # the alias draw is taken only when there is a pool to draw from,
            # which is what keeps the mechanical stream byte-identical to v1.1
            if ri > 0 and aliases and arng.random() < alias_p:
                surface = aliases[int(arng.integers(0, len(aliases)))]
                n_alias += 1
            iid = f"f{fi}:{ri}"
            tagged.append((year, int(rng.integers(0, 1 << 30)),
                           tpl.format(s=surface, v=value), iid))
            value_of[iid] = value
        qs = RELATIONS[f.relation]["questions"]
        qi = int(qrng.integers(0, len(qs))) if qrng is not None else 0
        queries.append((qs[qi].format(s=f.subject), f"f{fi}", f.current_runs))
        gold.append(f.current_value)
        subjects.append(f.subject)

    n_distract = 0
    if distractors:
        drng = np.random.default_rng(seed + 20_000)      # distractor stream
        for di, f in enumerate(distractors):
            st = RELATIONS[f.relation]["statements"]
            for ri, (year, value) in enumerate(f.runs):
                if paraphrase_p > 0 and drng.random() < paraphrase_p:
                    tpl = st[1 + int(drng.integers(0, len(st) - 1))]
                else:
                    tpl = st[0]
                iid = f"d{di}:{ri}"
                tagged.append((year, int(drng.integers(0, 1 << 30)),
                               tpl.format(s=f.subject, v=value), iid))
                value_of[iid] = value
                n_distract += 1

    tagged.sort()                                  # chronological; random within year
    events = [(text, iid, float(year - _YEAR0) + 1e-4 * seq)
              for seq, (year, _, text, iid) in enumerate(tagged)]

    # subjects are counted as entities (QIDs): one subject can carry two facts
    qids = [f.key.split("_")[0] for f in facts]
    with_forms = {q for q, p in zip(qids, pool_of) if p}
    natural = ({q for q, f in zip(qids, facts) if natural_forms(f, inventory)}
               if inventory is not None else set())
    coverage: Dict[str, object] = {
        "alias_source": alias_source,
        "alias_p": alias_p,
        "n_facts": len(facts),
        "n_subjects": len(set(qids)),
        "n_alias_eligible": n_eligible,
        "n_alias_events": n_alias,
        "n_subjects_with_forms": len(with_forms),
        "n_subjects_natural": len(natural),
        "n_subjects_fallback": len(with_forms - natural),
        "inventory_path": (os.path.abspath(alias_inventory or NATURAL_INVENTORY)
                           if inventory is not None else None),
        "inventory_size": len(inventory) if inventory is not None else 0,
    }
    return RealInstance(events, queries, len(facts),
                        text_of={iid: text for text, iid, _ in events},
                        value_of=value_of,
                        year_of={iid: year for year, _, _, iid in tagged},
                        gold=gold, subjects=subjects,
                        n_alias_events=n_alias, n_distractor_events=n_distract,
                        alias_coverage=coverage)
