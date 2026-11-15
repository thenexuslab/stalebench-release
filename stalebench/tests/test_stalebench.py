"""Tests for the StaleBench controlled benchmark, policies, metrics, and the
TempLAMA real-data track (loader tests skip if data/ is absent)."""

import os

import numpy as np
import pytest

from stalebench.bm25 import BM25
from stalebench.env import make_stream
from stalebench.metrics import (_decode, bootstrap_ci, holm, paired_permutation,
                                score_ranking, score_stream)
from stalebench.policies import (BM25RAG, CosineRAG, KeepLatestOracle,
                                 LexicalKey, Recency, Reconsolidation,
                                 RecencyWeighted, TimeRerank)

_DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data",
                     "templama_val.json")


# --------------------------------------------------------------------------- #
# Controlled environment
# --------------------------------------------------------------------------- #


def test_stream_labels_are_generator_defined():
    st = make_stream(n_entities=6, n_steps=120, seed=0)
    seen = {e: [] for e in range(6)}
    for _, e, v, _ in st.events:
        seen[e].append(v)
    for q, e, cur in st.queries:
        assert cur == max(seen[e])
    st2 = make_stream(n_entities=6, n_steps=120, seed=0)
    assert np.allclose(st.queries[0][0], st2.queries[0][0])


def test_key_jitter_perturbs_same_entity_events():
    def mean_pairwise_same_entity(jitter):
        st = make_stream(n_entities=4, n_steps=80, value_gap=0.0,
                         key_jitter=jitter, seed=3)
        sims = []
        by_e = {}
        for emb, e, _, _ in st.events:
            by_e.setdefault(e, []).append(emb)
        for embs in by_e.values():
            for i in range(len(embs)):
                for j in range(i + 1, len(embs)):
                    sims.append(float(embs[i] @ embs[j]))
        return np.mean(sims)
    assert mean_pairwise_same_entity(0.0) > 0.999          # identical keys
    assert mean_pairwise_same_entity(0.5) < 0.9            # paraphrase analog


def test_cosine_rag_suffers_staleness():
    rows = [score_stream(CosineRAG(), make_stream(value_gap=0.4, seed=s))
            for s in range(8)]
    stale = np.mean([r["stale_rate"] for r in rows])
    assert stale > 0.4


def test_keep_latest_oracle_eliminates_staleness():
    rows = [score_stream(KeepLatestOracle(), make_stream(value_gap=0.4, seed=s))
            for s in range(8)]
    assert np.mean([r["stale_rate"] for r in rows]) == 0.0
    assert np.mean([r["current@1"] for r in rows]) > 0.95


def test_reconsolidation_beats_cosine_rag():
    def m(fac):
        rows = [score_stream(fac(), make_stream(value_gap=0.4, seed=s))
                for s in range(8)]
        return np.mean([r["current@1"] for r in rows])
    assert m(lambda: Reconsolidation(tau=0.6)) > m(lambda: CosineRAG()) + 0.3


def test_recency_budget_enforced():
    p = Recency(budget=5); p.reset()
    for i in range(20):
        e = np.zeros(8, np.float32); e[i % 8] = 1.0
        p.observe(e, f"{i}:0", i)
    assert len(p._ids) == 5


# --------------------------------------------------------------------------- #
# Time-aware policies
# --------------------------------------------------------------------------- #


def _toy_store(policy):
    """Three versions of entity a (t=0,1,2) + one distractor b, orthogonal."""
    dim = 8
    a = np.zeros(dim, np.float32); a[0] = 1.0
    b = np.zeros(dim, np.float32); b[1] = 1.0
    for t, iid in [(0, "a:0"), (1, "a:1"), (2, "a:2")]:
        v = a.copy(); v[2 + t] = 0.1; v /= np.linalg.norm(v)
        policy.observe(v, iid, t, text=f"alpha version {t}")
    policy.observe(b, "b:0", 3, text="beta only")
    return a, b


def test_time_rerank_answers_most_recent_of_topk():
    p = TimeRerank(k=3)
    a, _ = _toy_store(p)
    assert p.answer(a) == "a:2"           # cosine alone is ambiguous; time wins
    # candidate SET is unchanged: contamination@3 equals CosineRAG's
    ranked = p.rank(a)
    assert set(ranked[:3]) == {"a:0", "a:1", "a:2"}


def test_recency_weighted_prefers_fresh_on_equal_sim():
    p = RecencyWeighted(lam=0.5, decay=0.9)
    a, _ = _toy_store(p)
    assert p.answer(a) == "a:2"


def test_lexical_key_overwrites_by_gazetteer_match():
    p = LexicalKey(gazetteer=["Marcelo Ebrard", "Marcelo", "Alice Smith"])
    e = np.zeros(4, np.float32); e[0] = 1.0
    p.observe(e, "a:0", 0, text="Alice Smith works for Acme.")
    p.observe(e, "a:1", 1, text="Alice Smith is employed by Beta Corp.")
    assert p.slots == 1 and p._ids == ["a:1"]          # same subject -> overwrite
    # longest-match wins: "Marcelo Ebrard" must not key as "Marcelo"
    p.observe(e, "b:0", 2, text="Marcelo Ebrard holds the position of minister.")
    p.observe(e, "c:0", 3, text="Marcelo plays for Real Madrid.")
    assert p.slots == 3                                 # distinct keys, no merge
    # no gazetteer match -> plain append
    p.observe(e, "d:0", 4, text="Unknown Person attended MIT.")
    assert p.slots == 4


def test_bm25_ranks_lexical_match_first():
    idx = BM25().fit(["the cat sat", "dogs bark loudly", "cat food bowl"])
    s = idx.score("cat")
    assert s[1] == 0.0 and s[0] > 0 and s[2] > 0
    p = BM25RAG()
    _toy_store(p)
    assert p.answer(np.zeros(8, np.float32), "beta") == "b:0"


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def test_metric_decode():
    assert _decode("3:7") == ("3", 7)
    assert _decode("f12:2") == ("f12", 2)


def test_score_ranking_contamination():
    cur = {2}
    r = score_ranking(["a:2", "a:0", "b:1"], "a", cur, ks=(3,))
    assert r["current@1"] == 1.0 and r["contamination@3"] == 1.0
    r = score_ranking(["a:2", "b:1"], "a", cur, ks=(3,))
    assert r["contamination@3"] == 0.0 and r["current@3"] == 1.0
    r = score_ranking(["a:0", "b:1"], "a", cur, ks=(3,))
    assert r["current@1"] == 0.0 and r["stale_rate"] == 1.0
    r = score_ranking(["b:1"], "a", cur, ks=(3,))
    assert r["miss_rate"] == 1.0
    # value-recurrence: multiple current versions all count as current
    r = score_ranking(["a:0"], "a", {0, 2}, ks=(3,))
    assert r["current@1"] == 1.0


def test_score_ranking_set_recall_charges_a_dropped_still_valid_value():
    """current@k is satisfied by any one current value; set recall is not."""
    # single-valued: set recall IS current@k, and the lookup adds nothing else
    vals = {"a:0": "X", "a:1": "Y"}
    base = score_ranking(["a:1", "a:0"], "a", {1}, ks=(3,))
    r = score_ranking(["a:1", "a:0"], "a", {1}, ks=(3,), value_of=vals)
    assert "current_set_recall@3" not in base
    assert {m: r[m] for m in base} == base
    assert r["current_set_recall@3"] == r["current@3"] == 1.0
    r = score_ranking(["a:0", "b:1"], "a", {1}, ks=(3,), value_of=vals)
    assert r["current_set_recall@3"] == r["current@3"] == 0.0

    # two simultaneously-valid values (B, C): one slot per fact scores half
    vals = {"f:0": "A", "f:1": "B", "f:2": "C"}
    cur = {1, 2}
    assert score_ranking(["f:1", "b:0"], "f", cur, ks=(3,),
                         value_of=vals)["current_set_recall@3"] == 0.5
    assert score_ranking(["f:2", "f:1"], "f", cur, ks=(3,),
                         value_of=vals)["current_set_recall@3"] == 1.0
    assert score_ranking(["f:0", "b:0"], "f", cur, ks=(3,),
                         value_of=vals)["current_set_recall@3"] == 0.0
    assert score_ranking([], "f", cur, ks=(3,),
                         value_of=vals)["current_set_recall@3"] == 0.0
    # the window is the window: the second value below the cut is not recalled
    r = score_ranking(["f:1", "b:0", "f:2"], "f", cur, ks=(2, 3), value_of=vals)
    assert r["current_set_recall@2"] == 0.5 and r["current_set_recall@3"] == 1.0

    # A-B-A recurrence: one distinct current value however many versions hold it
    vals = {"f:0": "A", "f:1": "B", "f:2": "A"}
    r = score_ranking(["f:0", "f:2"], "f", {0, 2}, ks=(3,), value_of=vals)
    assert r["current_set_recall@3"] == 1.0 and r["contamination@3"] == 0.0
    r = score_ranking(["f:1"], "f", {0, 2}, ks=(3,), value_of=vals)
    assert r["current_set_recall@3"] == 0.0 and r["stale_rate"] == 1.0

    # a stale version beside one of two current values: contaminated AND half
    vals = {"f:0": "A", "f:1": "B", "f:2": "C"}
    r = score_ranking(["f:1", "f:0"], "f", {1, 2}, ks=(3,), value_of=vals)
    assert r["current@1"] == 1.0 and r["contamination@3"] == 1.0
    assert r["current_set_recall@3"] == 0.5


def test_bootstrap_and_permutation():
    rng = np.random.default_rng(0)
    x = rng.normal(0.5, 0.1, 200)
    ci = bootstrap_ci(x, n_boot=2000, seed=0)
    assert ci["lo"] < ci["mean"] < ci["hi"]
    assert ci["hi"] - ci["lo"] < 0.1
    a = rng.normal(0.6, 0.1, 100); b = a - 0.05
    assert paired_permutation(a, b, n_perm=2000) < 0.01
    assert paired_permutation(a, a) == 1.0
    h = holm({"x": 0.01, "y": 0.04})
    assert h["x"] == 0.02 and h["y"] == 0.04


# --------------------------------------------------------------------------- #
# TempLAMA track
# --------------------------------------------------------------------------- #


needs_data = pytest.mark.skipif(not os.path.exists(_DATA),
                                reason="TempLAMA data not downloaded")


@needs_data
def test_scale_set_recall_denominator_is_the_facts_current_value_set():
    """The scale track derives the value set from ids; it must be the fact's.

    That is what makes single-valued facts get current@k back and charges a
    multi-valued fact once per distinct still-valid value, so a one-slot-per-fact
    ranking scores exactly 1/|current values|.
    """
    from stalebench.templama import build_instance, load_all_changing_facts

    facts, _ = load_all_changing_facts()
    inst = build_instance(facts, paraphrase_p=0.5, seed=0)
    n_multi_current = 0
    for f, (_, key, cur) in zip(facts, inst.queries):
        want = {inst.value_of[f"{key}:{v}"] for v in cur}
        assert want == set(f.current_values)
        assert len(want) == 1 or f.is_multi_valued
        n_multi_current += len(want) > 1
        one_slot = score_ranking([f"{key}:{max(cur)}"], key, cur, (5,),
                                 value_of=inst.value_of)
        assert one_slot["current@5"] == 1.0
        assert one_slot["current_set_recall@5"] == 1.0 / len(want)
    assert n_multi_current > 0


@needs_data
def test_templama_filter_counts_and_split():
    from stalebench.templama import load_facts, split_facts
    facts, stats = load_facts()
    assert stats["fact_keys"] == 5839
    assert stats["single_valued_changing"] == len(facts) == 892
    val, test = split_facts(facts)
    assert len(val) + len(test) == 892
    assert not {f.key for f in val} & {f.key for f in test}
    val2, _ = split_facts(facts)
    assert [f.key for f in val] == [f.key for f in val2]      # deterministic


@needs_data
def test_templama_subjects_and_canonical_statements():
    from stalebench.templama import RELATIONS, load_facts, build_instance
    facts, _ = load_facts()
    assert {f.relation for f in facts} <= set(RELATIONS)
    inst = build_instance(facts[:20], paraphrase_p=0.0, seed=0)
    # canonical (p=0) statement == the TempLAMA cloze with the value filled in
    f0 = facts[0]
    want = RELATIONS[f0.relation]["statements"][0].format(
        s=f0.subject, v=f0.runs[0][1])
    texts = [e[0] for e in inst.events]
    assert want in texts
    # value-agnostic query never contains the current value
    for (q, key, _), f in zip(inst.queries, facts[:20]):
        assert f.current_value not in q


@needs_data
def test_templama_gold_is_value_based():
    from stalebench.templama import load_facts
    facts, _ = load_facts()
    aba = [f for f in facts if len(f.current_runs) > 1]
    assert len(aba) >= 1                     # A,B,A facts exist (7 in v1)
    f = aba[0]
    for i in f.current_runs:
        assert f.runs[i][1] == f.current_value


@needs_data
def test_templama_events_chronological():
    from stalebench.templama import load_facts, build_instance
    facts, _ = load_facts()
    inst = build_instance(facts[:50], paraphrase_p=0.5, seed=0)
    ts = [t for _, _, t in inst.events]
    assert ts == sorted(ts)


@needs_data
def test_query_template_seed_samples_paraphrases():
    from stalebench.templama import load_facts, build_instance
    facts, _ = load_facts()
    base = build_instance(facts[:60], paraphrase_p=0.0, seed=0)
    qp = build_instance(facts[:60], paraphrase_p=0.0, seed=0,
                        query_template_seed=1)
    assert [q[1] for q in base.queries] == [q[1] for q in qp.queries]  # same gold
    changed = sum(a[0] != b[0] for a, b in zip(base.queries, qp.queries))
    assert changed > 0                                   # some non-canonical
    qp2 = build_instance(facts[:60], paraphrase_p=0.0, seed=0,
                         query_template_seed=1)
    assert [q[0] for q in qp.queries] == [q[0] for q in qp2.queries]  # deterministic


def test_reconsolidation_ignores_timestamps():
    """Retention decisions never read t: permuting timestamps changes nothing."""
    from stalebench.env import make_stream
    from stalebench.metrics import score_stream
    st = make_stream(value_gap=0.4, seed=3)
    a = score_stream(Reconsolidation(tau=0.6), st)
    st_sh = make_stream(value_gap=0.4, seed=3)
    ts = [t for *_, t in st_sh.events]
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(ts))
    st_sh.events = [(e, ent, v, float(ts[perm[i]]))
                    for i, (e, ent, v, _) in enumerate(st_sh.events)]
    b = score_stream(Reconsolidation(tau=0.6), st_sh)
    assert a["current@1"] == b["current@1"] and a["stale_rate"] == b["stale_rate"]


# --------------------------------------------------------------------------- #
# Anchor-free track: alias inventory
# --------------------------------------------------------------------------- #


def test_alias_forms_are_plausible_short_references():
    from stalebench.alias import alias_forms
    assert alias_forms("Marcelo Ebrard") == ["Ebrard", "Marcelo", "M. Ebrard"]
    assert alias_forms("Dida") == []                    # single token, no form
    # organizational type words are never a standalone reference
    assert alias_forms("Malmo FF") == ["Malmo"]
    assert "Line" not in alias_forms("Nordland Line")
    assert alias_forms("Municipality of Thessaloniki") == ["Thessaloniki"]
    # initials only when the leading token is itself a usable reference
    assert not any(a.startswith("M. Thess") for a in
                   alias_forms("Municipality of Thessaloniki"))
    # drop-last shortening for longer names, but no leading-token fragments
    la = alias_forms("Los Angeles Rams")
    assert "Rams" in la and "Los Angeles" in la and "Angeles Rams" not in la
    assert all(a != "Marcelo Ebrard" for a in alias_forms("Marcelo Ebrard"))


def test_alias_map_keeps_ambiguity():
    from stalebench.alias import build_alias_map
    m = build_alias_map(["Kamala Harris", "Neil Harris", "Peter Sagan"])
    assert m["Harris"] == {"Kamala Harris", "Neil Harris"}   # collision retained
    assert m["Sagan"] == {"Peter Sagan"}


# --------------------------------------------------------------------------- #
# Keyers and retention mechanisms
# --------------------------------------------------------------------------- #


def _emb(i, dim=8):
    e = np.zeros(dim, np.float32); e[i % dim] = 1.0
    return e


def test_alias_keyer_resolves_short_forms_to_canonical():
    from stalebench.policies import AliasKey
    from stalebench.alias import build_alias_map
    gaz = ["Marcelo Ebrard", "Peter Sagan"]
    p = AliasKey(gaz, build_alias_map(gaz), "skip")
    p.observe(_emb(0), "a:0", 0, text="Marcelo Ebrard serves as Foreign Minister.")
    p.observe(_emb(0), "a:1", 1, text="Ebrard now occupies the office of Economy.")
    assert p.slots == 1 and p._ids == ["a:1"]      # alias linked to full name
    p.observe(_emb(1), "b:0", 2, text="Peter Sagan plays for Bora.")
    assert p.slots == 2


def test_alias_keyer_ambiguity_policies_differ():
    from stalebench.policies import AliasKey
    from stalebench.alias import build_alias_map
    gaz = ["Kamala Harris", "Neil Harris"]
    amap = build_alias_map(gaz)
    skip = AliasKey(gaz, amap, "skip")
    for i, txt in enumerate(["Kamala Harris holds the position of Senator.",
                             "Neil Harris is coached by nobody.",
                             "Harris holds the position of Vice President."]):
        skip.observe(_emb(i), f"x{i}:0", i, text=txt)
    assert skip.slots == 3                           # refuses to link "Harris"
    assert skip.keyer.n_ambiguous_hits == 1

    recent = AliasKey(gaz, amap, "recent")
    for i, txt in enumerate(["Kamala Harris holds the position of Senator.",
                             "Neil Harris is coached by nobody.",
                             "Harris holds the position of Vice President."]):
        recent.observe(_emb(i), f"x{i}:0", i, text=txt)
    assert recent.slots == 2                         # links to most recent owner


def test_keyers_are_deterministic_across_hash_seeds():
    """Inventory order must not inherit the interpreter's string hash seed."""
    from stalebench.policies import SurfaceMatcher
    gaz = ["Ann Lee", "Bob Kim", "Cal Roy", "Dee Fox"]     # all equal length
    a = SurfaceMatcher(gaz)._by_ngram
    b = SurfaceMatcher(list(reversed(gaz)))._by_ngram
    assert a == b and len(a) == 4


def test_surface_matcher_requires_a_whole_mention():
    """Substring hits inside other names must not link (the §5.7 defect)."""
    from stalebench.policies import SurfaceMatcher
    m = SurfaceMatcher(["Richard Blumenthal", "John", "Democratic Party",
                        "Valerie Hackl", "Malmo FF", "M. Ebrard"])
    # a longer capitalized mention is not the shorter inventory entry
    assert m.match("Richard Cresswell plays for Sheffield United.") is None
    assert m.match("Acme employs Valerie Ashby.") is None
    # genuine mentions still match, including at a sentence boundary
    assert m.match("Richard Blumenthal holds the position of Senator.") == \
        "Richard Blumenthal"
    assert m.match("John plays for Ajax.") == "John"
    assert m.match("M. Ebrard serves as minister.") == "M. Ebrard"
    assert m.match("Malmo FF is coached by someone.") == "Malmo FF"
    # word boundaries: no matching inside a longer token
    assert m.match("Johnson plays for Ajax.") is None
    # longest form wins when several match
    m2 = SurfaceMatcher(["Los Angeles", "Los Angeles Rams"])
    assert m2.match("Los Angeles Rams is owned by someone.") == "Los Angeles Rams"


def test_temporal_kg_invalidates_without_deleting():
    from stalebench.policies import OracleKeyer, TemporalKG
    p = TemporalKG(OracleKeyer())
    for v in range(3):
        p.observe(_emb(0), f"a:{v}", v, text=f"alpha v{v}")
    p.observe(_emb(1), "b:0", 3, text="beta")
    assert p.slots == 4 and p.active_slots == 2      # nothing deleted, 2 valid
    ranked = p.rank(_emb(0))
    assert ranked[0] == "a:2"                        # valid edge outranks stale
    assert set(ranked) == {"a:0", "a:1", "a:2", "b:0"}   # stale still recoverable


def test_llm_keyer_uses_model_judgment_and_never_sees_truth():
    from stalebench.policies import LLMKey

    class Stub:
        name = "stub"
        def __init__(self): self.seen = []
        def same_fact(self, new, cands):
            self.seen.append((new, list(cands)))
            return 1 if cands and new.split()[0] == cands[0].split()[0] else 0

    stub = Stub()
    p = LLMKey(stub)
    p.observe(_emb(0), "a:0", 0, text="alpha one")
    p.observe(_emb(0), "a:1", 1, text="alpha two")
    p.observe(_emb(1), "b:0", 2, text="beta one")
    assert p.slots == 2 and "a:1" in p._ids
    flat = " ".join(n + " ".join(c) for n, c in stub.seen)
    assert "a:0" not in flat and "a:1" not in flat    # only text, never ids


def test_retention_mechanisms_share_one_keying_decision():
    """Overwrite and temporal-KG differ only in what they do after keying."""
    from stalebench.policies import KeyedRetention, OracleKeyer, TemporalKG
    ow, kg = KeyedRetention(OracleKeyer()), TemporalKG(OracleKeyer())
    for pol in (ow, kg):
        for v in range(3):
            pol.observe(_emb(0), f"a:{v}", v, text=f"v{v}")
    assert ow.rank(_emb(0))[0] == kg.rank(_emb(0))[0] == "a:2"
    assert ow.slots == 1 and kg.slots == 3 and kg.active_slots == 1


def test_store_eviction_keeps_buffer_and_keys_consistent():
    from stalebench.policies import KeyedRetention, OracleKeyer
    p = KeyedRetention(OracleKeyer(), budget=2)
    for i in range(4):
        p.observe(_emb(i), f"e{i}:0", i, text=f"t{i}")
    assert p.slots == 2 and p._ids == ["e2:0", "e3:0"]
    assert p._emb.shape[0] == 2 and np.allclose(p._emb[0], _emb(2))
    p.observe(_emb(2), "e2:1", 9, text="update")       # still keyed after shift
    assert p.slots == 2 and p._ids == ["e2:1", "e3:0"]


# --------------------------------------------------------------------------- #
# Anchor-free instances
# --------------------------------------------------------------------------- #


@needs_data
def test_alias_track_keeps_first_mention_and_query_anchored():
    from stalebench.templama import load_facts, split_facts, build_instance
    facts, _ = load_facts()
    _, test = split_facts(facts)
    base = build_instance(test, paraphrase_p=0.5, seed=0)
    al = build_instance(test, paraphrase_p=0.5, seed=0, alias_p=1.0)
    assert al.n_alias_events > 0 and base.n_alias_events == 0
    # alias_p=0 must reproduce the v0.2 stream exactly (separate RNG streams)
    assert [e[0] for e in base.events] == [
        e[0] for e in
        build_instance(test, paraphrase_p=0.5, seed=0, alias_p=0.0).events]
    # the introducing statement and every query keep the full subject name
    for (q, _, _), f in zip(al.queries, test):
        assert f.subject in q
    for fi, f in enumerate(test):
        assert f.subject in al.text_of[f"f{fi}:0"]


@needs_data
def test_distractors_are_stored_but_never_queried():
    from stalebench.templama import (load_facts, load_distractor_facts,
                                     split_facts, build_instance)
    facts, _ = load_facts()
    _, test = split_facts(facts)
    _, dist = split_facts(load_distractor_facts())
    inst = build_instance(test, paraphrase_p=0.5, seed=0, distractors=dist)
    assert inst.n_distractor_events > 3000
    assert len(inst.queries) == len(test)             # queries unchanged
    ids = {iid for _, iid, _ in inst.events}
    assert any(i.startswith("d") for i in ids)
    qkeys = {k for _, k, _ in inst.queries}
    assert not any(k.startswith("d") for k in qkeys)
    ts = [t for _, _, t in inst.events]
    assert ts == sorted(ts)


# --------------------------------------------------------------------------- #
# Reader scoring
# --------------------------------------------------------------------------- #


def test_reader_answer_decomposition():
    from stalebench.reader import score_answer
    r = score_answer("Yamaha Motor Racing", "Yamaha Motor Racing", ["Ducati"])
    assert r["correct"] == 1.0 and r["stale"] == 0.0
    r = score_answer("Ducati", "Yamaha Motor Racing", ["Ducati"])
    assert r["stale"] == 1.0 and r["correct"] == 0.0
    r = score_answer("Yamaha Motor Racing (formerly Ducati)",
                     "Yamaha Motor Racing", ["Ducati"])
    assert r["hedged"] == 1.0 and r["correct"] == 0.0
    r = score_answer("Yamaha", "Yamaha Motor Racing", ["Ducati"])
    assert r["correct"] == 1.0                      # bare short form counts
    r = score_answer("Real", "Real Madrid CF", ["Ducati"])
    assert r["correct"] == 1.0                      # whole-token prefix
    r = score_answer("Yamah", "Yamaha Motor Racing", ["Ducati"])
    assert r["other"] == 1.0                        # partial token does not
    r = score_answer("Honda", "Yamaha Motor Racing", ["Ducati"])
    assert r["other"] == 1.0
    r = score_answer("The memories do not specify.", "Yamaha", ["Ducati"])
    assert r["abstain"] == 1.0 and r["other"] == 1.0


def test_reader_prompt_shapes():
    from stalebench.reader import build_prompt
    p = build_prompt(["a plays for X.", "a plays for Y."], "Who?", years=[2011, 2014])
    assert "1. [2011] a plays for X." in p and "2. [2014] a plays for Y." in p
    assert "[2011]" not in build_prompt(["a plays for X."], "Who?")
    assert "Memories" not in build_prompt([], "Who?")     # closed-book control


# --------------------------------------------------------------------------- #
# LLM client plumbing (no network)
# --------------------------------------------------------------------------- #


def test_llm_client_caches_and_parses_same_fact(tmp_path):
    from stalebench.llm import _BaseClient

    class Fake(_BaseClient):
        model = "fake-1"
        def __init__(self, **kw):
            self.replies = []
            super().__init__(**kw)
        def _complete(self, prompt, max_tokens):
            self.replies.append(prompt)
            return " 2. ", 0.001

    c = Fake(cache_dir=str(tmp_path))
    assert c.same_fact("new stmt", ["a", "b", "c"]) == 2      # parses "2."
    assert c.n_calls == 1 and c.usd == 0.001
    assert c.same_fact("new stmt", ["a", "b", "c"]) == 2      # served from disk
    assert c.n_calls == 1 and c.n_cached == 1
    assert c.same_fact("other", []) == 0                      # no candidates
    prompt = c.replies[0]
    assert "new stmt" in prompt and "1. a" in prompt

    c2 = Fake(cache_dir=str(tmp_path))                        # cache persists
    assert c2.same_fact("new stmt", ["a", "b", "c"]) == 2
    assert c2.n_calls == 0 and c2.n_cached == 1


def test_llm_client_does_not_retry_auth_failures(tmp_path):
    import urllib.error
    from stalebench.llm import _BaseClient

    class Failing(_BaseClient):
        model = "fake-2"
        def __init__(self, **kw):
            self.attempts = 0
            super().__init__(**kw)
        def _open(self, *a, **k):
            self.attempts += 1
            raise urllib.error.HTTPError("u", 402, "Payment Required", None, None)

    c = Failing(cache_dir=str(tmp_path))
    import urllib.request
    orig = urllib.request.urlopen
    urllib.request.urlopen = c._open
    try:
        with pytest.raises(RuntimeError, match="402"):
            c._post("http://x", {}, {})
    finally:
        urllib.request.urlopen = orig
    assert c.attempts == 1            # a spent key must fail fast, not 12 times


# --------------------------------------------------------------------------- #
# Release package
# --------------------------------------------------------------------------- #


@needs_data
def test_release_round_trip_reproduces_headline(tmp_path):
    """The released files + frozen evaluator must reproduce the paper's table."""
    import json
    import subprocess
    import sys
    from stalebench.embed import Embedder, MINILM
    from stalebench.release import export

    out = str(tmp_path / "rel")
    m = export(out)
    from stalebench.release import CONFIGS
    # the shipped configs must be exactly what CONFIGS declares -- a config
    # present on disk but absent from the manifest (or the HF card) breaks
    # load_dataset for it, which is how full_corpus_anchor_free was unusable
    assert set(m["configs"]) == set(CONFIGS)
    assert {"scale", "scale_anchor_free"} <= set(m["configs"])
    assert (m["configs"]["scale"]["files"]["queries_test"]["rows"]
            > 5 * m["configs"]["anchored"]["files"]["queries_test"]["rows"])
    assert m["configs"]["anchored"]["files"]["queries_test"]["rows"] == 468
    assert (m["configs"]["full_corpus"]["files"]["statements_test"]["rows"]
            > m["configs"]["anchored"]["files"]["statements_test"]["rows"])

    st = [json.loads(l) for l in open(f"{out}/statements_anchored_test.jsonl")]
    qs = [json.loads(l) for l in open(f"{out}/queries_anchored_test.jsonl")]
    key = [json.loads(l) for l in open(f"{out}/key_anchored_test.jsonl")]

    # the public files must not carry the label in any form
    public = set(st[0]) | set(qs[0])
    assert not (public & {"fact_key", "fact_key_id", "version", "value",
                          "is_current", "subject", "current_versions",
                          "gold_value", "n_versions"})
    assert all(s["id"].startswith("s") and ":" not in s["id"] for s in st)
    assert all(k["label_source"] == "wikidata_validity_year" for k in key)

    cache = os.path.join(os.path.dirname(__file__), "..", "..", "results",
                         "emb_cache")
    emb = Embedder(MINILM, cache_dir=cache)
    st_sorted = sorted(st, key=lambda s: s["t"])
    ev = emb.encode([s["text"] for s in st_sorted], "release_ev_anchored_test")
    qe = emb.encode([q["question"] for q in qs], "test_q_p0.5")

    p = CosineRAG()
    for s, e in zip(st_sorted, ev):
        p.observe(e, s["id"], s["t"], s["text"])
    sub = tmp_path / "sub.jsonl"
    with open(sub, "w") as fh:
        for q, x in zip(qs, qe):
            fh.write(json.dumps({"query_id": q["query_id"],
                                 "ranking": p.rank(x, q["question"])[:20]}) + "\n")
    r = subprocess.run([sys.executable, f"{out}/evaluate.py",
                        "--key", f"{out}/key_anchored_test.jsonl",
                        "--submission", str(sub)],
                       capture_output=True, text=True, check=True)
    got = json.loads(r.stdout)
    assert got["n_queries"] == 468
    assert abs(got["current@1"] - 0.464) < 0.01      # the paper's headline row


@needs_data
def test_release_ids_do_not_leak_the_label(tmp_path):
    """A system that reads only the public files cannot recover the answer.

    v0.2 shipped ids of the form ``f171:0``; grouping by that prefix and taking
    the latest scored current@1 = 1.000, above the paper's own oracle.
    """
    import json
    import subprocess
    import sys
    from stalebench.release import export

    out = str(tmp_path / "rel")
    export(out)
    st = [json.loads(l) for l in open(f"{out}/statements_anchored_test.jsonl")]
    qs = [json.loads(l) for l in open(f"{out}/queries_anchored_test.jsonl")]

    by_t = [s["id"] for s in sorted(st, key=lambda x: -x["t"])][:20]
    sub = tmp_path / "attack.jsonl"
    with open(sub, "w") as fh:
        for q in qs:
            fh.write(json.dumps({"query_id": q["query_id"],
                                 "ranking": by_t}) + "\n")
    r = subprocess.run([sys.executable, f"{out}/evaluate.py",
                        "--key", f"{out}/key_anchored_test.jsonl",
                        "--submission", str(sub)],
                       capture_output=True, text=True, check=True)
    assert json.loads(r.stdout)["current@1"] < 0.05


def test_release_evaluator_rejects_incomplete_submission(tmp_path):
    import json
    import subprocess
    import sys
    from stalebench.release import _EVALUATOR
    ev = tmp_path / "evaluate.py"; ev.write_text(_EVALUATOR)
    q = tmp_path / "k.jsonl"
    q.write_text("\n".join(json.dumps(
        {"query_id": f"q{i}", "fact_ids": [f"s{i}a"], "current_ids": [f"s{i}a"]})
        for i in range(3)))
    s = tmp_path / "s.jsonl"
    s.write_text(json.dumps({"query_id": "q0", "ranking": ["s0a"]}))
    r = subprocess.run([sys.executable, str(ev), "--key", str(q),
                        "--submission", str(s)], capture_output=True, text=True)
    assert r.returncode != 0 and "missing 2 queries" in r.stderr


def test_llm_cache_ships_and_round_trips(tmp_path):
    """Shipped judgments must expand back to the exact keys the client reads."""
    import json
    from stalebench.llm import _BaseClient
    from stalebench.release import export_llm_cache, import_llm_cache

    src, dst, rel = tmp_path / "src", tmp_path / "dst", tmp_path / "rel"
    src.mkdir(); rel.mkdir()

    class Fake(_BaseClient):
        model = "fake-3"
        def _complete(self, prompt, max_tokens):
            return f"answer to {prompt}", 0.0

    a = Fake(cache_dir=str(src))
    for i in range(3):
        a.generate(f"prompt {i}", max_tokens=16)
    assert a.n_calls == 3

    meta = export_llm_cache(str(rel), cache_dir=str(src))
    assert meta["entries"] == 3 and meta["by_model"] == {"fake-3": 3}
    n = import_llm_cache(str(rel / meta["path"]), cache_dir=str(dst))
    assert n == 3

    b = Fake(cache_dir=str(dst))
    for i in range(3):
        assert b.generate(f"prompt {i}", max_tokens=16) == f"answer to prompt {i}"
    assert b.n_calls == 0 and b.n_cached == 3     # served entirely from the ship


def test_temporal_kg_sheds_tombstones_first_under_budget():
    """Invalidated edges are the footprint a budgeted store gives up first."""
    from stalebench.policies import OracleKeyer, TemporalKG
    p = TemporalKG(OracleKeyer(), budget=3)
    for v in range(3):
        p.observe(_emb(0), f"a:{v}", v, text=f"alpha v{v}")   # 2 tombstones
    p.observe(_emb(1), "b:0", 3, text="beta")
    assert p.slots == 3 and p.active_slots == 2
    assert "a:2" in p._ids and "b:0" in p._ids       # valid edges survive
    assert "a:0" not in p._ids                        # oldest tombstone shed
    p.observe(_emb(2), "c:0", 4, text="gamma")
    assert p.slots == 3 and p.active_slots == 3       # all tombstones gone
    assert set(p._ids) == {"a:2", "b:0", "c:0"}
    assert p.rank(_emb(0))[0] == "a:2"                 # still keyed correctly


def test_fuzzy_keyer_links_short_forms_without_an_alias_table():
    """The inventory-free rung: canonical names only, no alias inventory."""
    from stalebench.policies import FuzzyKeyer
    gaz = ["Marcelo Ebrard", "Peter Sagan", "Malmo FF", "Richard Blumenthal"]
    k = FuzzyKeyer(gaz, tau=0.34)
    assert k.key_of(None, None, None,
                    "Ebrard now occupies the office of Economy.") == "Marcelo Ebrard"
    assert k.key_of(None, None, None,
                    "M. Ebrard serves as minister.") == "Marcelo Ebrard"
    assert k.key_of(None, None, None,
                    "Sagan is on the roster of Bora.") == "Peter Sagan"
    assert k.key_of(None, None, None, "Nothing relevant here.") is None
    # a cached "no link" must not be confused with a memo miss
    before = k.n_unlinked
    assert k.key_of(None, None, None, "Nothing relevant here.") is None
    assert k.n_unlinked == before + 1
    # memoization must not change the answer
    k2 = FuzzyKeyer(gaz, tau=0.34)
    for t in ["Ebrard now occupies the office of Economy.", "Nothing relevant here."]:
        assert k2.key_of(None, None, None, t) == k.key_of(None, None, None, t)


@needs_data
def test_context_order_ablation_reverses_the_same_topk():
    """The reversed-order arm must reorder, not re-retrieve."""
    from stalebench.embed import Embedder, MINILM
    from stalebench.run_reader import _policy_contexts
    from stalebench.templama import build_instance, load_facts, split_facts

    facts, _ = load_facts()
    _, test = split_facts(facts)
    test = test[:40]
    inst = build_instance(test, paraphrase_p=0.5, seed=0)
    cache = os.path.join(os.path.dirname(__file__), "..", "..", "results",
                         "emb_cache")
    emb = Embedder(MINILM, cache_dir=cache)
    ev = emb.encode([e[0] for e in inst.events], "order_ablation_ev")
    qe = emb.encode([q[0] for q in inst.queries], "order_ablation_q")

    fwd_p, fwd_r, fwd_t = _policy_contexts(lambda: CosineRAG(), inst, ev, qe,
                                           False, reverse=False)
    rev_p, rev_r, rev_t = _policy_contexts(lambda: CosineRAG(), inst, ev, qe,
                                           False, reverse=True)
    # identical retrieval, identical candidate sets, reversed presentation
    assert [r["current@1"] for r in fwd_r] == [r["current@1"] for r in rev_r]
    assert all(set(a) == set(b) for a, b in zip(fwd_t, rev_t))
    assert any(p != q for p, q in zip(fwd_p, rev_p))
    first_fwd = fwd_p[0].split("\n")[1]
    last_rev = [l for l in rev_p[0].split("\n") if l.strip()][5]
    assert first_fwd[3:] in last_rev            # top hit moved to the end


def test_fuzzy_keyer_is_hash_seed_independent():
    """Set/dict iteration must not leak the interpreter's string hash seed.

    This defect was recorded in the appendix for the gazetteer keyer and then
    reintroduced in FuzzyKeyer, which iterated a set of trigrams and broke
    similarity ties by arrival order. The test shuffles the inventory and the
    candidate order so a hash-order dependence cannot hide.
    """
    from stalebench.policies import FuzzyKeyer
    gaz = ["Marcelo Ebrard", "Marcela Ebrardo", "Peter Sagan", "Petra Sagana",
           "Malmo FF", "Malmo FC"]
    texts = ["Ebrard now occupies the office of Economy.",
             "Sagan is on the roster of Bora.",
             "Malmo is coached by someone.",
             "Nothing relevant appears here."]
    ref = [FuzzyKeyer(gaz, 0.2).key_of(None, None, None, t) for t in texts]
    for perm in ([2, 0, 5, 1, 4, 3], [5, 4, 3, 2, 1, 0], [1, 3, 0, 4, 2, 5]):
        k = FuzzyKeyer([gaz[i] for i in perm], 0.2)
        assert [k.key_of(None, None, None, t) for t in texts] == ref


def test_fuzzy_keyer_memo_key_distinguishes_inventories():
    """Two inventories of equal size and extremes must not share a cache."""
    from stalebench.policies import FuzzyKeyer
    a = FuzzyKeyer(["Aaron Zeta", "Michael Brown", "Zoe Aaronson"], 0.2)
    b = FuzzyKeyer(["Aaron Zeta", "Michelle Green", "Zoe Aaronson"], 0.2)
    assert a._memo is not b._memo


def test_read_path_keying_resolves_the_query_not_just_the_stream():
    """Symmetric keying must rank by the query's resolved key, not by cosine."""
    from stalebench.policies import ReadWriteFuzzyKey
    gaz = ["Marcelo Ebrard", "Peter Sagan"]

    def unit(i):
        v = np.eye(4, dtype=np.float32)[i % 4] + 0.01
        return v / np.linalg.norm(v)

    p = ReadWriteFuzzyKey(gaz, 0.20)
    p.observe(unit(0), "a:0", 0, "Marcelo Ebrard holds the position of Minister.")
    p.observe(unit(1), "a:1", 1, "Ebrard now occupies the office of Economy.")
    p.observe(unit(2), "b:0", 2, "Peter Sagan plays for Bora.")
    assert p.slots == 2                       # write-path keying merged a:0/a:1
    # the query embedding is orthogonal to everything, so cosine cannot help
    got = p.rank(unit(3), "What position does Marcelo Ebrard currently hold?")
    assert got[0] == "a:1"
    # an unresolvable query falls back to similarity rather than returning junk
    assert p.rank(unit(3), "Who knows?")[0] in {"a:1", "b:0"}

    # store-all variant keys only on read: nothing is merged, but the freshest
    # statement of the queried fact still ranks first
    q = ReadWriteFuzzyKey(gaz, 0.20, write_keys=False)
    q.observe(unit(0), "a:0", 0, "Marcelo Ebrard holds the position of Minister.")
    q.observe(unit(1), "a:1", 1, "Ebrard now occupies the office of Economy.")
    assert q.slots == 2
    assert q.rank(unit(3), "What position does Marcelo Ebrard currently hold?")[0] == "a:1"


@pytest.mark.skipif(not os.environ.get("STALEBENCH_TEMPLAMA_OK", "1") == "1",
                    reason="needs the TempLAMA cache")
def test_controlled_pair_shares_rank_one_on_the_reader_configuration():
    """The reader track's headline rests on rank 1 being *identical*.

    ``fuzzy-read-key`` and ``fuzzy-rw-key`` share a read path, so on the
    anchored configuration the reader track uses they must return the same
    statement first for every query --- that is what makes the 65-point reader
    gap attributable to ranks 2..k rather than to a better first hit. It is a
    property of the code, not an observation about one run, so it is pinned
    here. (Anchor-free it is 467/468: an unresolved query falls back to
    similarity over two differently sized stores, and the paper scopes the
    claim to the configuration where it holds exactly.)
    """
    pytest.importorskip("sentence_transformers")
    from stalebench.embed import MINILM, Embedder
    from stalebench.policies import ReadWriteFuzzyKey
    from stalebench.templama import build_instance, load_facts, split_facts

    facts, _ = load_facts()
    _, test = split_facts(facts)
    inst = build_instance(test, paraphrase_p=0.5, seed=0)     # alias_p=0.0
    emb = Embedder(MINILM)
    ev = emb.encode([e[0] for e in inst.events], "pt_p0.5_a0.0")
    qe = emb.encode([q[0] for q in inst.queries], "ptq_p0.5_a0.0")
    gaz = [f.subject for f in test]

    tops = {}
    for name, write_keys in (("read", False), ("rw", True)):
        pol = ReadWriteFuzzyKey(gaz, 0.2, write_keys=write_keys)
        pol.reset()
        for (text, iid, t), e in zip(inst.events, ev):
            pol.observe(e, iid, t, text)
        tops[name] = [pol.rank(q, qt)[0]
                      for (qt, _, _), q in zip(inst.queries, qe)]

    assert tops["read"] == tops["rw"], (
        "the controlled pair no longer shares rank 1; the reader-track claim "
        "that its accuracy gap is caused by ranks 2..k is void")


def test_release_readme_advertises_every_shipped_config():
    """The HF front matter is the load path; omitting a config breaks it.

    ``full_corpus_anchor_free`` shipped as files, was named in the card's own
    table as the hardest configuration, and was missing from the ``configs:``
    front matter -- so ``load_dataset(..., "full_corpus_anchor_free")`` failed.
    The block is generated from CONFIGS now; this pins that it stays generated.
    """
    import re
    from stalebench.release import CONFIGS
    path = os.path.join(os.path.dirname(__file__), "..", "..", "release",
                        "README.md")
    if not os.path.exists(path):
        pytest.skip("release not built")
    front = open(path).read().split("---")[1]
    advertised = set(re.findall(r"config_name:\s*(\S+)", front))
    assert advertised == set(CONFIGS), (
        f"HF card advertises {advertised}, release ships {set(CONFIGS)}")


def test_wrapped_system_metrics_are_withheld_relative_to_the_baseline():
    """A wrapped system's miss rate is judged against store-all, not a constant.

    An absolute cutoff of 0.9 passed a run whose miss rate was 0.527 against a
    store-all baseline of 0.007 on the identical stream -- 75x -- and that
    broken number reached the paper. The comparison that catches it is
    relative: a wrapped system may legitimately miss more than plain cosine,
    but not by an order of magnitude, because that means its retrieval layer is
    not returning what its store holds.
    """
    def usable(miss, ref_miss):
        return miss <= max(0.30, 10 * ref_miss)

    # the two runs that were wrongly published, against their own baselines
    assert not usable(0.527, 0.0067), "the 2.0.11 faiss run must be withheld"
    assert not usable(0.980, 0.0067), "the 0.1.118 faiss run must be withheld"
    # a system that merely retrieves somewhat worse than cosine is still usable
    assert usable(0.05, 0.0067)
    assert usable(0.20, 0.05)
    # and the floor keeps a near-zero baseline from making the gate absurd
    assert usable(0.29, 0.0)
    assert not usable(0.31, 0.0)


def test_text_scoring_catches_the_merge_that_id_attribution_cannot():
    """The failure mode the id proxy is blind to, made explicit.

    A system that fuses two versions of a fact into one memory is attributed to
    whichever statement wrote it last, so id-level scoring calls the context
    current and uncontaminated while the text a reader is handed names both
    values. That is the disclosed weakness of the systems section; this is the
    measurement that bounds it.
    """
    from stalebench.metrics import score_texts

    gold, stale = "TotalEnergies", ["Bora-Hansgrohe"]

    # id-level: one memory, written last by the current statement -> perfect
    assert score_ranking(["f0:1"], "f0", {1}) == {
        "current@1": 1.0, "stale_rate": 0.0, "miss_rate": 0.0,
        "current@5": 1.0, "contamination@5": 0.0}
    # text-level on the merged text the system actually surfaced
    merged = score_texts(
        ["Peter Sagan rides for TotalEnergies (formerly Bora-Hansgrohe)."],
        gold, stale)
    assert merged["hedged@1_text"] == 1.0
    assert merged["current@1_text"] == 0.0
    assert merged["contamination@5_text"] == 1.0

    # a clean supersession scores the same either way
    clean = score_texts(["Peter Sagan rides for TotalEnergies."], gold, stale)
    assert clean["current@1_text"] == 1.0 and clean["hedged@1_text"] == 0.0
    assert clean["contamination@5_text"] == 0.0

    # and the stale/miss halves of the decomposition still separate
    assert score_texts(["Peter Sagan rides for Bora-Hansgrohe."],
                       gold, stale)["stale_rate_text"] == 1.0
    assert score_texts(["Peter Sagan is a cyclist."],
                       gold, stale)["miss_rate_text"] == 1.0
    assert score_texts([], gold, stale)["miss_rate_text"] == 1.0
    # contamination is a property of the context, not of any one memory
    split = score_texts(["Peter Sagan rides for TotalEnergies.",
                         "Peter Sagan rides for Bora-Hansgrohe."], gold, stale)
    assert split["current@1_text"] == 1.0
    assert split["contamination@5_text"] == 1.0


@needs_data
def test_text_and_id_scoring_agree_when_nothing_rewrites_the_corpus():
    """The text scorer is calibrated against the id scorer where both are valid.

    A policy that returns the corpus verbatim cannot be scored differently by
    the two conventions except through value strings that collide across facts.
    Requiring near-exact agreement here is what licenses reading a *divergence*
    on a wrapped system as evidence about that system rather than about the
    scorer.
    """
    from stalebench.embed import Embedder, MINILM
    from stalebench.metrics import score_texts
    from stalebench.reader import stale_values_of
    from stalebench.run_systems import _hits
    from stalebench.templama import build_instance, load_facts, split_facts

    facts, _ = load_facts()
    _, test = split_facts(facts)
    test = test[:40]
    inst = build_instance(test, paraphrase_p=0.5, seed=0)
    cache = os.path.join(os.path.dirname(__file__), "..", "..", "results",
                         "emb_cache")
    emb = Embedder(MINILM, cache_dir=cache)
    ev = emb.encode([e[0] for e in inst.events], "order_ablation_ev")
    qe = emb.encode([q[0] for q in inst.queries], "order_ablation_q")
    stale = [stale_values_of(f.runs, f.current_value) for f in test]

    for fac in (CosineRAG, lambda: TimeRerank(k=2), KeepLatestOracle):
        pol = fac()
        pol.reset()
        for (text, iid, t), e in zip(inst.events, ev):
            pol.observe(e, iid, t, text)
        agree = {"current@1": 0, "stale_rate": 0, "contamination@5": 0}
        for i, ((qtext, key, cur), q) in enumerate(zip(inst.queries, qe)):
            hits = _hits(pol, inst, q, qtext, 20)
            a = score_ranking([h["sid"] for h in hits], key, cur)
            b = score_texts([h["text"] for h in hits], inst.gold[i], stale[i])
            for m in agree:
                agree[m] += int(a[m] == b[f"{m}_text"])
        for m, n in agree.items():
            assert n >= 0.95 * len(inst.queries), f"{pol.name} {m}: {n}"


def test_post_hoc_read_keying_promotes_then_merges():
    """The two halves of the read-path fix, applied to a store's output.

    Resolving the query promotes every version of the queried fact, so the
    ranking metric improves while the context fills with the fact contradicting
    itself; collapsing to one memory per entity is the read-side image of
    merging on write. Keeping them separate is the point -- it is what lets a
    deployed system be given the paper's decomposition without changing its
    write path.
    """
    from stalebench.policies import FuzzyKeyer
    from stalebench.run_sysreader import collapse, read_key_rerank

    keyer = FuzzyKeyer(["Peter Sagan", "Valentino Rossi"], 0.2)
    hits = [{"sid": "f1:0", "text": "Valentino Rossi rides for Yamaha.",
             "updated_at": "2020-01-01"},
            {"sid": "f0:0", "text": "Peter Sagan rides for Bora-Hansgrohe.",
             "updated_at": "2018-01-01"},
            {"sid": "f0:1", "text": "Peter Sagan rides for TotalEnergies.",
             "updated_at": "2022-01-01"}]
    q = "Which team does Peter Sagan currently ride for?"

    rk = read_key_rerank(hits, q, keyer)
    assert [h["sid"] for h in rk] == ["f0:1", "f0:0", "f1:0"], "freshest first"
    assert len(rk) == len(hits), "re-ranking must not drop anything"

    col = collapse(rk, keyer)
    assert [h["sid"] for h in col] == ["f0:1", "f1:0"], "one per entity, freshest"

    # a query that does not resolve leaves the store's own order alone
    assert read_key_rerank(hits, "What happened?", keyer) == hits

    # a memory that does not resolve is never merged away: silently dropping
    # what the keyer cannot read would flatter the arm
    noisy = list(hits) + [{"sid": "d0:0", "text": "the weather turned."},
                          {"sid": "d1:0", "text": "prices rose again."}]
    assert len(collapse(noisy, keyer)) == 4


@needs_data
def test_post_hoc_read_keying_recovers_the_directly_measured_rung():
    """Calibrate the post-hoc arm against the rung the paper measures live.

    The arm re-ranks contexts a store already produced; the rung re-ranks a
    store we control. If the two disagree on the same stream, the arm is
    measuring its own implementation rather than the paper's fix, and the
    system-side number built on it means nothing.
    """
    from stalebench.embed import Embedder, MINILM
    from stalebench.policies import FuzzyKeyer, ReadWriteFuzzyKey
    from stalebench.run_systems import _hits
    from stalebench.run_sysreader import read_key_rerank
    from stalebench.templama import build_instance, load_facts, split_facts

    facts, _ = load_facts()
    _, test = split_facts(facts)
    test = test[:40]
    gaz = [f.subject for f in test]
    inst = build_instance(test, paraphrase_p=0.5, seed=0, alias_p=1.0)
    cache = os.path.join(os.path.dirname(__file__), "..", "..", "results",
                         "emb_cache")
    emb = Embedder(MINILM, cache_dir=cache)
    ev = emb.encode([e[0] for e in inst.events], "sysreader_cal_ev")
    qe = emb.encode([q[0] for q in inst.queries], "sysreader_cal_q")

    store = CosineRAG()
    store.reset()
    live = ReadWriteFuzzyKey(gaz, 0.2, write_keys=False)
    live.reset()
    for (text, iid, t), e in zip(inst.events, ev):
        store.observe(e, iid, t, text)
        live.observe(e, iid, t, text)

    keyer = FuzzyKeyer(gaz, 0.2)
    agree = 0
    for (qtext, key, cur), q in zip(inst.queries, qe):
        post = read_key_rerank(_hits(store, inst, q, qtext, 20), qtext, keyer)
        a = score_ranking([h["sid"] for h in post], key, cur)
        b = score_ranking(live.rank(q, qtext), key, cur)
        agree += int(a["current@1"] == b["current@1"])
    assert agree >= 0.95 * len(inst.queries), f"{agree}/{len(inst.queries)}"


def test_mem0_wrapper_records_the_backend_it_used():
    """Comparing two releases requires holding the backend fixed and saying so.

    The FAISS backend returned the least-similar memories on both release
    lines, which is a fact about the backend rather than either system, so a
    result from this wrapper is uninterpretable without it.
    """
    pytest.importorskip("mem0")
    from stalebench.systems import Mem0Policy
    assert Mem0Policy().vector_store == "faiss"
    assert Mem0Policy(vector_store="qdrant").vector_store == "qdrant"
    old = os.environ.get("STALEBENCH_MEM0_STORE")
    os.environ["STALEBENCH_MEM0_STORE"] = "chroma"
    try:
        assert Mem0Policy().vector_store == "chroma"
    finally:
        if old is None:
            del os.environ["STALEBENCH_MEM0_STORE"]
        else:
            os.environ["STALEBENCH_MEM0_STORE"] = old


def test_graphiti_wrapper_dates_a_statement_by_its_benchmark_year():
    """The stream's float timestamp must round-trip to the year it encodes.

    ``build_instance`` packs the year and the stream position into one float;
    Graphiti dates episodes, chooses an extractor's context by that date and
    copies it into each fact's ``valid_at``, which is what its invalidation
    compares. A wrong year here would silently corrupt the mechanism under
    test rather than raise.
    """
    from stalebench.systems import statement_time
    from stalebench.templama import _YEAR0

    for year in (2010, 2013, 2020):
        for seq in (0, 1, 329, 2999):
            when = statement_time(float(year - _YEAR0) + 1e-4 * seq)
            assert (when.year, when.month) == (year, 1)
            jan1 = when.replace(month=1, day=1, hour=0, minute=0)
            assert (when - jan1).total_seconds() == 60 * seq
    # later in the stream is later in time and never a tie, so "the episodes
    # before this one" is exactly the stream so far
    ts = [statement_time(float(y) + 1e-4 * s) for y in range(3)
          for s in range(0, 300, 7)]
    assert ts == sorted(ts) and len(set(ts)) == len(ts)


@needs_data
def test_graphiti_reference_time_matches_build_instance():
    from stalebench.systems import statement_time
    from stalebench.templama import build_instance, load_facts, split_facts

    facts, _ = load_facts()
    _, test = split_facts(facts)
    inst = build_instance(test[:40], paraphrase_p=0.5, seed=0)
    for _, iid, t in inst.events:
        assert statement_time(t).year == inst.year_of[iid]


def test_run_systems_routes_a_graphiti_spec_beside_mem0():
    """A second system joins the runner without touching the first's path."""
    from stalebench.run_systems import _policy_for
    from stalebench.systems import GraphitiPolicy, Mem0Policy

    pol = _policy_for("graphiti:ollama:qwen2.5:14b")
    assert isinstance(pol, GraphitiPolicy)
    assert pol.name == "graphiti(qwen2.5:14b)" and pol.llm_model == "qwen2.5:14b"
    assert pol.graph_backend == "falkordb"
    m = _policy_for("ollama:qwen2.5:14b")
    assert isinstance(m, Mem0Policy) and m.name == "mem0(qwen2.5:14b)"
    # a hosted adjudicator is a second LLM backend for the same system, and
    # the third-person extraction prompt is a labelled variant of Mem0
    g = _policy_for("graphiti:zai:glm-4.6")
    assert isinstance(g, GraphitiPolicy) and g.llm_backend == "zai"
    assert g.name == "graphiti(glm-4.6)" and g.hosted_base_url
    t = _policy_for("zai:glm-4.6@third-person")
    assert isinstance(t, Mem0Policy) and t.extraction_prompt_label == "third-person"
    assert t.name == "mem0(glm-4.6|third-person)"
    with pytest.raises(ValueError):
        _policy_for("graphiti:neo4j:whatever")
    with pytest.raises(ValueError):
        _policy_for("ollama:qwen2.5:14b@first-person")
    with pytest.raises(ValueError):
        _policy_for("neo4j:whatever")
