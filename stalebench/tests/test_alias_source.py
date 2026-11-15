"""Tests for the anchor-free track's alias *source* (v1.2): natural short forms
from a Wikidata/Wikipedia inventory beside the mechanical ones, with the
default left byte-identical to the v1.1 stream."""

import json
import os

import numpy as np
import pytest

from stalebench.alias import alias_forms, alias_map_from_pools
from stalebench.policies import AliasKey
from stalebench.templama import (Fact, alias_pool, alias_pools, build_instance,
                                 load_natural_inventory)

_DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data",
                     "templama_val.json")
needs_data = pytest.mark.skipif(not os.path.exists(_DATA),
                                reason="TempLAMA data not downloaded")


def _facts():
    a = [(2010, "Secretary of Foreign Affairs"), (2013, "Mayor"),
         (2016, "Senator")]
    b = [(2010, "Liquigas"), (2014, "Tinkoff"), (2018, "Bora")]
    return [Fact("Q1_P39", "P39", "Marcelo Ebrard", dict(a), a),
            Fact("Q2_P54", "P54", "Peter Sagan", dict(b), b)]


def _inventory(tmp_path):
    # a full-name form (case-insensitive) and a blank must be dropped; Sagan
    # has no natural form at all
    inv = {"Q1": {"subject": "Marcelo Ebrard",
                  "natural": ["Ebrard", "marcelo ebrard", " ", "El Canciller"]},
           "Q2": {"subject": "Peter Sagan", "natural": []}}
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(inv))
    return str(path)


def _surface(inst, iid):
    """Subject surface form of a canonical-template statement."""
    text = inst.text_of[iid]
    for marker in (" holds the position of ", " plays for "):
        if marker in text:
            return text.split(marker)[0]
    raise AssertionError(text)


@needs_data
def test_default_source_is_byte_identical_to_the_mechanical_stream():
    from stalebench.templama import load_facts, split_facts
    facts, _ = load_facts()
    _, test = split_facts(facts)
    for a in (0.5, 1.0):
        base = build_instance(test, paraphrase_p=0.5, seed=0, alias_p=a)
        expl = build_instance(test, paraphrase_p=0.5, seed=0, alias_p=a,
                              alias_source="mechanical")
        assert base.events == expl.events and base.queries == expl.queries
        assert base.n_alias_events == expl.n_alias_events
    # the v1.1 stream as recorded in results/stalebench_alias_results.json
    assert base.n_alias_events == 472
    assert base.alias_coverage["n_alias_eligible"] == 527
    assert base.alias_coverage["inventory_path"] is None


def test_natural_source_draws_only_inventory_forms(tmp_path):
    inv = _inventory(tmp_path)
    inst = build_instance(_facts(), seed=0, alias_p=1.0, alias_source="natural",
                          alias_inventory=inv)
    assert _surface(inst, "f0:0") == "Marcelo Ebrard"     # introducing statement
    for ri in (1, 2):
        assert _surface(inst, f"f0:{ri}") in {"Ebrard", "El Canciller"}
    assert all(f.subject in q for (q, _, _), f in zip(inst.queries, _facts()))
    cov = inst.alias_coverage
    assert cov["alias_source"] == "natural"
    assert cov["n_alias_events"] == inst.n_alias_events == 2
    assert cov["n_alias_eligible"] == 4
    assert (cov["n_subjects"], cov["n_subjects_natural"],
            cov["n_subjects_fallback"]) == (2, 1, 0)
    assert cov["inventory_path"] == os.path.abspath(inv)
    assert cov["inventory_size"] == 2


def test_uncovered_subject_stays_anchored_or_falls_back(tmp_path):
    inv = _inventory(tmp_path)
    nat = build_instance(_facts(), seed=0, alias_p=1.0, alias_source="natural",
                         alias_inventory=inv)
    for ri in range(3):
        assert _surface(nat, f"f1:{ri}") == "Peter Sagan"
    fb = build_instance(_facts(), seed=0, alias_p=1.0,
                        alias_source="natural+fallback", alias_inventory=inv)
    assert _surface(fb, "f1:0") == "Peter Sagan"
    for ri in (1, 2):
        assert _surface(fb, f"f1:{ri}") in alias_forms("Peter Sagan")
        assert _surface(fb, f"f0:{ri}") in {"Ebrard", "El Canciller"}
    assert fb.n_alias_events == 4
    assert (fb.alias_coverage["n_subjects_natural"],
            fb.alias_coverage["n_subjects_fallback"]) == (1, 1)
    # the mechanical source ignores the inventory entirely
    mech = build_instance(_facts(), seed=0, alias_p=1.0)
    for ri in (1, 2):
        assert _surface(mech, f"f0:{ri}") in alias_forms("Marcelo Ebrard")


def test_natural_source_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_instance(_facts(), alias_p=1.0, alias_source="natural",
                       alias_inventory=str(tmp_path / "missing.json"))
    with pytest.raises(ValueError):
        build_instance(_facts(), alias_p=1.0, alias_source="wikidata")
    with pytest.raises(ValueError):
        alias_pool(_facts()[0], "natural")           # inventory not supplied


def test_linker_inventory_is_the_pool_the_stream_drew_from(tmp_path):
    inv = load_natural_inventory(_inventory(tmp_path))
    facts = _facts()
    assert alias_pools(facts, "natural", inv) == {
        "Marcelo Ebrard": ["Ebrard", "El Canciller"], "Peter Sagan": []}
    fb = alias_pools(facts, "natural+fallback", inv)
    assert fb["Peter Sagan"] == alias_forms("Peter Sagan")
    amap = alias_map_from_pools(alias_pools(facts, "natural", inv))
    assert amap == {"Ebrard": {"Marcelo Ebrard"},
                    "El Canciller": {"Marcelo Ebrard"}}
    assert "Sagan" in alias_map_from_pools(fb)
    # handed that map, the entity-linker rung merges the natural short form
    gaz = [f.subject for f in facts]
    p = AliasKey(gaz, amap, "skip")
    e = np.ones(4, np.float32) / 2
    p.observe(e, "a:0", 0.0, "Marcelo Ebrard holds the position of Mayor.")
    p.observe(e, "a:1", 1.0, "El Canciller holds the position of Senator.")
    p.observe(e, "b:0", 2.0, "Peter Sagan plays for Bora.")
    p.observe(e, "b:1", 3.0, "Sagan plays for Tinkoff.")   # not in the map
    assert p._ids == ["a:1", "b:0", "b:1"]
