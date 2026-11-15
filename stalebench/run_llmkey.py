"""Is anchor-free same-fact detection solved by asking a language model?

The keying ladder prices the same-fact decision that retention depends on. Its
cheap rungs are mechanical (cosine threshold, gazetteer string match, entity
linker), and the anchor-free track shows the string rung collapsing once
statements stop carrying a verbatim subject. The obvious next move --- the one
every production memory system now makes --- is to let an LLM adjudicate.

This runner adds that rung. For each incoming statement the model is shown the
top-4 stored candidates by cosine and asked which, if any, describes the same
fact. It never sees a fact key, an id, or a timestamp: only statement text. The
result is compared against the mechanical rungs and the oracle, both by
end-to-end current@1 and by the retrieval-independent clustering audit (merge
and split errors against the true fact partition).

Judgments are cached on the exact prompt, so the reported numbers replay
offline at zero cost.

Run:  python -m stalebench.run_llmkey [model-spec ...]
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Sequence

import numpy as np

from stalebench.alias import build_alias_map
from stalebench.embed import MINILM, Embedder
from stalebench.llm import make_client
from stalebench.metrics import bootstrap_ci, paired_permutation
from stalebench.policies import (AliasKey, AliasKeyer, KeepLatestOracle,
                                 KeyedRetention, LLMKey, LLMKeyer, LexicalKey,
                                 LexicalKeyer, OracleKeyer, Reconsolidation,
                                 ThresholdKeyer)
from stalebench.run_alias import _audit_keyer, _tune, score_real
from stalebench.templama import (build_instance, load_facts, split_facts,
                                 subjects_with_class)

_HERE = os.path.dirname(__file__)
_OUT = os.path.join(_HERE, "..", "results", "stalebench_llmkey_results.json")

_PARA = 0.5
_ALIAS_LEVELS = (0.0, 1.0)
_N_CAND = 4
_DEFAULT_MODELS = ["ollama:qwen2.5:14b", "zai:glm-4.6"]
_LIMIT = int(os.environ.get("STALEBENCH_KEY_LIMIT", "0")) or None
_OUT = os.environ.get("STALEBENCH_KEY_OUT") or _OUT


def _cand_recall(inst, emb, n_cand: int) -> Dict:
    """Is the LLM keyer's candidate generator the bottleneck?

    For every statement whose fact already occupies a slot, check whether that
    slot is among the top-``n_cand`` by cosine. If recall is high, a low linking
    rate is the model's judgment rather than a retrieval failure -- which is the
    difference between "an LLM cannot do this" and "we never showed it the
    answer".
    """
    seen: Dict[str, int] = {}
    mat: List[np.ndarray] = []
    ids: List[str] = []
    hit = tot = 0
    for (text, iid, _), e in zip(inst.events, emb):
        k = iid.split(":")[0]
        if k in seen:
            tot += 1
            order = np.argsort(-(np.stack(mat) @ e))[:n_cand]
            if k in {ids[j].split(":")[0] for j in order}:
                hit += 1
        seen[k] = 1
        mat.append(e)
        ids.append(iid)
    return {"n_linkable": tot, "recall_at_n_cand": hit / max(1, tot)}


def run(model_specs: Sequence[str]) -> Dict:
    facts, stats = load_facts()
    val, test = split_facts(facts)
    if _LIMIT:
        val, test = val[:_LIMIT], test[:_LIMIT]
    gaz = [f.subject for f in test]
    amap = build_alias_map(subjects_with_class(test))
    emb = Embedder(MINILM)

    out: Dict = {"config": {"paraphrase_p": _PARA, "n_cand": _N_CAND,
                            "n_test_facts": len(test), "models": list(model_specs)},
                 "reference": {}, "by_model": {}}

    prepared = {}
    for a in _ALIAS_LEVELS:
        inst_v = build_instance(val, paraphrase_p=_PARA, seed=0, alias_p=a)
        inst_t = build_instance(test, paraphrase_p=_PARA, seed=0, alias_p=a)
        ev_v = emb.encode([e[0] for e in inst_v.events], f"val_ev_p{_PARA}_a{a}")
        qv = emb.encode([q[0] for q in inst_v.queries], f"val_q_p{_PARA}_a{a}")
        ev_t = emb.encode([e[0] for e in inst_t.events], f"test_ev_p{_PARA}_a{a}")
        qt = emb.encode([q[0] for q in inst_t.queries], f"test_q_p{_PARA}_a{a}")
        tau = _tune(inst_v, ev_v, qv)["reconsolidation"]["tau"]
        prepared[a] = (inst_t, ev_t, qt, tau)
        out.setdefault("candidate_recall", {})[f"alias_p={a}"] = _cand_recall(
            inst_t, ev_t, _N_CAND)

        ref = {
            "reconsolidation": lambda: Reconsolidation(tau=tau),
            "lexical-key": lambda: LexicalKey(gaz),
            "alias-key(recent)": lambda: AliasKey(gaz, amap, "recent"),
            "keep-latest-oracle": lambda: KeepLatestOracle(),
        }
        res = {n: score_real(f(), inst_t, ev_t, qt) for n, f in ref.items()}
        out["reference"][f"alias_p={a}"] = {
            "tau": tau,
            "n_events": len(inst_t.events),
            "test": {n: r["summary"] for n, r in res.items()},
            "records": {n: [x["current@1"] for x in r["records"]]
                        for n, r in res.items()},
        }
        print(f"[alias_p={a}] reference " +
              "  ".join(f"{n}:{r['summary']['current@1']:.3f}"
                        for n, r in res.items()), flush=True)

    for spec in model_specs:
        client = make_client(spec)
        cell: Dict = {}
        for a in _ALIAS_LEVELS:
            inst_t, ev_t, qt, tau = prepared[a]
            keyer = LLMKeyer(client, n_cand=_N_CAND)
            r = score_real(KeyedRetention(keyer, name="llm-key"),
                           inst_t, ev_t, qt)
            audit = _audit_keyer(LLMKeyer(client, n_cand=_N_CAND), inst_t, ev_t)
            orc = out["reference"][f"alias_p={a}"]["test"]["keep-latest-oracle"]
            base = out["reference"][f"alias_p={a}"]["records"]
            cur = [x["current@1"] for x in r["records"]]
            cell[f"alias_p={a}"] = {
                "summary": r["summary"],
                "ci": {m: bootstrap_ci([x[m] for x in r["records"]], seed=1)
                       for m in ("current@1", "contamination@5")},
                "keying_deficit": orc["current@1"] - r["summary"]["current@1"],
                "audit": audit,
                "n_llm_calls": keyer.n_calls,
                "n_linked": keyer.n_linked,
                "vs_alias_key_p": paired_permutation(
                    cur, base["alias-key(recent)"], seed=2),
                "vs_oracle_p": paired_permutation(
                    cur, base["keep-latest-oracle"], seed=2),
            }
            print(f"[alias_p={a}] {client.name}: current@1 "
                  f"{r['summary']['current@1']:.3f} "
                  f"(deficit {cell[f'alias_p={a}']['keying_deficit']:.3f}), "
                  f"split {audit['frac_split']:.1%}, impure {audit['impure_clusters']}, "
                  f"{keyer.n_calls} calls", flush=True)
        cell["usage"] = {"n_calls": client.n_calls, "n_cached": client.n_cached,
                         "usd": round(client.usd, 4), "model": client.model}
        print(f"  usage: {client.n_calls} calls, {client.n_cached} cached, "
              f"${client.usd:.3f}", flush=True)
        out["by_model"][client.name] = cell
    return out


def merge(paths: Sequence[str], out_path: str = _OUT) -> Dict:
    """Combine per-model keyer results (models differ in throughput by ~10x)."""
    merged = None
    for p in paths:
        d = json.load(open(p))
        if merged is None:
            merged = d
        else:
            merged["by_model"].update(d["by_model"])
            merged["config"]["models"] = sorted(
                set(merged["config"]["models"]) | set(d["config"]["models"]))
    json.dump(merged, open(out_path, "w"), indent=1)
    return merged


def main():
    if sys.argv[1:2] == ["--merge"]:
        m = merge(sys.argv[2:])
        print(f"merged {len(m['by_model'])} models -> {os.path.abspath(_OUT)}")
        for name, cell in m["by_model"].items():
            print(f"  {name}: " + "  ".join(
                f"{a} cur@1 {cell[a]['summary']['current@1']:.3f}"
                for a in ("alias_p=0.0", "alias_p=1.0")))
        return
    specs = sys.argv[1:] or _DEFAULT_MODELS
    res = run(specs)
    os.makedirs(os.path.dirname(_OUT), exist_ok=True)
    json.dump(res, open(_OUT, "w"), indent=1)
    print(f"\nWrote {os.path.abspath(_OUT)}")
    total = sum(m["usage"]["usd"] for m in res["by_model"].values())
    print(f"total API cost: ${total:.2f}")


if __name__ == "__main__":
    main()
