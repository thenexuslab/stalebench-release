"""What the id-attribution convention hides, measured on a real run's contexts.

The systems track scores a wrapped system by mapping each returned memory back
to a benchmark statement id: a memory is attributed to the statement that most
recently wrote it. \\S5.9 disclosed that this is a proxy and a generous one, and
named the correction --- scoring the memory texts --- as its natural successor
rather than performing it. This module performs it, and reports the three
structural facts that explain the gap, so the correction is a measurement
rather than a caveat.

Two distinct memories can be attributed to the same benchmark id. When a
release's UPDATE fires on a memory, the wrapper stamps that memory with the
incoming statement's id; if the release then leaves the memory's *text*
unchanged, the store holds a superseded value under a current version's id.
Both the rank-1 metric and contamination@k read that as clean, because both
identify a version by its id and the ids now agree.

Nothing here changes how a run is scored. It reports the same contexts under
both conventions side by side, plus the counts that say why they differ:

* ``n_collapsed``  --- queries whose top-k holds two or more distinct memories
  sharing one benchmark id, which is the convention losing information;
* ``n_both_values`` --- queries whose top-k *text* names a current and a
  superseded value, i.e. contamination as a reader would meet it;
* ``n_updated_stale`` --- queries whose top-k holds a memory the system marked
  as updated that still reads as a superseded value.

and, over the distinct memories a run surfaced rather than per query,
``n_id_disagrees``: memories whose id names one subject while their text names
another. This separates two failures that an id-level metric reports
identically. A release whose ids are right but coarse relabels a memory on
update without rewriting it, so a current id sits on stale text and the metric
is too kind. A release whose ids are simply wrong makes every id-level number a
statement about the wrapper's bookkeeping, and its miss rate --- which looks
like a system that cannot retrieve --- is the bookkeeping failing, not the
retrieval. The same convention can therefore overstate one release and
understate another, which is exactly what it does here.

Run:  python -m stalebench.diag_attribution results/<...>.contexts.json
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Sequence

from stalebench.metrics import score_ranking, score_texts
from stalebench.reader import _matches, stale_values_of
from stalebench.templama import build_instance, load_facts, split_facts

_HERE = os.path.dirname(__file__)
_OUT_DIR = os.path.join(_HERE, "..", "results", "attribution")
_TOPK = 5


def _names(text: str, values: Sequence[str]) -> bool:
    return any(_matches(text or "", v) for v in values)


def _was_updated(hit: Dict) -> bool:
    """Did the store actually change this memory after writing it?

    One release leaves ``updated_at`` unset until an update fires; another
    stamps it equal to ``created_at`` on every write. Only a difference between
    the two is evidence of an update, so presence alone must not be read as one.
    """
    upd, made = hit.get("updated_at"), hit.get("created_at")
    return bool(upd) and upd != made


def analyse(ctx_path: str) -> Dict:
    ctx = json.load(open(ctx_path))
    cfg = ctx["config"]
    facts, _ = load_facts()
    _, test = split_facts(facts)
    test = test[:cfg["n_test_facts"]] if cfg.get("subsampled") else test
    if len(test) != cfg["n_test_facts"]:
        raise SystemExit(f"{ctx_path}: cannot rebuild its {cfg['n_test_facts']}"
                         f"-fact split")

    out: Dict = {"contexts": os.path.basename(ctx_path), "top_k": _TOPK,
                 "by_config": {}}
    for name, cell in ctx["by_config"].items():
        alias_p = 1.0 if name == "anchor_free" else 0.0
        inst = build_instance(test, paraphrase_p=cfg["paraphrase_p"], seed=0,
                              alias_p=alias_p)
        subj_of = {f"f{i}": fact.subject for i, fact in enumerate(test)}
        per_policy: Dict = {}
        for pol, hits_per_query in cell.items():
            n_collapsed = n_both = n_upd_stale = 0
            surfaced: Dict[str, Dict] = {}
            id_rec: List[Dict] = []
            tx_rec: List[Dict] = []
            for hits, (_, key, cur), fact in zip(hits_per_query, inst.queries,
                                                 test):
                top = hits[:_TOPK]
                gold = fact.current_value
                stale = [v for v in stale_values_of(fact.runs, gold)
                         if v != gold]
                ranked: List[str] = []
                for h in top:
                    if h.get("sid") and h["sid"] not in ranked:
                        ranked.append(h["sid"])
                id_rec.append(score_ranking(ranked, key, cur, (_TOPK,)))
                tx_rec.append(score_texts([h.get("text") or "" for h in top],
                                          gold, stale, (_TOPK,)))
                # a memory id is the store's own identity; a benchmark sid is
                # ours. More of the former than the latter means the convention
                # merged two things the store keeps apart.
                n_mem = len({h.get("id") for h in top if h.get("id")})
                if n_mem > len({h["sid"] for h in top if h.get("sid")}):
                    n_collapsed += 1
                if (any(_names(h.get("text"), [gold]) for h in top)
                        and any(_names(h.get("text"), stale) for h in top)):
                    n_both += 1
                # "the system updated this memory" is a change to it, not the
                # presence of a modification time: one release stamps
                # updated_at at creation, so reading the field as an event
                # counted every memory it ever wrote as an updated one
                if any(_was_updated(h) and _names(h.get("text"), stale)
                       for h in top):
                    n_upd_stale += 1
                for h in top:
                    if h.get("id"):
                        surfaced.setdefault(h["id"], h)
            # does a memory's id name the subject its text names? Counted over
            # distinct memories, because this is a property of the store's
            # contents rather than of any one query's ranking.
            n_agree = n_disagree = n_unknown = 0
            for h in surfaced.values():
                sid = h.get("sid") or h.get("meta_sid")
                claimed = subj_of.get(sid.rsplit(":", 1)[0]) if sid else None
                text_l = (h.get("text") or "").lower()
                named = [s for s in subj_of.values() if s.lower() in text_l]
                if not named or claimed is None:
                    n_unknown += 1
                elif claimed in named:
                    n_agree += 1
                else:
                    n_disagree += 1
            n_q = len(hits_per_query)
            per_policy[pol] = {
                "n_queries": n_q,
                "id_level": {m: sum(r[m] for r in id_rec) / n_q
                             for m in id_rec[0]},
                "text_level": {m: sum(r[m] for r in tx_rec) / n_q
                               for m in tx_rec[0]},
                "n_collapsed": n_collapsed,
                "n_both_values": n_both,
                "n_updated_stale": n_upd_stale,
                "n_memories_surfaced": len(surfaced),
                "n_id_agrees": n_agree,
                "n_id_disagrees": n_disagree,
                "n_id_unknown": n_unknown,
            }
        out["by_config"][name] = per_policy
    return out


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m stalebench.diag_attribution "
                         "<contexts.json> [...]")
    os.makedirs(_OUT_DIR, exist_ok=True)
    for path in sys.argv[1:]:
        res = analyse(path)
        tag = os.path.basename(path).replace(".contexts.json", "")
        dest = os.path.join(_OUT_DIR, f"{tag}.json")
        json.dump(res, open(dest, "w"), indent=1)
        print(f"\n=== {tag} ===")
        for cfgname, per_policy in res["by_config"].items():
            print(f"[{cfgname}]")
            for pol, row in per_policy.items():
                idl, txl = row["id_level"], row["text_level"]
                n_q = row["n_queries"]
                print(f"  {pol:24s} cur@1 id {idl['current@1']:.3f} / text "
                      f"{txl['current@1_text']:.3f}   contam@5 id "
                      f"{idl['contamination@5']:.3f} / text "
                      f"{txl['contamination@5_text']:.3f}   collapsed "
                      f"{row['n_collapsed']}/{n_q}  updated-but-stale "
                      f"{row['n_updated_stale']}/{n_q}  id-names-other-subject "
                      f"{row['n_id_disagrees']}/{row['n_memories_surfaced']}")
        print(f"Wrote {os.path.abspath(dest)}")


if __name__ == "__main__":
    main()
