"""Generate paper/fig_*.pdf from the results JSONs.

Same rule as the tables: every plotted value is read from a results file, none
is transcribed. Three figures carry the paper's three claims.

``fig_ladder``  current@1 for each keyer as the lexical anchor is withdrawn.
                The rungs are indistinguishable at a=0 and fan out by 70 points
                at a=1: the measurement that turns "same-fact detection is the
                open problem" from a claim into a quantity.
``fig_reader``  retrieval current@1 against reader accuracy, per policy. The
                diagonal is what a retrieval metric would predict; timestamp
                reranking sits far below it.
``fig_dose``    reader accuracy as stale versions are added to a constructed
                context, with and without dates surfaced.

Run:  python -m stalebench.make_figures
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

_HERE = os.path.dirname(__file__)
_RES = os.path.join(_HERE, "..", "results")
_PAPER = os.path.join(_HERE, "..", "paper")

_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]

# colourblind-safe, print-safe
_C = {"oracle": "#000000", "entity": "#0072B2", "lexical": "#D55E00",
      "threshold": "#009E73", "rerank": "#CC79A7", "cosine": "#999999"}


def _load(name):
    path = os.path.join(_RES, name)
    return json.load(open(path)) if os.path.exists(path) else None


def fig_ladder(alias) -> None:
    series = [("alias-rw-key", "symmetric + alias table", "#8B4513", "-", "D"),
              ("fuzzy-rw-key", "symmetric fuzzy (read+write)", "#E69F00",
               "-", "*"),
              ("keep-latest-oracle", "true-key oracle (write-path)",
               _C["oracle"], "-", "o"),
              ("alias-key(recent)", "linker + alias table", _C["entity"], "-", "s"),
              ("fuzzy-key", "fuzzy linker (no aliases)", "#56B4E9", "-", "P"),
              ("lexical-key", "gazetteer (verbatim)", _C["lexical"], "-", "^"),
              ("reconsolidation", "threshold $\\tau$", _C["threshold"], "-", "D"),
              ("time-rerank", "time-rerank (no keying)", _C["rerank"], "--", "v"),
              ("cosine-rag", "store-all cosine", _C["cosine"], ":", "x")]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    series = [t for t in series if t[0] in alias["alias_sweep"]["alias_p=1.0"]["test"]]
    for key, label, colour, ls, mk in series:
        ys = [alias["alias_sweep"][f"alias_p={a}"]["test"][key]["summary"]["current@1"]
              for a in _LEVELS]
        ax.plot(_LEVELS, ys, ls, color=colour, marker=mk, ms=4.5, lw=1.6,
                label=label)
    ax.set_xlabel("fraction of later mentions using a short form  ($a$)")
    ax.set_ylabel("current@1")
    ax.set_ylim(0, 1.02)
    ax.set_xticks(_LEVELS)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=7, loc="lower left", ncol=2, framealpha=0.95,
              borderpad=0.4, columnspacing=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(_PAPER, "fig_ladder.pdf"))
    plt.close(fig)


def fig_reader(reader, primary: str) -> None:
    pt = reader["readers"][primary]["policy_track"]["no_dates"]
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.plot([0, 1], [0, 1], color="#bbbbbb", lw=1, ls="--", zorder=0)
    label = {"cosine-rag": "store-all cosine", "time-rerank": "time-rerank",
             "reconsolidation": "threshold $\\tau$", "lexical-key": "gazetteer",
             "keep-latest-oracle": "oracle",
             "fuzzy-read-key": "read-path key\n(contaminated)",
             "fuzzy-rw-key": "symmetric key"}
    colour = {"cosine-rag": _C["cosine"], "time-rerank": _C["rerank"],
              "reconsolidation": _C["threshold"], "lexical-key": _C["lexical"],
              "keep-latest-oracle": _C["oracle"],
              "fuzzy-read-key": "#8B4513", "fuzzy-rw-key": "#E69F00"}
    # the two retention rungs sit almost on top of each other near (1,1), so
    # their labels are placed by hand rather than by a shared rule
    offset = {"cosine-rag": (0, 11, "center"),
              "time-rerank": (-9, -4, "right"),
              "reconsolidation": (-8, 9, "right"),
              "lexical-key": (0, -15, "center"),
              "keep-latest-oracle": (9, 5, "left"),
              # label to the RIGHT of its own dot: placed left it hovers over
              # the time-rerank marker and reads as labelling that instead,
              # which inverts the comparison the arrow exists to make
              "fuzzy-read-key": (9, -3, "left"),
              "fuzzy-rw-key": (-6, 11, "right")}
    # the controlled pair sits at one x with a large gap in y: same rank-1 for
    # every query, so the vertical distance is what contamination costs. Drawn
    # as a connector because that is the whole argument in one mark.
    if all(k in pt for k in ("fuzzy-read-key", "fuzzy-rw-key")):
        xr = pt["fuzzy-read-key"]["retrieval_current@1"]
        y0 = pt["fuzzy-read-key"]["summary"]["correct"]
        y1 = pt["fuzzy-rw-key"]["summary"]["correct"]
        ax.annotate("", xy=(xr, y1), xytext=(xr, y0),
                    arrowprops=dict(arrowstyle="<->", color="#444444",
                                    lw=1.1, shrinkA=4, shrinkB=4), zorder=2)
        ax.annotate(f"{100 * (y1 - y0):.0f} pts at\nidentical current@1",
                    (xr, (y0 + y1) / 2), textcoords="offset points",
                    xytext=(-7, -2), ha="right", va="center", fontsize=7,
                    color="#444444")
    for key, cell in pt.items():
        if key not in label:
            continue
        x, y = cell["retrieval_current@1"], cell["summary"]["correct"]
        ax.scatter([x], [y], s=52, color=colour[key], zorder=3)
        dx, dy, ha = offset.get(key, (0, 11, "center"))
        ax.annotate(label[key], (x, y), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, fontsize=7.5)
    ax.set_xlabel("retrieval current@1")
    ax.set_ylabel("reader accuracy")
    ax.set_xlim(0.35, 1.20)
    ax.set_ylim(0.2, 1.05)
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_title("what the retrieval metric predicts vs. what the reader does",
                 fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(_PAPER, "fig_reader.pdf"))
    plt.close(fig)


def fig_dose(reader) -> None:
    order = [("current_only", "current only"),
             ("plus1_current_first", "+1 stale, current 1st"),
             ("plus1_stale_first", "+1 stale, current last"),
             ("plus2", "+2 stale"),
             ("stale_only", "stale only")]
    # only readers that ran the full constructed-context set have a curve
    models = [m for m in reader["readers"]
              if len(reader["readers"][m]["dose_response"]["no_dates"]) > 1]
    if not models:
        return
    fig, axes = plt.subplots(1, len(models),
                             figsize=(max(4.6, 3.0 * len(models)), 3.4),
                             sharey=True)
    if len(models) == 1:
        axes = [axes]
    for ax, m in zip(axes, models):
        dr = reader["readers"][m]["dose_response"]
        xs = range(len(order))
        for dk, colour, mk, lab in [("no_dates", _C["lexical"], "o", "no dates"),
                                    ("with_dates", _C["entity"], "s",
                                     "dates in context")]:
            if dk not in dr:
                continue            # reader ran without the date ablation
            ys = [dr[dk][k]["summary"]["correct"] if k in dr[dk] else float("nan")
                  for k, _ in order]
            ax.plot(list(xs), ys, color=colour, marker=mk, ms=4.5, lw=1.6, label=lab)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([lbl for _, lbl in order], fontsize=7,
                           rotation=30, ha="right")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25, lw=0.5)
        ax.set_title(m, fontsize=8.5)
    axes[0].set_ylabel("reader accuracy")
    axes[0].legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(_PAPER, "fig_dose.pdf"))
    plt.close(fig)
    return True


def main() -> None:
    os.makedirs(_PAPER, exist_ok=True)
    made = []
    alias = _load("stalebench_alias_results.json")
    if alias:
        fig_ladder(alias)
        made.append("fig_ladder.pdf")
    reader = _load("stalebench_reader_results.json")
    if reader and reader.get("readers"):
        primary = list(reader["readers"])[0]
        fig_reader(reader, primary)
        made.append("fig_reader.pdf")
        if fig_dose(reader) is not False:
            made.append("fig_dose.pdf")
    print(f"Wrote {', '.join(made) or 'nothing'} to {os.path.abspath(_PAPER)}")


if __name__ == "__main__":
    main()
