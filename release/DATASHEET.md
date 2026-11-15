# Datasheet: StaleBench

Following Gebru et al., *Datasheets for Datasets*.

## Motivation

**Why was the dataset created?** To isolate *stale retrieval* in agent memory:
the failure in which a store holding several versions of a fact surfaces an
outdated one for a value-agnostic query. Existing freshness benchmarks ask
whether a *model* is current with the world; knowledge-conflict work studies a
*reader* given conflicting context. Neither measures which version reaches the
context, which is the decision an agent's memory actually makes.

**Who created it?** The authors of the accompanying paper. No external funding
specific to this dataset.

## Composition

**What do instances represent?** Two kinds. A *statement* is one natural
language assertion of a (subject, relation, value) triple that was valid over a
run of consecutive years, carrying its year as metadata rather than in the text.
A *query* is a value-agnostic current-value question naming only the subject.

**How many instances?** 5839 TempLAMA fact keys reduce to
900 that are single-valued in every year and
892 whose value also changes at least once --- the
queried set. Each config materializes those facts into ~1k statements; the
`full_corpus` config adds the 4939 multi-valued
keys as unqueried distractors.

**Is any information missing?** Facts that are multi-valued in some year (a
footballer playing for club and country) are excluded from the queried set
because they defeat crisp top-1 scoring. They appear only as distractors. This
is a real restriction on coverage, stated in the paper's limitations.

**Is the data linked to people?** Yes --- public figures (athletes, politicians,
executives) as recorded in Wikidata. No private individuals, no sensitive
attributes beyond public employment, office, party and education.

**Labels.** `gold_value`, and the `current_ids` set in the key file, come from
the Wikidata validity year recorded in TempLAMA, i.e. the value valid in the final year of the window.
They are not produced by any model or retrieval score.

## Collection

**How was the data acquired?** Derived from the public TempLAMA release
(Dhingra et al., 2022), itself built from Wikidata start/end qualifiers.
Verbalization uses hand-written per-relation templates (one canonical, three
paraphrases, three question forms); the canonical statement template is the
TempLAMA cloze with the value filled in.

**Sampling.** No sampling: every TempLAMA key passing the single-valued and
changing filters is included. Val/test is a deterministic MD5 parity split on
the fact key, so it is reproducible and cannot be re-drawn to taste.

## Preprocessing

Consecutive years with the same value collapse into one *run*, so a fact
contributes one statement per distinct value interval rather than one per year.
This prevents inflating `n` by repetition and keeps version counts meaningful.
The alias inventory is derived mechanically from subject strings (head token,
leading token, initialed form, drop-last phrase), filtered for organizational
type words so that no alias is a form no writer would use. Ambiguous aliases
(one form, several subjects) are deliberately **kept**: entity-resolution
ambiguity is part of what the benchmark prices.

## Uses

**Intended use.** Evaluating retention and retrieval policies for agent memory:
which version of a fact reaches the reader's context, at what memory footprint.

**Out of scope.** It is not a test of world knowledge, not a QA benchmark for
readers in general, and not a measure of factual accuracy at large. Statements
are templated, English-only, and cover nine relations; results do not transfer
automatically to naturalistic dialogue.

**Risks.** Facts about named public figures are as of 2020 and are now
outdated; the dataset must not be used as a current-facts source. The "current"
label means *current within the 2010--2020 window*.

## Distribution and maintenance

Released under CC0-1.0, matching Wikidata's dedication. Versioned with the
paper; `manifest.json` carries per-file row counts and MD5s so a run can prove
which revision it scored. Statement records deliberately carry no fact key,
version, or currency field: an earlier revision encoded them in the statement
id, which made the test split solvable by string manipulation. The evaluator (`evaluate.py`) is frozen: metric
definitions must not change across submissions, and any future config is added
as a new config rather than by editing an existing one.
