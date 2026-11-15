"""StaleBench: a benchmark for stale-retrieval / knowledge-freshness in agent memory.

Motivation. Retrieval-augmented agents accumulate facts that *change over time*.
When a user asks "what is X's current value?" (a value-agnostic query that names
the entity, not the answer), a similarity retriever cannot tell an old value of X
from the new one --- both mention X --- and routinely surfaces the **stale**
value. Standard long-term-memory benchmarks hide this failure (LongMemEval's
knowledge-update questions have multi-session gold and weak stale interference,
so plain recency wins).

StaleBench isolates the failure with value-agnostic queries, mechanism-independent
current-version labels, and a memory budget, so that:

* a similarity retriever's staleness error is measurable and severe, and
* a freshness-aware retention policy (keep the latest value per key) can be shown
  to fix it.

The package ships a controlled generator and a real-data track over TempLAMA
(Wikidata temporal triples).
"""

__version__ = "1.2.1"
