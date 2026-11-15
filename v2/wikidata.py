"""Shared Wikidata access for the StaleBench resource-depth jobs.

Every job under ``v2/`` needs the same three things: fetch an
entity, read its labels/aliases, and read the values of a property together with
the start/end qualifiers that make them time-scoped. This module provides those
once, with an on-disk cache so a re-run costs no network traffic and every
downstream number is reproducible from the cache alone.

Design notes:

* Wikidata rejects requests without a descriptive ``User-Agent`` (HTTP 403), so
  one is always sent.
* Entities are cached as raw JSON under ``cache/``. The cache is the unit of
  reproducibility: delete it to re-pull, keep it to re-derive.
* ``valid_values`` implements the same convention TempLAMA uses -- a value is
  valid in year *y* if its ``P580`` (start time) is absent or <= y and its
  ``P582`` (end time) is absent or >= y.
"""

from __future__ import annotations

import gzip
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Sequence

UA = "StaleBench-research/0.1 (benchmark label audit; academic use)"
ENTITY_URL = "https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
API_URL = "https://www.wikidata.org/w/api.php"
WP_API = "https://en.wikipedia.org/w/api.php"

START_Q, END_Q = "P580", "P582"
_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


# --------------------------------------------------------------------------- #
# transport
# --------------------------------------------------------------------------- #
def _get(url: str, tries: int = 4, pause: float = 0.35) -> Optional[dict]:
    """GET JSON with retry/backoff. Returns None on permanent failure."""
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code in (429, 503) and attempt < tries - 1:
                time.sleep(2 ** attempt * 2)
                continue
            if attempt == tries - 1:
                return None
        except Exception:
            if attempt == tries - 1:
                return None
            time.sleep(2 ** attempt)
        time.sleep(pause)
    return None


def _cache_path(qid: str) -> str:
    sub = os.path.join(_CACHE, qid[1:3] if len(qid) > 3 else "xx")
    os.makedirs(sub, exist_ok=True)
    return os.path.join(sub, f"{qid}.json.gz")


def entity(qid: str, refresh: bool = False) -> Optional[dict]:
    """Fetch one entity, cached on disk."""
    path = _cache_path(qid)
    if not refresh and os.path.exists(path):
        try:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    data = _get(ENTITY_URL.format(qid=qid))
    if data is None:
        return None
    ent = data.get("entities", {}).get(qid)
    if ent is None:
        return None
    # keep only what the jobs need; full entities are large
    slim = {
        "id": ent.get("id"),
        "labels": ent.get("labels", {}),
        "aliases": ent.get("aliases", {}),
        "claims": ent.get("claims", {}),
        "sitelinks": {k: v for k, v in (ent.get("sitelinks") or {}).items()
                      if k in ("enwiki", "dewiki", "eswiki", "bnwiki")},
    }
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(slim, f)
    return slim


def prefetch(qids: Sequence[str], pause: float = 0.25, log_every: int = 50) -> Dict[str, bool]:
    """Warm the cache for many entities. Returns {qid: ok}."""
    out = {}
    todo = [q for q in dict.fromkeys(qids)]
    for i, q in enumerate(todo, 1):
        cached = os.path.exists(_cache_path(q))
        out[q] = entity(q) is not None
        if not cached:
            time.sleep(pause)
        if log_every and i % log_every == 0:
            print(f"  prefetch {i}/{len(todo)}", flush=True)
    return out


# --------------------------------------------------------------------------- #
# accessors
# --------------------------------------------------------------------------- #
def label(qid: str, lang: str = "en") -> Optional[str]:
    ent = entity(qid)
    if not ent:
        return None
    lab = ent.get("labels", {}).get(lang)
    return lab.get("value") if lab else None


def labels_multi(qid: str, langs: Sequence[str]) -> Dict[str, str]:
    ent = entity(qid)
    if not ent:
        return {}
    out = {}
    for lg in langs:
        lab = ent.get("labels", {}).get(lg)
        if lab:
            out[lg] = lab["value"]
    return out


def aliases(qid: str, lang: str = "en") -> List[str]:
    ent = entity(qid)
    if not ent:
        return []
    return [a["value"] for a in ent.get("aliases", {}).get(lang, [])]


def surface_forms(qid: str, lang: str = "en") -> List[str]:
    """Label plus every alias, deduplicated, label first."""
    lab = label(qid, lang)
    forms = ([lab] if lab else []) + aliases(qid, lang)
    return list(dict.fromkeys(forms))


def _qual_year(qualifiers: dict, pid: str) -> Optional[int]:
    try:
        t = qualifiers[pid][0]["datavalue"]["value"]["time"]
        return int(t[1:5])
    except Exception:
        return None


def timed_values(qid: str, pid: str) -> List[dict]:
    """All item-valued statements of ``pid`` with their validity window.

    Returns dicts: {value_qid, start, end, rank, deprecated}.
    """
    ent = entity(qid)
    if not ent:
        return []
    out = []
    for st in ent.get("claims", {}).get(pid, []):
        mv = st.get("mainsnak", {}).get("datavalue", {}).get("value")
        if not isinstance(mv, dict) or "id" not in mv:
            continue
        q = st.get("qualifiers", {}) or {}
        out.append({
            "value_qid": mv["id"],
            "start": _qual_year(q, START_Q),
            "end": _qual_year(q, END_Q),
            "rank": st.get("rank", "normal"),
            "deprecated": st.get("rank") == "deprecated",
        })
    return out


def valid_values(qid: str, pid: str, year: int, drop_deprecated: bool = True) -> List[str]:
    """Value QIDs whose validity window covers ``year`` (TempLAMA convention)."""
    out = []
    for v in timed_values(qid, pid):
        if drop_deprecated and v["deprecated"]:
            continue
        s, e = v["start"], v["end"]
        if (s is None or s <= year) and (e is None or e >= year):
            out.append(v["value_qid"])
    return list(dict.fromkeys(out))


def value_history(qid: str, pid: str, y0: int, y1: int,
                  drop_deprecated: bool = True) -> Dict[int, List[str]]:
    """Year -> value QIDs valid that year, over [y0, y1]."""
    vals = [v for v in timed_values(qid, pid)
            if not (drop_deprecated and v["deprecated"])]
    hist = {}
    for y in range(y0, y1 + 1):
        cur = [v["value_qid"] for v in vals
               if (v["start"] is None or v["start"] <= y)
               and (v["end"] is None or v["end"] >= y)]
        hist[y] = list(dict.fromkeys(cur))
    return hist


def enwiki_title(qid: str) -> Optional[str]:
    ent = entity(qid)
    if not ent:
        return None
    sl = ent.get("sitelinks", {}).get("enwiki")
    return sl.get("title") if sl else None


def wikipedia_redirects(title: str, limit: int = 200) -> List[str]:
    """Titles that redirect to ``title`` on English Wikipedia."""
    return wikipedia_redirects_batch([title], limit).get(title, [])


def wikipedia_redirects_batch(titles: Sequence[str], limit: int = 500,
                              batch: int = 50) -> Dict[str, List[str]]:
    """Redirects for many titles. The API accepts up to 50 titles per request,
    which matters: one request per subject is unusably slow once Wikimedia
    starts rate-limiting, and this turns ~470 requests into ~10."""
    out: Dict[str, List[str]] = {t: [] for t in titles}
    uniq = [t for t in dict.fromkeys(titles) if t]
    for i in range(0, len(uniq), batch):
        chunk = uniq[i:i + batch]
        params = {
            "action": "query", "format": "json", "prop": "redirects",
            "titles": "|".join(chunk), "rdlimit": "max", "formatversion": "2",
        }
        data = _get(f"{WP_API}?{urllib.parse.urlencode(params)}", pause=0.6)
        if not data:
            continue
        q = data.get("query", {})
        # the API normalises some titles; map them back to what we asked for
        norm = {n["to"]: n["from"] for n in q.get("normalized", [])}
        for page in q.get("pages", []):
            title = page.get("title")
            key = norm.get(title, title)
            if key in out:
                out[key] = [r["title"] for r in page.get("redirects", [])]
        time.sleep(0.4)
    return out


def cache_stats() -> dict:
    n = tot = 0
    for root, _, files in os.walk(_CACHE):
        for f in files:
            if f.endswith(".json.gz"):
                n += 1
                tot += os.path.getsize(os.path.join(root, f))
    return {"entities": n, "bytes": tot}
