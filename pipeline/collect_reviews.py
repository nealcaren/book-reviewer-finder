#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx", "pandas", "python-dotenv", "pyarrow"]
# ///
"""Collect sociology book reviews from OpenAlex (public data only).

Builds the corpus behind the internal "find a reviewer for a new book" tool.
Idea: a person who has REVIEWED a book semantically similar to a new book is a
strong candidate to review the new one — they've shown both topical fit and a
willingness to write book reviews. This script gathers those (reviewer -> book
reviewed) pairs from public OpenAlex records.

Sources (see reviewer_match/soc_venues.py for the venue list):
  * Contemporary Sociology (ISSN 0094-3061) is a pure book-review journal, but
    OpenAlex types most of its reviews as `article`, so we take ALL of its
    article/book-review works.
  * Every other sociology venue: only `type:book-review` (reliable there;
    research articles are correctly typed `article` and excluded).

For each review we keep the reviewer(s) (name, OpenAlex id, institution) and the
reviewed book, parsed out of OpenAlex's citation-style title
("<title> <title>, by <Author>. <City>: <Publisher>, <Year>. pp. ISBN").

Output: book_reviewer_finder/reviews.parquet — one row per (review, reviewer).

    uv run book_reviewer_finder/collect_reviews.py                 # 2020-01-01+
    uv run book_reviewer_finder/collect_reviews.py --since 2019-01-01
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

import httpx
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from soc_venues import SOC_VENUES  # noqa: E402 (vendored alongside this script)

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent  # repo root
OUT = ROOT / "reviews.parquet"
OPENALEX = "https://api.openalex.org/works"
CONTEMP_SOC = "0094-3061"  # pure book-review journal — take all works

KEY = os.environ.get("OPENALEX_API_KEY", "")


def _get(params: dict) -> dict:
    if KEY:
        params["api_key"] = KEY
    for attempt in range(5):
        try:
            r = httpx.get(OPENALEX, params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(2 * (attempt + 1))
                continue
            r.raise_for_status()
        except httpx.HTTPError as e:
            if attempt == 4:
                raise
            time.sleep(1.5 * (attempt + 1))
    return {}


def _paginate(filt: str) -> list[dict]:
    """Cursor-paginate all works matching `filt`."""
    out, cursor = [], "*"
    fields = ("id,title,publication_year,type,authorships,topics,keywords,"
              "abstract_inverted_index,primary_location,referenced_works_count")
    while cursor:
        data = _get({"filter": filt, "per-page": 200, "cursor": cursor,
                     "select": fields})
        out.extend(data.get("results", []))
        cursor = data.get("meta", {}).get("next_cursor")
        if not data.get("results"):
            break
    return out


# --- title parsing --------------------------------------------------------

_PUB_TAIL = re.compile(
    r",\s*by\s+(?P<author>.+?)\.\s*(?P<rest>.+)$", re.I | re.S)

# Social Forces / AJS title their reviews 'Review of "<Book Title>"'.
_REVIEW_OF = re.compile(r'^\s*(?:book\s+)?review of\s+(?P<t>.+?)\s*$', re.I | re.S)


def is_review_of_title(title: str | None) -> bool:
    return bool(title and _REVIEW_OF.match(title))


def _names(items: list | None, k: int) -> str | None:
    """Join the display_names of the first k OpenAlex topic/keyword objects."""
    vals = [x.get("display_name", "") for x in (items or [])[:k] if x.get("display_name")]
    return "; ".join(vals) or None


def _reconstruct_abstract(inv: dict | None) -> str | None:
    """Rebuild plain-text abstract from OpenAlex's inverted index (the review's
    own summary of the book — the best free enrichment for embedding)."""
    if not inv:
        return None
    pos = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    if not pos:
        return None
    return " ".join(pos[i] for i in sorted(pos))[:4000] or None


def _dedupe_doubled(title: str) -> str:
    """OpenAlex doubles the book title ("X X, by ..."). Return a single copy."""
    t = title.strip()
    # If the string before ", by" is exactly two copies of the same text, halve it.
    head = t.split(", by ")[0].strip()
    n = len(head)
    if n > 6 and n % 2 == 1:
        half = head[: n // 2].strip()
        if head[n // 2 + 1:].strip() == half:
            return half
    # exact even-length doubling
    if n > 6 and n % 2 == 0 and head[: n // 2].strip() == head[n // 2:].strip():
        return head[: n // 2].strip()
    return head


def parse_book(title: str | None) -> dict:
    """Pull book_title / book_author / publisher / year out of the citation-style
    review title. Best-effort; missing pieces come back None."""
    if not title:
        return {"book_title": None, "book_author": None, "publisher": None}
    # 'Review of "<Book Title>"' (Social Forces / AJS style): strip the prefix
    # and any surrounding quotes; no author is present in this format.
    rm = _REVIEW_OF.match(title)
    if rm:
        bt = rm.group("t").strip().strip('"“”‘’').strip()
        return {"book_title": bt or None, "book_author": None, "publisher": None}
    book_title = _dedupe_doubled(title)
    author = publisher = None
    m = _PUB_TAIL.search(title)
    if m:
        author = re.sub(r"\s+", " ", m.group("author")).strip(" .,")
        rest = m.group("rest")
        # "<City>: <Publisher>, <Year>. ..."
        pm = re.search(r":\s*([^,.]+?),\s*(?:19|20)\d{2}", rest)
        if pm:
            publisher = pm.group(1).strip()
        else:
            pm = re.search(r"([A-Z][A-Za-z&' ]+(?:Press|Publishers|Books|Routledge|Polity))",
                           rest)
            if pm:
                publisher = pm.group(1).strip()
    return {"book_title": book_title or None, "book_author": author,
            "publisher": publisher}


# --- collection -----------------------------------------------------------

def collect(since: str) -> pd.DataFrame:
    rows: list[dict] = []
    seen_work: set[str] = set()

    def add_work(w: dict, journal: str):
        wid = (w.get("id") or "").split("/")[-1]
        if not wid or wid in seen_work:
            return
        seen_work.add(wid)
        book = parse_book(w.get("title"))
        if not book["book_title"]:
            return
        for a in w.get("authorships", []):
            au = a.get("author", {})
            insts = a.get("institutions", []) or []
            rows.append({
                "work_id": wid,
                "journal": journal,
                "year": w.get("publication_year"),
                "reviewer_name": au.get("display_name"),
                "reviewer_oaid": (au.get("id") or "").split("/")[-1] or None,
                "reviewer_institution": insts[0]["display_name"] if insts else None,
                **book,
                # OpenAlex classifies every work — topics + keywords describe the
                # book's subject and are available for ALL reviews (used for the
                # title+keywords search mode, which stays short/uniform).
                "review_topics": _names(w.get("topics"), 3),
                "review_keywords": _names(w.get("keywords"), 6),
                "review_abstract": _reconstruct_abstract(w.get("abstract_inverted_index")),
                "raw_title": w.get("title"),
            })

    # 1. Contemporary Sociology — all article/book-review works.
    print("Contemporary Sociology (all reviews)...", file=sys.stderr)
    cs = _paginate(f"locations.source.issn:{CONTEMP_SOC},"
                   f"type:article|book-review,from_publication_date:{since}")
    for w in cs:
        add_work(w, "Contemporary Sociology")
    print(f"  {len(cs)} works", file=sys.stderr)

    # 2. Other soc venues — (a) type:book-review works, plus (b) any work whose
    #    title actually starts with "Review of" (Social Forces / AJS style; the
    #    OpenAlex title.search is a loose full-text match, so we client-filter to
    #    genuine "Review of ..." titles, which are book reviews regardless of type).
    for issn, name, _us in SOC_VENUES:
        if issn == CONTEMP_SOC:
            continue
        typed = _paginate(f"locations.source.issn:{issn},type:book-review,"
                          f"from_publication_date:{since}")
        titled_all = _paginate(f"locations.source.issn:{issn},"
                               f"title.search:review of,from_publication_date:{since}")
        titled = [w for w in titled_all if is_review_of_title(w.get("title"))]
        n_before = len(seen_work)
        for w in typed + titled:
            add_work(w, name)
        added = len(seen_work) - n_before
        if added:
            print(f"  {name}: {len(typed)} book-review + {len(titled)} 'Review of' "
                  f"-> {added} new", file=sys.stderr)

    df = pd.DataFrame(rows)
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", default="2020-01-01",
                    help="Earliest publication date (YYYY-MM-DD). Default 2020-01-01.")
    args = ap.parse_args()

    df = collect(args.since)
    ROOT.mkdir(exist_ok=True)
    df.to_parquet(OUT, index=False)
    n_books = df["work_id"].nunique()
    n_rev = df["reviewer_oaid"].nunique()
    print(f"\nCollected {len(df)} (review, reviewer) rows: "
          f"{n_books} reviews, {n_rev} distinct reviewers -> {OUT}")
    print(f"book_author parsed on {df['book_author'].notna().mean()*100:.0f}% of rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
