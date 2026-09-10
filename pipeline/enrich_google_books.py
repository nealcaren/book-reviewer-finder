#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx", "pandas", "pyarrow", "python-dotenv"]
# ///
"""Enrich reviewed books with Google Books descriptions + categories.

The best public source of real book blurbs. For each distinct reviewed book we
query the Google Books API by title (and author when we parsed one), take the
best-matching volume, and keep its `description` (usually a full paragraph) and
`categories`. This is the richest embedding text available — it lifts semantic
coverage well beyond titles/topics.

Needs GOOGLE_BOOKS_API_KEY in .env (see the Books API in Google Cloud console).
The free quota is ~1,000 requests/day, so a full backfill of the ~3,800-book
corpus spans a few days — but this script is CACHED + RESUMABLE and stops
cleanly when it hits the daily quota, so just re-run it each day (or request a
quota bump) until it's done.

Output: google_books.parquet (work_id, gb_title, gb_description, gb_categories,
gb_authors, gb_publisher, gb_published). All come from the SAME volume response,
so keeping author/publisher costs no extra quota.

    uv run book_reviewer_finder/enrich_google_books.py
    uv run book_reviewer_finder/enrich_google_books.py --max 900   # cap per run
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

load_dotenv()
ROOT = Path(__file__).resolve().parent.parent  # repo root
OUT = ROOT / "google_books.parquet"
API = "https://www.googleapis.com/books/v1/volumes"
KEY = os.environ.get("GOOGLE_BOOKS_API_KEY", "")


class QuotaExceeded(Exception):
    pass


def _words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", (s or "").lower()) if len(w) > 3}


def query(title: str, author: str | None) -> dict:
    """Return {gb_title, gb_description, gb_categories} for the best volume match,
    or empties. Raises QuotaExceeded on a 429/quota 403 so the caller can stop."""
    q = f'intitle:{title[:120]}'
    if isinstance(author, str) and author.strip():
        q += f' inauthor:{author.split(",")[0].split(" and ")[0].strip()}'
    params = {"q": q, "maxResults": 1, "country": "US", "key": KEY}
    for attempt in range(3):
        try:
            r = httpx.get(API, params=params, timeout=30)
            if r.status_code in (429,) or (r.status_code == 403 and "quota" in r.text.lower()):
                raise QuotaExceeded()
            if r.status_code != 200:
                return {}
            it = (r.json().get("items") or [{}])[0].get("volumeInfo", {})
            gt = it.get("title") or ""
            # guard against a wildly-wrong match: require some title-word overlap
            if gt and not (_words(title) & _words(gt)):
                return {}
            return {"gb_title": gt or None,
                    "gb_description": it.get("description") or None,
                    "gb_categories": "; ".join(it.get("categories") or []) or None,
                    "gb_authors": "; ".join(it.get("authors") or []) or None,
                    "gb_publisher": it.get("publisher") or None,
                    "gb_published": it.get("publishedDate") or None}
        except httpx.HTTPError:
            if attempt == 2:
                return {}
            time.sleep(1.0 * (attempt + 1))
    return {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max", type=int, default=100000,
                    help="Max NEW lookups this run (to stay under the daily quota).")
    ap.add_argument("--journal", default=None,
                    help="Only enrich books reviewed in this journal (e.g. 'Social Forces').")
    args = ap.parse_args()
    if not KEY:
        sys.exit("GOOGLE_BOOKS_API_KEY not set in .env")

    df = pd.read_parquet(ROOT / "reviews.parquet")
    if args.journal:
        df = df[df["journal"] == args.journal]
        print(f"restricted to journal={args.journal!r}", file=sys.stderr)
    df = df.drop_duplicates("work_id")
    df = df[df["book_title"].notna()][["work_id", "book_title", "book_author"]]

    done: dict[str, dict] = {}
    if OUT.exists():
        for r in pd.read_parquet(OUT).to_dict("records"):
            done[r["work_id"]] = r
    todo = df[~df["work_id"].isin(done)]
    print(f"{len(df)} books; {len(done)} cached; {len(todo)} to do "
          f"(cap {args.max} this run)", file=sys.stderr)

    rows = list(done.values())
    n_new = 0
    hit_quota = False
    for _, b in todo.iterrows():
        if n_new >= args.max:
            break
        try:
            res = query(str(b["book_title"]), b["book_author"])
        except QuotaExceeded:
            hit_quota = True
            print("  daily quota reached — stopping (resume tomorrow).", file=sys.stderr)
            break
        rows.append({"work_id": b["work_id"], **res})
        n_new += 1
        if n_new % 200 == 0:
            pd.DataFrame(rows).to_parquet(OUT, index=False)
            print(f"  {n_new} new looked up "
                  f"({sum(1 for r in rows if r.get('gb_description'))} w/ description)",
                  file=sys.stderr)
        time.sleep(0.15)

    out = pd.DataFrame(rows)
    out.to_parquet(OUT, index=False)
    cov = out["gb_description"].notna().mean() * 100 if len(out) else 0
    print(f"\nCached {len(out)}/{len(df)} books "
          f"({cov:.0f}% with a description) -> {OUT}")
    if hit_quota or len(out) < len(df):
        print(f"  {len(df) - len(out)} remaining — re-run tomorrow (resumable) "
              f"or raise the Books API quota.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
