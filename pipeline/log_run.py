#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "pyarrow"]
# ///
"""Append a one-row snapshot of corpus/enrichment state to run_log.csv.

Run at the end of each refresh (manually or by the daily GitHub Action) so the
repo keeps a permanent, human-readable history of how the corpus and its
Google-Books enrichment grow over time. Columns:

  date, total_books, reviewers, journals,
  gb_attempted, gb_with_description, review_abstracts,
  full_text_books, title_only_books,
  new_books, new_descriptions            (deltas vs the previous logged row)

    uv run pipeline/log_run.py
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Works whether run from the repo (parquets at repo root) or the build folder.
HERE = Path(__file__).resolve().parent
ROOT = next((p for p in (HERE, HERE.parent) if (p / "reviews.parquet").exists()), HERE.parent)
LOG = ROOT / "run_log.csv"

FIELDS = ["date", "total_books", "reviewers", "journals", "gb_attempted",
          "gb_with_description", "review_abstracts", "full_text_books",
          "title_only_books", "new_books", "new_descriptions"]


def main() -> int:
    rev = pd.read_parquet(ROOT / "reviews.parquet").drop_duplicates("work_id")
    gb_path = ROOT / "google_books.parquet"
    gb = pd.read_parquet(gb_path) if gb_path.exists() else pd.DataFrame(columns=["gb_description"])

    total = len(rev)
    gb_desc = int(gb["gb_description"].notna().sum()) if len(gb) else 0
    abstracts = int(rev["review_abstract"].notna().sum())
    # full-text = has a Google Books description OR a review abstract (== embed_local's set)
    desc_ids = set(gb.loc[gb["gb_description"].notna(), "work_id"]) if len(gb) else set()
    abs_ids = set(rev.loc[rev["review_abstract"].notna(), "work_id"])
    full_text = len(desc_ids | abs_ids)

    row = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "total_books": total,
        "reviewers": int(rev["reviewer_oaid"].nunique()),
        "journals": int(rev["journal"].nunique()),
        "gb_attempted": len(gb),
        "gb_with_description": gb_desc,
        "review_abstracts": abstracts,
        "full_text_books": full_text,
        "title_only_books": total - full_text,
        "new_books": "", "new_descriptions": "",
    }

    prev = None
    if LOG.exists():
        rows = list(csv.DictReader(LOG.open()))
        if rows:
            prev = rows[-1]
    if prev:
        try:
            row["new_books"] = total - int(prev["total_books"])
            row["new_descriptions"] = gb_desc - int(prev["gb_with_description"])
        except (ValueError, KeyError):
            pass

    exists = LOG.exists()
    with LOG.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)
    print(f"logged: {row['date']} — {total} books, {full_text} full-text "
          f"(+{row['new_books']} books, +{row['new_descriptions']} descriptions) -> {LOG.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
