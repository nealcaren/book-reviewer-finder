#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "pyarrow", "numpy", "openai", "python-dotenv"]
# ///
"""Embed each reviewed book's title for semantic reviewer search.

Reads book_reviewer_finder/reviews.parquet, embeds one vector per distinct
reviewed book (its title) with OpenAI text-embedding-3-small, and writes:
  * books.parquet        — one row per book (work_id, book_title, journal, year,
                           book_author, n_reviewers), row-aligned to...
  * book_embeddings.npy  — the (n_books x 1536) embedding matrix.

Titles are short but descriptive; this is enough to place a new book near
topically-similar reviewed books, whose reviewers become candidate reviewers.

    uv run book_reviewer_finder/embed_reviews.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
ROOT = Path(__file__).resolve().parent
MODEL = "text-embedding-3-small"


def embed_texts(client: OpenAI, texts: list[str], batch: int = 256) -> np.ndarray:
    out = []
    for i in range(0, len(texts), batch):
        chunk = [t[:8000] or " " for t in texts[i:i + batch]]
        resp = client.embeddings.create(model=MODEL, input=chunk)
        out.extend([d.embedding for d in resp.data])
        print(f"  embedded {min(i+batch, len(texts))}/{len(texts)}", file=sys.stderr)
    return np.asarray(out, dtype=np.float32)


def main() -> int:
    df = pd.read_parquet(ROOT / "reviews.parquet")
    # one row per book (a book may have multiple reviewer rows)
    agg = {"book_title": ("book_title", "first"), "journal": ("journal", "first"),
           "year": ("year", "first"), "book_author": ("book_author", "first"),
           "n_reviewers": ("reviewer_oaid", "nunique")}
    if "review_abstract" in df.columns:
        agg["review_abstract"] = ("review_abstract", "first")
    books = df.sort_values("year").groupby("work_id", as_index=False).agg(**agg)
    books = books[books["book_title"].notna()].reset_index(drop=True)

    # Merge in OpenAlex topic tags if enrich_books.py has been run.
    topics_path = ROOT / "book_topics.parquet"
    if topics_path.exists():
        tp = pd.read_parquet(topics_path)
        tp["topics_str"] = tp["topics"].apply(
            lambda xs: "; ".join(xs) if isinstance(xs, (list, tuple, np.ndarray)) else "")
        books = books.merge(tp[["work_id", "topics_str"]], on="work_id", how="left")
    else:
        books["topics_str"] = ""

    # Merge in Google Books descriptions if enrich_google_books.py has been run.
    gb_path = ROOT / "google_books.parquet"
    if gb_path.exists():
        gb = pd.read_parquet(gb_path)
        cols = [c for c in ("work_id", "gb_description", "gb_categories") if c in gb.columns]
        books = books.merge(gb[cols], on="work_id", how="left")
    for c in ("gb_description", "gb_categories"):
        if c not in books.columns:
            books[c] = None

    # Embedding text, richest-first: title + Google Books category/description +
    # OpenAlex topics + the review's own abstract. Title is always present; the
    # extras enrich as coverage allows.
    def emb_text(r):
        parts = [str(r["book_title"])]
        if isinstance(r.get("gb_categories"), str) and r["gb_categories"].strip():
            parts.append(r["gb_categories"])
        if r.get("topics_str"):
            parts.append("Topics: " + r["topics_str"])
        for fld in ("gb_description", "review_abstract"):
            v = r.get(fld)
            if isinstance(v, str) and v.strip():
                parts.append(v)
        return ". ".join(parts)

    texts = books.apply(emb_text, axis=1).tolist()
    enriched = sum(1 for t, r in zip(texts, books.itertuples())
                   if len(t) > len(str(r.book_title)) + 3)
    print(f"embedding {len(texts)} books ({enriched} enriched beyond title)",
          file=sys.stderr)

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    emb = embed_texts(client, texts)

    books.to_parquet(ROOT / "books.parquet", index=False)
    np.save(ROOT / "book_embeddings.npy", emb)
    print(f"\nEmbedded {len(books)} books ({emb.shape}) -> books.parquet + "
          f"book_embeddings.npy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
