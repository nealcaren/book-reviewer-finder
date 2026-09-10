#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "pyarrow", "numpy", "fastembed"]
# ///
"""Embed reviewed books with all-MiniLM-L6-v2 for the BACKEND-FREE web app.

The static site (web/index.html) embeds the user's query IN THE BROWSER with
transformers.js (Xenova/all-MiniLM-L6-v2). For the query vector and the corpus
vectors to be comparable they must come from the SAME model, so here we embed
the books with the ONNX all-MiniLM-L6-v2 via fastembed (384-dim, mean-pooled +
normalized — matching transformers.js {pooling:'mean', normalize:true}).

Embedding text = title + Google Books categories + description + review abstract
(whatever enrichment is available). Output is a JS file the page loads directly
(no fetch, so it works from file://):

    web/data.js  ->  const BOOKS = [{t, y, j, r:[[name,inst,oaid]...], e:[...384]}]

    uv run book_reviewer_finder/embed_local.py --journal "Social Forces"
    uv run book_reviewer_finder/embed_local.py                # whole corpus
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd
from fastembed import TextEmbedding

ROOT = Path(__file__).resolve().parent.parent  # repo root
WEB = ROOT  # data.js at repo root (served by Pages)
MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _join(r, fields) -> str:
    parts = [str(r["book_title"])]
    for fld in fields:
        v = r.get(fld)
        if isinstance(v, str) and v.strip():
            parts.append(v)
    return ". ".join(parts)


def _title_text(r) -> str:
    """Short, uniform representation — title + OpenAlex keywords/topics, which
    every book has, so all title-mode docs stay comparable length."""
    return _join(r, ("review_keywords", "review_topics"))


def _full_text(r) -> str | None:
    """Rich representation (title + keywords + categories + description +
    abstract). None when the book has no description/abstract beyond keywords, so
    full-text search is restricted to books we actually have prose for — keeping
    ALL docs in that mode comparable length (no title-vs-abstract length bias)."""
    extra = [r.get(f) for f in ("gb_description", "review_abstract")]
    if not any(isinstance(v, str) and v.strip() for v in extra):
        return None
    return _join(r, ("review_keywords", "review_topics", "gb_categories",
                     "gb_description", "review_abstract"))[:2000]


def _quantize(vecs: np.ndarray):
    """Unit-normalize + int8-quantize with a single global scale. Returns
    (int8 matrix, scale)."""
    vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
    scale = float(np.abs(vecs).max()) / 127.0
    q = np.clip(np.round(vecs / scale), -127, 127).astype(np.int8)
    return q, scale


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--journal", default=None,
                    help="Restrict to one journal (e.g. 'Social Forces').")
    args = ap.parse_args()

    df = pd.read_parquet(ROOT / "reviews.parquet")
    if args.journal:
        df = df[df["journal"] == args.journal]
    gb_path = ROOT / "google_books.parquet"
    if gb_path.exists():
        gb = pd.read_parquet(gb_path)
        cols = [c for c in ("work_id", "gb_description", "gb_categories", "gb_authors")
                if c in gb.columns]
        df = df.merge(gb[cols], on="work_id", how="left")

    # collapse to books, keeping the reviewer list per book
    books = []
    for wid, g in df.groupby("work_id"):
        first = g.iloc[0]
        if not isinstance(first["book_title"], str):
            continue
        reviewers = []
        seen = set()
        for _, rv in g.iterrows():
            nm = rv.get("reviewer_name")
            if not isinstance(nm, str) or not nm.strip() or nm in seen:
                continue
            seen.add(nm)
            inst = rv.get("reviewer_institution")
            oaid = rv.get("reviewer_oaid")
            reviewers.append([nm,
                              inst if isinstance(inst, str) else "",
                              oaid if isinstance(oaid, str) else ""])
        books.append({"wid": wid, "row": first, "reviewers": reviewers})

    model = TextEmbedding(model_name=MODEL)

    # TWO embeddings per the two search modes. Within each mode every document is
    # comparable length, so there's no title-vs-abstract length bias (see the
    # bias analysis): TITLE = title only, all books; FULL = title + description +
    # abstract, ONLY for books that have such text.
    #   Vectors are int8-quantized with one global scale each (constant scale
    #   drops out of ranking; the browser multiplies by it only to show a ~0-1
    #   similarity) and base64-packed.
    title_texts = [_title_text(b["row"]) for b in books]
    print(f"embedding {len(title_texts)} TITLE texts with {MODEL} ...")
    qt, scale_t = _quantize(np.array(list(model.embed(title_texts)), dtype=np.float32))

    full_idx = [i for i, b in enumerate(books) if _full_text(b["row"]) is not None]
    full_texts = [_full_text(books[i]["row"]) for i in full_idx]
    print(f"embedding {len(full_texts)} FULL-TEXT books (have description/abstract) ...")
    qf, scale_f = _quantize(np.array(list(model.embed(full_texts)), dtype=np.float32))
    ef_by_i = {full_idx[k]: qf[k] for k in range(len(full_idx))}

    out = []
    for i, b in enumerate(books):
        r = b["row"]
        ef = ef_by_i.get(i)
        # Prefer the Google Books author (broad coverage, all journals); fall back
        # to the author parsed from the citation-style review title.
        author = r.get("gb_authors") or r.get("book_author")
        author = author if isinstance(author, str) and author.strip() else None
        out.append({
            "t": str(r["book_title"]),
            "a": author,
            "y": int(r["year"]) if pd.notna(r["year"]) else None,
            "j": str(r["journal"]),
            "r": b["reviewers"],
            "et": base64.b64encode(qt[i].tobytes()).decode("ascii"),
            "ef": base64.b64encode(ef.tobytes()).decode("ascii") if ef is not None else None,
        })

    WEB.mkdir(exist_ok=True)
    (WEB / "data.js").write_text(
        f"const EMB_SCALE_TITLE = {scale_t:.8g};\n"
        f"const EMB_SCALE_FULL = {scale_f:.8g};\n"
        "const BOOKS = " + json.dumps(out, ensure_ascii=False) + ";\n")
    n_rev = len({rv[0] for b in out for rv in b["r"]})
    size_mb = (WEB / "data.js").stat().st_size / 1e6
    print(f"wrote {len(out)} books ({len(full_idx)} with full-text), {n_rev} reviewers "
          f"-> web/data.js ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
