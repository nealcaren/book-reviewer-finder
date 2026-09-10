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
import json
from pathlib import Path

import numpy as np
import pandas as pd
from fastembed import TextEmbedding

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def emb_text(r) -> str:
    parts = [str(r["book_title"])]
    for fld in ("gb_categories", "gb_description", "review_abstract"):
        v = r.get(fld)
        if isinstance(v, str) and v.strip():
            parts.append(v)
    return ". ".join(parts)[:2000]


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
        cols = [c for c in ("work_id", "gb_description", "gb_categories") if c in gb.columns]
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

    texts = [emb_text(b["row"]) for b in books]
    print(f"embedding {len(texts)} books with {MODEL} ...")
    model = TextEmbedding(model_name=MODEL)
    vecs = np.array(list(model.embed(texts)), dtype=np.float32)
    vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)

    out = []
    for b, v in zip(books, vecs):
        r = b["row"]
        out.append({
            "t": str(r["book_title"]),
            "y": int(r["year"]) if pd.notna(r["year"]) else None,
            "j": str(r["journal"]),
            "r": b["reviewers"],
            "e": [round(float(x), 4) for x in v],
        })

    WEB.mkdir(exist_ok=True)
    (WEB / "data.js").write_text(
        "const BOOKS = " + json.dumps(out, ensure_ascii=False) + ";\n")
    n_rev = len({rv[0] for b in out for rv in b["r"]})
    size_mb = (WEB / "data.js").stat().st_size / 1e6
    print(f"wrote {len(out)} books, {n_rev} reviewers -> web/data.js "
          f"({size_mb:.1f} MB, dim={vecs.shape[1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
