#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "pyarrow", "numpy", "openai", "python-dotenv", "flask"]
# ///
"""Internal 'find a book-review reviewer' web app (public data only).

Paste a new book's title (and optionally a short blurb). The app embeds it,
finds the most semantically-similar books that were reviewed in sociology
journals over the last ~5 years, and surfaces the PEOPLE who reviewed those
books as candidate reviewers — they've shown both topical fit and a willingness
to write book reviews. All from public OpenAlex data.

Build the corpus first:
    uv run book_reviewer_finder/collect_reviews.py
    uv run book_reviewer_finder/embed_reviews.py
Then run:
    uv run book_reviewer_finder/app.py            # http://127.0.0.1:5057
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, request
from openai import OpenAI

load_dotenv()
ROOT = Path(__file__).resolve().parent
MODEL = "text-embedding-3-small"

app = Flask(__name__)
_books = pd.read_parquet(ROOT / "books.parquet")
_reviews = pd.read_parquet(ROOT / "reviews.parquet")
_emb = np.load(ROOT / "book_embeddings.npy").astype(np.float32)
_emb /= (np.linalg.norm(_emb, axis=1, keepdims=True) + 1e-9)
_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# reviews indexed by work_id for fast reviewer lookup
_rev_by_work = {w: g for w, g in _reviews.groupby("work_id")}


def _embed(text: str) -> np.ndarray:
    v = np.asarray(_client.embeddings.create(model=MODEL, input=[text[:8000]]).data[0].embedding,
                   dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def find_reviewers(query: str, exclude_author: str = "",
                   n_books: int = 60, n_out: int = 25):
    q = _embed(query)
    sims = _emb @ q
    top = np.argsort(-sims)[:n_books]

    excl = {p.strip().lower() for p in exclude_author.split(",") if p.strip()}
    cand: dict[str, dict] = {}
    for idx in top:
        row = _books.iloc[idx]
        sim = float(sims[idx])
        g = _rev_by_work.get(row["work_id"])
        if g is None:
            continue
        for _, rv in g.iterrows():
            name = rv["reviewer_name"]
            if not name or name.strip().lower() in excl:
                continue
            key = rv["reviewer_oaid"] or name
            c = cand.setdefault(key, {
                "name": name, "oaid": rv["reviewer_oaid"],
                "institution": rv["reviewer_institution"],
                "score": 0.0, "best": 0.0, "matches": []})
            c["score"] += sim
            c["best"] = max(c["best"], sim)
            if not c["institution"] and rv["reviewer_institution"]:
                c["institution"] = rv["reviewer_institution"]
            c["matches"].append({"title": row["book_title"], "sim": sim,
                                 "journal": row["journal"], "year": int(row["year"] or 0)})
    ranked = sorted(cand.values(), key=lambda c: (c["score"], c["best"]), reverse=True)
    for c in ranked:
        c["matches"].sort(key=lambda m: -m["sim"])
    return ranked[:n_out]


PAGE = """<!doctype html><meta charset=utf-8>
<title>SF Book-Review Reviewer Finder</title>
<style>
 body{{font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem;color:#1a1a1a}}
 h1{{font-size:1.4rem}} .sub{{color:#666;margin-bottom:1.5rem}}
 textarea,input{{width:100%;padding:.5rem;font:inherit;border:1px solid #ccc;border-radius:6px;box-sizing:border-box}}
 label{{font-weight:600;font-size:.85rem;color:#444;display:block;margin:.8rem 0 .2rem}}
 button{{margin-top:1rem;padding:.55rem 1.4rem;font:inherit;background:#2b6cb0;color:#fff;border:0;border-radius:6px;cursor:pointer}}
 .cand{{border:1px solid #e2e2e2;border-radius:8px;padding:.7rem 1rem;margin:.6rem 0}}
 .cand h3{{margin:0;font-size:1.05rem}} .inst{{color:#666;font-size:.85rem}}
 .sc{{float:right;color:#2b6cb0;font-weight:700}}
 .m{{font-size:.82rem;color:#555;margin:.15rem 0 0 .2rem}} .mj{{color:#999}}
 .bar{{display:inline-block;height:.5rem;background:#2b6cb0;border-radius:3px;vertical-align:middle;margin-right:.4rem}}
</style>
<h1>Find a book-review reviewer</h1>
<div class=sub>Public OpenAlex data · {nbooks} reviewed books · {nrev} reviewers · sociology journals, last ~5 yrs</div>
<form method=post>
 <label>New book — title (and blurb / subtitle helps)</label>
 <textarea name=q rows=3 placeholder="e.g. The Sum of Us: How Racism Costs Everyone and How We Can Prosper Together">{q}</textarea>
 <label>Exclude names (COI — the book's author/known conflicts, comma-separated)</label>
 <input name=excl value="{excl}" placeholder="Jane Smith, John Doe">
 <button type=submit>Find reviewers</button>
</form>
{results}
"""


def render(cands, query):
    if not query:
        return ""
    if not cands:
        return "<p>No candidates found.</p>"
    out = [f"<h2 style='font-size:1.1rem;margin-top:2rem'>Top {len(cands)} candidate reviewers</h2>"]
    for c in cands:
        inst = c["institution"] or "—"
        oa = (f" · <a href='https://openalex.org/{c['oaid']}' target=_blank>OpenAlex</a>"
              if c["oaid"] else "")
        ms = "".join(
            f"<div class=m><span class=bar style='width:{int(m['sim']*60)}px'></span>"
            f"{m['sim']:.2f} &nbsp;{m['title']} "
            f"<span class=mj>· {m['journal']} {m['year']}</span></div>"
            for m in c["matches"][:5])
        out.append(
            f"<div class=cand><span class=sc>{c['score']:.2f}</span>"
            f"<h3>{c['name']}</h3><div class=inst>{inst}{oa} · "
            f"{len(c['matches'])} similar book(s) reviewed</div>{ms}</div>")
    return "\n".join(out)


@app.route("/", methods=["GET", "POST"])
def home():
    q = request.form.get("q", "").strip()
    excl = request.form.get("excl", "").strip()
    cands = find_reviewers(q, excl) if q else []
    return PAGE.format(nbooks=len(_books), nrev=_reviews["reviewer_oaid"].nunique(),
                       q=q, excl=excl, results=render(cands, q))


if __name__ == "__main__":
    app.run(port=5057, debug=False)
