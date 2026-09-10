# Book-Reviewer Finder

A small, **backend-free** tool that suggests candidate reviewers for a *new*
book a journal needs reviewed. Paste the book's title (a subtitle or blurb
helps); the page finds sociologists who reviewed **semantically similar** books
and ranks them as candidates — the idea being that someone who reviewed a
similar book has both topical fit and a demonstrated willingness to review.

Everything runs **in the browser**: the query is embedded client-side with
[transformers.js](https://github.com/huggingface/transformers.js)
(`all-MiniLM-L6-v2`) and cosine-ranked against a prebuilt vector index. No server,
no API keys at runtime.

Built for *Social Forces* editorial use; currently seeded with Social Forces
book reviews (2020–present).

## Data

All from **public sources**:
- **OpenAlex** — book reviews in sociology journals: the reviewer (name,
  institution, OpenAlex id) and the reviewed book (title).
- **Google Books** — book descriptions, used to enrich the embeddings.

Reviewer names/institutions are public academic records (published book-review
bylines). This is a **discovery aid, not an endorsement** — apply your own
editorial and conflict-of-interest judgment.

## Run the site

It's a static site — serve the repo root with any static file server:

```
python3 -m http.server        # then open http://localhost:8000
```

(or host it on GitHub Pages / any static host). First load downloads the
~25 MB embedding model once; it's cached by the browser thereafter.

## Updating the site

Everything needed to refresh the data lives in **`pipeline/`** — including the
two **downloaders** (OpenAlex reviews, Google Books descriptions). Scripts run
with [uv](https://docs.astral.sh/uv/) (each declares its own dependencies) and
read/write at the repo root, so a rebuild regenerates `data.js` in place.

Run in order from the repo root:

```sh
# 1. DOWNLOAD reviews from OpenAlex  ->  reviews.parquet
#    (which journals: pipeline/soc_venues.py; --since sets the date floor)
uv run pipeline/collect_reviews.py --since 2020-01-01

# 2. DOWNLOAD book descriptions from Google Books  ->  google_books.parquet
#    Needs GOOGLE_BOOKS_API_KEY in a .env file (Google Cloud -> Books API).
#    Free quota ~1,000/day; cached + resumable, so re-run daily until done.
#    Omit --journal to cover every journal; --max N caps a single run.
uv run pipeline/enrich_google_books.py

# 3. EMBED + quantize  ->  data.js  (all-MiniLM-L6-v2, int8, ~3 MB)
uv run pipeline/embed_local.py            # or --journal "Social Forces" to narrow

# 4. PUBLISH
git add data.js && git commit -m "refresh data" && git push
```

GitHub Pages rebuilds automatically after the push (~1–2 min).

- **Add a journal?** Add its ISSN to `pipeline/soc_venues.py`, then rerun from
  step 1. (Mobilization, Social Movement Studies, Gender & Society, AJS, Social
  Forces, Contemporary Sociology, etc. are already included.)
- **Adding new books only** is cheap — steps 1–3 are incremental/cached; only
  Google Books is quota-limited.

### How `data.js` is packed
Book vectors are 384-d `all-MiniLM-L6-v2` embeddings, **int8-quantized** with a
single global scale (`EMB_SCALE`) and base64-packed — ~7× smaller than raw
floats, with ranking unchanged (the constant scale drops out of the sort; the
browser multiplies by `EMB_SCALE` only to show a 0–1 similarity). The query is
embedded live in-browser with the same model, so query and corpus vectors are
comparable.

## License

MIT — see [LICENSE](LICENSE).
