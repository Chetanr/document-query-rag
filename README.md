# document-query-rag

Ask natural-language questions about your own `.md` and `.txt` documents and get
answers grounded in their content. It's a retrieval-augmented generation (RAG)
API built with FastAPI, ChromaDB and Anthropic Claude, with a toxicity filter
in front of the LLM.

## How it works

```
 /ingest                                   /ask
 .md/.txt files                            question
      │                                       │
   chunking                          toxicity check (Detoxify) ──► 422 if toxic
      │                                       │
 sentence-transformers                 sentence-transformers
 (embed chunks)                        (embed question)
      │                                       │
 ChromaDB (vectors)  ◄── top-K nearest chunks retrieval
                                              │
                                   Claude answers using only
                                   the retrieved context
                                              │
                                          {"answer": ...}
```

1. **Ingest:** `POST /ingest` reads every `.md` and `.txt` file under
   `data/docs/` (recursively, as UTF-8). It splits each file into
   1000-character chunks with a 100-character overlap, embeds each chunk with a
   [sentence-transformers](https://www.sbert.net/) model, and stores the vectors
   in a persistent ChromaDB collection (`data/chroma/`, cosine distance).
   Each chunk's ID is derived from its file path and position, so re-ingesting
   overwrites existing chunks in place (see [Limitations](#limitations-and-roadmap)).
2. **Screen:** `POST /ask` first scores the question with
   [Detoxify](https://github.com/unitaryai/detoxify). A score at or above
   `TOXICITY_THRESHOLD` (`0.5` in `.env.example`) is rejected with HTTP `422`,
   and nothing reaches retrieval or the LLM.
3. **Retrieve:** the question is embedded with the same model, and ChromaDB
   returns the `TOP_K` nearest chunks.
4. **Generate:** Anthropic Claude answers using only those chunks as context.

## Tech stack

Python · FastAPI · ChromaDB · sentence-transformers · Anthropic API · Detoxify · pytest · Ruff

## Quick start

### Requirements

- Python 3.9.1 or newer.
- A recent pip. Older versions, such as pip 20.x bundled with some Python 3.9
  installs, can't do editable installs from `pyproject.toml`.
- Network access and several GB of disk space. Installing pulls in PyTorch,
  and the first run downloads the embedding model and the Detoxify weights.

### Install and configure

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
cp .env.example .env        # then set ANTHROPIC_API_KEY
```

| Variable             | Purpose                                      |
|----------------------|----------------------------------------------|
| `ANTHROPIC_API_KEY`  | Your Anthropic API key                       |
| `ANTHROPIC_MODEL`    | Model ID used to generate answers            |
| `TOP_K`              | Number of chunks to retrieve (example: `3`)  |
| `TOXICITY_THRESHOLD` | Toxicity cutoff, 0–1 (example: `0.5`)        |
| `EMBEDDING_MODEL`    | sentence-transformers model ID (example: `sentence-transformers/all-MiniLM-L6-v2`) |

All five variables are required. Settings are read and checked once, when the
server starts. If `TOP_K`, `TOXICITY_THRESHOLD` or `EMBEDDING_MODEL` is missing
or invalid, the server won't start and logs the error. If `ANTHROPIC_API_KEY`
or `ANTHROPIC_MODEL` is missing, the server starts, but `/ask` returns `503`.

`.env` is loaded from the project root, and values already set in the process
environment take precedence. `.env` is git-ignored. Restart the server after
editing `.env`.

### Run

Add `.md` or `.txt` files to `data/docs/`, then:

```bash
uvicorn src.api:app --reload
curl -X POST http://127.0.0.1:8000/ingest
# {"chunks_ingested": 3}

curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What does the document say?"}'
# {"answer":"..."}
```

Interactive API docs are available at <http://127.0.0.1:8000/docs>.

The first ingest downloads the sentence-transformers model named in
`EMBEDDING_MODEL`, and the first `/ask` downloads the Detoxify weights.

> **Note:** everything in `data/docs/` is indexed, and the folder is **not**
> git-ignored. Don't put private documents there, or ignore them locally
> (for example in `.git/info/exclude`).

If you change `EMBEDDING_MODEL`, or you have a `data/chroma/` folder created
before this setting existed, delete `data/chroma/` and run `/ingest` again.
Vectors from a different model have a different dimension and distance metric,
so queries against them fail or return poor matches.

## API

| Endpoint | Request body | Success | Errors |
|---|---|---|---|
| `POST /ingest` | none | `200 {"chunks_ingested": n}` | `404` document directory missing |
| `POST /ask` | `{"question": "..."}` | `200 {"answer": "..."}` | `422` empty or toxic question · `503` missing API key or model |

## Project layout

```
src/
  api.py            FastAPI app, startup wiring (settings + pipeline), endpoints
  config.py         Settings loaded and validated from .env / environment
  pipeline.py       RAGPipeline: ingest (load → chunk → embed → store) and ask
  loaders.py        Reads .md/.txt files from a directory
  chunking.py       Fixed-size character chunking with overlap
  embeddings.py     Embedder interface and sentence-transformers implementation
  vectorstore.py    ChromaDB wrapper (upsert, query)
  llm.py            LLM interface and Anthropic implementation
  utils/toxicity.py Detoxify-based toxicity check
tests/              pytest suite (conftest.py isolates settings and toxicity)
data/docs/          Documents to ingest
data/chroma/        Persistent vector store (git-ignored)
```

## Development

```bash
python -m pytest
ruff check .
ruff format --check .
```

The tests run offline. `tests/conftest.py` overrides settings and stubs the
toxicity check, so no API key or model download is needed.

## Limitations and roadmap

- Only `.md` and `.txt` files are supported.
- Toxicity is checked on the question only, not on retrieved content or answers.
  The Detoxify `original` model is English-only.
- Re-ingesting doesn't remove chunks from documents that were shortened or
  deleted, so stale content can still be retrieved. Workaround: delete
  `data/chroma/` and run `/ingest` again.
- Anthropic timeouts, rate limits and API errors aren't mapped to specific
  responses; they surface as `500`.
- Planned: more file types, source citations in answers, streaming responses.
