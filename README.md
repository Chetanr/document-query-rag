# document-query-rag

`document-query-rag` is a retrieval-augmented generation (RAG) app for querying
your documents with natural-language questions.

## Project status

- **File ingestion is complete:** `.md` and `.txt` files can be chunked and
  stored in ChromaDB using the `/ingest` endpoint.
- **RAG querying is in progress:** document retrieval and answer generation
  are not yet available.

## Run the API

From the project root, create a virtual environment and install the app and
development dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[dev]"
```

Copy the example environment file and add your Anthropic API key:

```bash
cp .env.example .env
```

Edit `.env` and set `ANTHROPIC_API_KEY`. Set `ANTHROPIC_MODEL` to the Anthropic
model ID to use and `TOP_K` to the number of matching chunks to retrieve (the
example uses `3`). The API loads `.env` from the project root; values already
set in the process environment take precedence. `.env` is git-ignored so the
API key is not committed.

Put `.md` or `.txt` files in `data/docs/`, then start the server:

```bash
uvicorn src.api:app --reload
```

In another terminal, trigger ingestion:

```bash
curl -X POST http://127.0.0.1:8000/ingest
```

The endpoint chunks the files and stores them in the persistent ChromaDB
collection at `data/chroma/`. The response reports the number of chunks
ingested, for example `{"chunks_ingested": 3}`. ChromaDB downloads its default
local embedding model on the first ingest.

ChromaDB uses its collection's default embedding model to embed each question
for retrieval. The `/ask` endpoint retrieves the three most similar chunks and
asks Anthropic to answer using that context:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What does the document say?"}'
```

The response contains the answer, for example `{"answer":"..."}`.

Run the test suite with:

```bash
python3 -m pytest
```
