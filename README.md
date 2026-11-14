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
python3 -m pip install --upgrade pip
python3 -m pip install -e ".[dev]"
```

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

Run the test suite with:

```bash
python -m pytest
```
