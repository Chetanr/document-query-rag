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

1. **Ingest:** `POST /ingest` chunks the files in `data/docs/`, embeds each chunk
   with a [sentence-transformers](https://www.sbert.net/) model and stores the
   vectors in a persistent ChromaDB collection (`data/chroma/`, cosine distance).
2. **Screen:** `POST /ask` first scores the question with
   [Detoxify](https://github.com/unitaryai/detoxify). A toxicity score of `0.5`
   or higher is rejected with HTTP `422`, and nothing reaches the LLM.
3. **Retrieve:** the question is embedded with the same model, and ChromaDB
   returns the `TOP_K` nearest chunks.
4. **Generate:** Anthropic Claude answers using those chunks as context.

## Tech stack

Python · FastAPI · ChromaDB · sentence-transformers · Anthropic API · Detoxify · pytest

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[dev]"
cp .env.example .env        # then set ANTHROPIC_API_KEY
```

| Variable            | Purpose                                      |
|---------------------|----------------------------------------------|
| `ANTHROPIC_API_KEY` | Your Anthropic API key                       |
| `ANTHROPIC_MODEL`   | Model ID used to generate answers            |
| `TOP_K`             | Number of chunks to retrieve (example: `3`)  |
| `TOXICITY_THRESHOLD`| Toxicity cutoff, 0–1 (example: `0.5`)        |
| `EMBEDDING_MODEL`   | sentence-transformers model ID (example: `sentence-transformers/all-MiniLM-L6-v2`) |

`.env` is loaded from the project root, and values already set in the process
environment take precedence. `.env` is git-ignored.

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

The first ingest downloads the sentence-transformers model named in
`EMBEDDING_MODEL`, and the first `/ask` downloads the Detoxify weights.

If you change `EMBEDDING_MODEL`, or you have a `data/chroma/` folder created
before this setting existed, delete `data/chroma/` and run `/ingest` again.
Vectors from a different model have a different dimension and distance metric,
so queries against them fail or return poor matches.

## Testing

```bash
python3 -m pytest
```

## Limitations and roadmap

- Only `.md` and `.txt` files are supported.
- Toxicity is checked on the question only, not on retrieved content or answers.
- Planned: more file types, source citations in answers, streaming responses.
