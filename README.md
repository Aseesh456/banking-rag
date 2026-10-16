# Banking RAG Assistant

A retrieval-augmented question-answering service for banking documents. Upload a PDF or image, ask a question, and get an answer grounded in the document with its sources. Personal learning project.

## How it works

1. **Ingest:** PDFs are read with pypdf, images with EasyOCR.
2. **Chunk and embed:** text is split into overlapping chunks and embedded with `all-MiniLM-L6-v2`.
3. **Store:** vectors are saved in a local FAISS index.
4. **Retrieve:** the question is embedded and the top-k closest chunks are fetched.
5. **Generate:** Gemini answers using only those chunks, and the API returns a concise answer plus its sources.

## Stack

Python, FastAPI, Streamlit, LangChain, FAISS, sentence-transformers, EasyOCR, Google Gemini, Docker.

## Run with Docker

```bash
cp .env.example .env        # then add your GEMINI_API_KEY
docker compose up --build
```

Open http://127.0.0.1:8000/docs

## Run locally

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn api:app --reload        # API
streamlit run app.py            # web UI
```

## API

All endpoints except `/` and `/health` need an `X-API-Key` header matching `APP_API_KEY` from your `.env`. In `/docs`, click **Authorize** and paste the key.

| Method | Path | Purpose |
|---|---|---|
| GET | /health | liveness check (no key needed) |
| POST | /upload | add a PDF or image to the index |
| POST | /ask | `{"question": "...", "top_k": 4, "source": "file.pdf", "include_sources": false}` returns `answer` and the source file names; `source` limits the search to one file, `include_sources` adds the retrieved chunks |
| GET | /documents | list uploaded files with chunk counts |
| DELETE | /documents | clear the knowledge base |

Uploading a file with the same name replaces its old chunks.

Quota errors from Gemini are retried with backoff and returned as HTTP 429.

## Limitations and next steps

- Single local FAISS index, one shared API key (no per-user accounts or separation).
- Retrieval quality not yet measured on a test set.
- Planned: evaluation set, CI pipeline, cloud deployment.
