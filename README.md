# Banking RAG Assistant

A retrieval-augmented question-answering service for banking documents. Upload a PDF or image, ask a question, and get an answer grounded in the document with its sources. Personal learning project.

## How it works

1. **Ingest:** PDFs are read with pypdf, images with EasyOCR.
2. **Chunk and embed:** text is split into overlapping chunks and embedded with `all-MiniLM-L6-v2`.
3. **Store:** vectors are saved in a local FAISS index.
4. **Retrieve and re-rank:** the question is embedded, FAISS fetches 20 candidate chunks, and a cross-encoder re-orders them; the top k go to the model.
5. **Generate:** Gemini answers using only those chunks, and the API returns a concise answer plus its sources.

## Stack

Python, FastAPI, Streamlit, LangChain, FAISS, sentence-transformers (embeddings and cross-encoder re-ranking), EasyOCR, Google Gemini, Docker.

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

## Evaluation

Retrieval was measured with `evaluate.py` on 16 factual questions over two public RBI reports (Annual Report 2025-26 and Report on Trend and Progress of Banking in India 2024-25). A retrieved chunk counts as correct when it contains all the expected keywords. Hit@k = share of questions with a correct chunk in the top k; MRR = mean reciprocal rank of the first correct chunk.

| Chunk size (chars) | Hit@4 | MRR |
|---|---|---|
| 300 | 75% | 0.61 |
| 600 | 69% | 0.51 |
| **1000 (default)** | **81%** | **0.66** |
| 1500 | 69% | 0.62 |
| 2000 | 56% | 0.52 |

**Re-ranking.** FAISS first fetches N candidate chunks, then a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) re-orders them by reading the question together with each chunk. Tested at chunk size 1000, k = 4:

| Setup | Hit@4 | MRR |
|---|---|---|
| FAISS only | 81% | 0.66 |
| + re-rank 10 candidates | 81% | 0.71 |
| **+ re-rank 20 candidates (default)** | **88%** | **0.74** |
| + re-rank 30 candidates | 81% | 0.69 |
| + re-rank 50 candidates | 88% | 0.71 |

Re-ranking raised MRR in every setting; the Hit@4 differences are one question each, so treat them as noisy. Set `RERANK=0` in `.env` to turn it off (faster, uses less memory).

Before re-ranking, with chunk size 1000 and FAISS only: Hit@6 = 81% (MRR 0.66) and Hit@8 = 88% (MRR 0.67). Before re-ranking, the three misses at k=4 had their answer in the index but ranked 7, 12 and 34, so the problem was ranking, not missing text. Possible reason for the drop above 1000 characters: the embedding model (all-MiniLM-L6-v2) only reads the first ~256 tokens of a chunk.

Caveats: only 16 questions, keyword matching is strict, and answer quality from Gemini was not scored. Run it yourself:

```bash
python evaluate.py --docs eval_docs --questions eval_set.json --k 4 --chunk-size 1000
```

## Limitations and next steps

- Single local FAISS index, one shared API key (no per-user accounts or separation).
- Small evaluation set (16 questions); ranking can still be improved (for example with hybrid keyword search).
- Planned: larger evaluation set, re-ranking, HTTPS in front of the EC2 deployment.
