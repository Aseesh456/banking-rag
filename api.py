import os
import secrets
import tempfile

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, Security, UploadFile
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from ingest import ingest_file
from rag_engine import answer_question
from vector_store import list_sources, reset_index

load_dotenv(dotenv_path=".env")

app = FastAPI(title="Banking RAG API")

# Clients send the key in an "X-API-Key" header. The /docs page shows an Authorize button for it.
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: str | None = Security(api_key_header)):
    expected = os.getenv("APP_API_KEY")
    if not expected:
        # secure by default: refuse to serve data until a key is configured
        raise HTTPException(status_code=503, detail="APP_API_KEY is not configured on the server.")
    if not key or not secrets.compare_digest(key, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


class Question(BaseModel):
    question: str
    top_k: int = 4
    source: str | None = None  # optional: answer from this file only
    include_sources: bool = False  # set true to also return the retrieved chunks


@app.get("/")
def home():
    return {"message": "Banking RAG API is running. Open /docs to try it."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload", dependencies=[Depends(require_api_key)])
async def upload(file: UploadFile = File(...)):
    name = os.path.basename(file.filename)  # drop any path parts from the upload name
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, name)  # keep the real name as the source label
        with open(path, "wb") as out:
            out.write(await file.read())
        try:
            chunks = ingest_file(path)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    return {"file": name, "chunks_added": chunks}


@app.post("/ask", dependencies=[Depends(require_api_key)])
def ask(body: Question):
    try:
        result = answer_question(body.question, k=body.top_k, source=body.source)
        if not body.include_sources:
            result = {"answer": result["answer"], "sources": [s["source"] for s in result["sources"]]}
        return result
    except RuntimeError as e:
        code = 429 if "quota" in str(e).lower() else 500
        raise HTTPException(status_code=code, detail=str(e))


@app.get("/documents", dependencies=[Depends(require_api_key)])
def documents():
    return list_sources()


@app.delete("/documents", dependencies=[Depends(require_api_key)])
def clear_documents():
    reset_index()
    return {"status": "all documents removed"}
