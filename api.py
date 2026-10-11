import os
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from ingest import ingest_file
from rag_engine import answer_question
from vector_store import list_sources, reset_index

app = FastAPI(title="Banking RAG API")


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


@app.post("/upload")
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


@app.post("/ask")
def ask(body: Question):
    try:
        result = answer_question(body.question, k=body.top_k, source=body.source)
        if not body.include_sources:
            result = {"answer": result["answer"], "sources": [s["source"] for s in result["sources"]]}
        return result
    except RuntimeError as e:
        code = 429 if "quota" in str(e).lower() else 500
        raise HTTPException(status_code=code, detail=str(e))


@app.get("/documents")
def documents():
    return list_sources()


@app.delete("/documents")
def clear_documents():
    reset_index()
    return {"status": "all documents removed"}
