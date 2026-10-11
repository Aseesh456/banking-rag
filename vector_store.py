import os
import shutil

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

INDEX_DIR = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_embeddings = None


def get_embeddings():
    """Load the embedding model once and reuse it."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def chunk_text(text, chunk_size=600, overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)


def add_document(text, source_name):
    """Chunk a document, embed it and add it to the FAISS index (creates the index if missing)."""
    chunks = chunk_text(text)
    if not chunks:
        return 0
    metadatas = [{"source": source_name, "chunk": i} for i in range(len(chunks))]
    embeddings = get_embeddings()

    if os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
        db = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
        # re-uploading a file with the same name replaces its old chunks
        old_ids = [i for i, d in db.docstore._dict.items() if d.metadata.get("source") == source_name]
        if old_ids:
            db.delete(old_ids)
        db.add_texts(chunks, metadatas=metadatas)
    else:
        db = FAISS.from_texts(chunks, embeddings, metadatas=metadatas)

    db.save_local(INDEX_DIR)
    return len(chunks)


def index_exists():
    return os.path.exists(os.path.join(INDEX_DIR, "index.faiss"))


def reset_index():
    """Delete all stored documents."""
    shutil.rmtree(INDEX_DIR, ignore_errors=True)


def list_sources():
    """Return {file name: number of chunks} for everything in the index."""
    if not index_exists():
        return {}
    db = FAISS.load_local(INDEX_DIR, get_embeddings(), allow_dangerous_deserialization=True)
    counts = {}
    for d in db.docstore._dict.values():
        name = d.metadata.get("source")
        counts[name] = counts.get(name, 0) + 1
    return counts


def search(question, k=4, source=None):
    """Return the k most similar chunks as (text, source, chunk_number, distance).
    If source is given, only chunks from that file are searched."""
    if not index_exists():
        return []
    db = FAISS.load_local(INDEX_DIR, get_embeddings(), allow_dangerous_deserialization=True)
    if source:
        results = db.similarity_search_with_score(question, k=k, filter={"source": source}, fetch_k=max(50, k * 10))
    else:
        results = db.similarity_search_with_score(question, k=k)
    return [
        (doc.page_content, doc.metadata.get("source"), doc.metadata.get("chunk"), float(score))
        for doc, score in results
    ]
