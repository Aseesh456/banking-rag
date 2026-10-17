import os
import time

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from dotenv import load_dotenv

from vector_store import search, index_exists

load_dotenv(dotenv_path=".env")

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
USE_RERANK = os.getenv("RERANK", "1") == "1"  # set RERANK=0 to skip the cross-encoder (faster, less accurate)

PROMPT = """You are a banking assistant. Answer the question using ONLY the context below.

Rules:
- Give a direct, concise answer in 1 to 4 sentences. Use a short bullet list only if the question asks for several items.
- Do NOT copy or repeat large parts of the context. Put the answer in your own words.
- Include exact figures, rates, dates and conditions when they are asked for.
- If the context does not contain the answer, reply exactly: "I could not find this in the uploaded documents."

Context:
{context}

Question: {question}

Answer:"""


def _model():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing. Add it to the .env file.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(MODEL_NAME)


def answer_question(question, k=4, source=None):
    """Retrieve the top-k chunks, ask Gemini to answer from them, return answer + sources."""
    if not index_exists():
        return {"answer": "No documents uploaded yet. Upload a PDF or image first.", "sources": []}

    hits = search(question, k=k, source=source, rerank=USE_RERANK)
    context = "\n\n---\n\n".join(text for text, _, _, _ in hits)

    prompt = PROMPT.format(context=context, question=question)
    response = None
    for attempt in range(3):
        try:
            response = _model().generate_content(prompt)
            break
        except ResourceExhausted:
            if attempt == 2:
                raise RuntimeError(
                    "Gemini quota exceeded (429). Wait a minute and try again, "
                    "or set GEMINI_MODEL to a lighter model such as gemini-flash-lite-latest."
                )
            time.sleep(10 * (attempt + 1))  # back off 10s, then 20s

    sources = [
        {"source": source, "chunk": chunk, "distance": round(score, 3), "preview": text[:150]}
        for text, source, chunk, score in hits
    ]
    return {"answer": response.text, "sources": sources}
