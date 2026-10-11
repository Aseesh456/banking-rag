import os
import tempfile

import streamlit as st

from ingest import ingest_file
from rag_engine import answer_question
from vector_store import list_sources, reset_index

st.set_page_config(page_title="Banking RAG Assistant", page_icon="🏦")
st.title("Banking Document Assistant")
st.caption("Upload banking documents, then ask questions. Answers come only from your documents.")

with st.sidebar:
    st.header("Upload documents")
    files = st.file_uploader("PDF or image", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True)
    if st.button("Add to knowledge base") and files:
        for f in files:
            path = os.path.join(tempfile.gettempdir(), f.name)
            with open(path, "wb") as out:
                out.write(f.getbuffer())
            with st.spinner(f"Processing {f.name}..."):
                n = ingest_file(path)
            os.remove(path)
            st.success(f"{f.name}: {n} chunks added")

    st.header("Knowledge base")
    docs = list_sources()
    if docs:
        for name, n in docs.items():
            st.write(f"- {name} ({n} chunks)")
    else:
        st.write("No documents yet.")
    if st.button("Clear all documents"):
        reset_index()
        st.rerun()

options = ["All documents"] + list(list_sources().keys())
chosen = st.selectbox("Search in", options)
question = st.text_input("Ask a question about your documents")

if st.button("Ask") and question:
    try:
        with st.spinner("Searching and answering..."):
            result = answer_question(question, source=None if chosen == "All documents" else chosen)
    except RuntimeError as e:
        st.error(str(e))
        st.stop()
    st.subheader("Answer")
    st.write(result["answer"])
    if result["sources"]:
        st.subheader("Sources")
        for s in result["sources"]:
            with st.expander(f'{s["source"]} (chunk {s["chunk"]}, distance {s["distance"]})'):
                st.write(s["preview"] + "...")
