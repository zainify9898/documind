"""DocuMind — Chat with Your PDFs. Streamlit web UI.

Run:  streamlit run app.py

Upload a PDF, wait for indexing, then ask questions in the chat box.
Answers are grounded in the document and cited by page number. Works
fully offline unless OPENAI_API_KEY is set, in which case answers are
generated with gpt-4o-mini from the retrieved passages.
"""

from __future__ import annotations

import os
import tempfile

import streamlit as st

from src.embeddings import EMBEDDING_DIM, Embedder
from src.ingest import ingest_pdf
from src.qa import DocuMindQA
from src.vectorstore import VectorStore

st.set_page_config(page_title="DocuMind — Chat with Your PDFs", page_icon="📄", layout="wide")

st.title("📄 DocuMind — Chat with Your PDFs")
st.write(
    "Upload a PDF and ask questions about it. Answers come only from the "
    "document, with page citations. Fully offline unless you set "
    "`OPENAI_API_KEY` for generative answers."
)


@st.cache_resource(show_spinner=False)
def get_embedder() -> Embedder:
    """Load the embedding model once and reuse it."""
    return Embedder()


def index_pdf(pdf_bytes: bytes) -> tuple[DocuMindQA, int]:
    """Ingest, embed and index an uploaded PDF."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    try:
        chunks = ingest_pdf(tmp_path)
        store = VectorStore(EMBEDDING_DIM)
        store.add(chunks, get_embedder().embed([c.text for c in chunks]))
        return DocuMindQA(store, get_embedder()), len(chunks)
    finally:
        os.unlink(tmp_path)


# --- Sidebar: upload & status ---------------------------------------------
with st.sidebar:
    st.header("Your document")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    mode = "generative" if os.environ.get("OPENAI_API_KEY") else "extractive"
    st.info(f"Answer mode: **{mode}**")
    if mode == "extractive":
        st.caption(
            "Extractive mode: answers are the most relevant passages from "
            "the document. Set OPENAI_API_KEY for generated answers."
        )
    if "qa" in st.session_state:
        if st.button("Clear document"):
            del st.session_state["qa"]
            st.session_state.pop("messages", None)
            st.rerun()

# --- Indexing --------------------------------------------------------------
if uploaded is not None and "qa" not in st.session_state:
    with st.spinner(f"Indexing {uploaded.name} ..."):
        qa, n_chunks = index_pdf(uploaded.read())
    st.session_state["qa"] = qa
    st.session_state["doc_name"] = uploaded.name
    st.session_state["messages"] = []
    st.success(f"✅ Indexed **{uploaded.name}** — {n_chunks} chunks ready.")

# --- Chat ------------------------------------------------------------------
if "qa" in st.session_state:
    st.subheader(f"Ask about: {st.session_state['doc_name']}")
    for message in st.session_state.get("messages", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("Sources"):
                    for source in message["sources"]:
                        st.markdown(f"**Page {source.page}** (similarity {source.score:.2f})")
                        st.caption(source.text[:500] + ("..." if len(source.text) > 500 else ""))

    question = st.chat_input("Ask a question about this PDF...")
    if question:
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.spinner("Searching the document..."):
            answer = st.session_state["qa"].ask(question)
        with st.chat_message("assistant"):
            st.markdown(answer.answer)
            st.caption(answer.note)
            if answer.sources:
                with st.expander("Sources"):
                    for source in answer.sources:
                        st.markdown(f"**Page {source.page}** (similarity {source.score:.2f})")
                        st.caption(source.text[:500] + ("..." if len(source.text) > 500 else ""))
        st.session_state["messages"].append(
            {"role": "assistant", "content": answer.answer, "sources": answer.sources}
        )
else:
    st.info("👆 Upload a PDF in the sidebar to get started. Try `data/sample.pdf` from the repo.")
