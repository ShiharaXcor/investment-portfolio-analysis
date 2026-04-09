import os
import sys
import glob
import streamlit as st
from dotenv import load_dotenv

# Add project root to Python path for backend imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.rag_agent.ingest import ingest_file, ingest_all_processed
from backend.rag_agent.query import retrieve
from backend.rag_agent.llm import answer

load_dotenv()
st.set_page_config(page_title="Finance RAG — Chat", layout="wide")
st.title("🤖 Finance RAG — Financial Assistant")

PROCESSED_DIR = "./processed"

# ---------------------
# Session state
# ---------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "filters" not in st.session_state:
    st.session_state.filters = None

if "ingested" not in st.session_state:
    st.session_state.ingested = False

# ---------------------
# Sidebar: Ingest / Controls
# ---------------------
with st.sidebar:
    st.header("Document Ingestion")

    uploaded_files = st.file_uploader(
        "Upload files (PDF, DOCX, CSV, XLSX, PPTX, TXT, HTML, PNG/JPG)",
        type=["pdf", "docx", "csv", "xlsx", "pptx", "txt", "md", "html", "htm", "png", "jpg", "jpeg", "webp", "tiff"],
        accept_multiple_files=True
    )

    if uploaded_files and st.button("Upload & Ingest"):
        for f in uploaded_files:
            dest = os.path.join(PROCESSED_DIR, f.name)
            with open(dest, "wb") as out:
                out.write(f.getbuffer())
            ingest_file(dest, auto=False)
        st.session_state.ingested = True
        st.success(f"✅ Uploaded & ingested {len(uploaded_files)} file(s)")

    st.divider()
    st.header("Query Settings")
    topk = st.number_input("Top-k", min_value=3, max_value=20, value=8, step=1)
    source_type = st.selectbox("Filter: source_type", ["", "upload", "auto"], index=0)
    st.session_state.filters = {"source_type": source_type} if source_type else None

    st.divider()
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------------------
# Auto-ingest files in processed/
# ---------------------
if not st.session_state.ingested:
    ingest_all_processed()
    st.session_state.ingested = True
    st.info("Auto-ingested all files from /processed/ folder (excluding chroma).")

# ---------------------
# Render chat messages
# ---------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and msg.get("citations"):
            with st.expander("📂 Citations"):
                for i, c in enumerate(msg["citations"], start=1):
                    st.write(f"{i}. {c}")

# ---------------------
# Chat input
# ---------------------
prompt = st.chat_input("Ask your financial questions…")

if prompt:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt, "citations": []})

    # Retrieve relevant context from RAG
    context, citations, _ = retrieve(
        question=prompt,
        top_k=int(topk),
        where=st.session_state.filters
    )

    if not context.strip():
        bot_text = "I don’t have enough context yet. Please ingest documents first."
        bot_cits = []
    else:
        # Pass chat history for continuity
        bot_text = answer(prompt, context, chat_history=st.session_state.messages)
        bot_cits = citations or []

    # Append assistant response
    st.session_state.messages.append({"role": "assistant", "content": bot_text, "citations": bot_cits})

    st.rerun()
