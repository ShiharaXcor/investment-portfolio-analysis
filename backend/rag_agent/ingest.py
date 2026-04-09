# backend/rag_agent/ingest.py

import os
import glob
from backend.rag_agent.extractor import extract_any
from backend.rag_agent.chunker import chunk_from_pages
from backend.rag_agent.embedder import embed_texts
from backend.rag_agent.store import upsert_chunks
from backend.rag_agent.db import log_ingestion

PROCESSED_DIR = "./processed"
CHROMA_DIR = os.path.join(PROCESSED_DIR, "chroma")

def ingest_file(filepath: str, auto: bool = False):
    """
    Ingest a single file and store its chunks and embeddings in the database.
    """
    try:
        content, meta, pages = extract_any(filepath)
        chunks = chunk_from_pages(pages)
        vectors = embed_texts([c["text"] for c in chunks])
        upsert_chunks(
            meta["document_name"],
            chunks,
            vectors,
            source_type="auto" if auto else "manual"
        )
        log_ingestion(filepath, "success")
        print(f"[SUCCESS] Ingested: {filepath}")
    except Exception as e:
        log_ingestion(filepath, f"error: {str(e)}")
        print(f"[ERROR] Failed to ingest {filepath}: {e}")


def ingest_all_processed():
    """
    Automatically ingest all supported files in the processed folder (excluding chroma).
    """
    files = [
        f for f in glob.glob(os.path.join(PROCESSED_DIR, "*.*"))
        if os.path.isfile(f) and "chroma" not in f.lower()
    ]
    for f in files:
        ingest_file(f, auto=True)


if __name__ == "__main__":
    # Optional: run auto-ingest if executed directly
    ingest_all_processed()
