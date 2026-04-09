import sys
import os
from backend.rag_agent.extractor import extract_any
from backend.rag_agent.chunker import chunk_from_pages
from backend.rag_agent.embedder import embed_texts
from backend.rag_agent.store import upsert_chunks
from backend.rag_agent.db import log_ingestion

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/ingest_file.py <path-to-file>")
        raise SystemExit(1)

    fp = sys.argv[1]

    if not os.path.exists(fp):
        print(f"File not found: {fp}")
        raise SystemExit(1)

    try:
        # Extract content
        full, meta, pages = extract_any(fp)

        # Chunk text
        chunks = chunk_from_pages(pages, max_tokens=900, overlap=100)

        # Embed
        texts = [c["text"] for c in chunks]
        vecs = embed_texts(texts)

        # Upsert into Chroma
        upsert_chunks(meta["document_name"], chunks, vecs, source_type="manual")

        # Log ingestion into DB
        log_ingestion(fp, "success")

        print({
            "document_name": meta["document_name"],
            "num_chunks": len(chunks),
            "status": "success"
        })

    except Exception as e:
        log_ingestion(fp, f"error: {str(e)}")
        print(f"Error ingesting {fp}: {e}")
