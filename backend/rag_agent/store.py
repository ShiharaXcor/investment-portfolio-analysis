import os
import chromadb
from chromadb.config import Settings

CHROMA_DIR = os.getenv("CHROMA_DIR", "./processed/chroma")
os.makedirs(CHROMA_DIR, exist_ok=True)

client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(allow_reset=False))
collection = client.get_or_create_collection("rag_collection", metadata={"hnsw:space":"cosine"})

def upsert_chunks(doc_name:str, chunks:list[dict], vectors:list[list[float]], source_type:str="upload"):
    ids, docs, metas = [], [], []
    for i, (row, vec) in enumerate(zip(chunks, vectors), 1):
        ids.append(f"{doc_name}::p{row['metadata'].get('page','NA')}::c{i}")
        docs.append(row["text"])
        metas.append({
            "document_name": doc_name,
            "page": row["metadata"].get("page"),
            "section": row["metadata"].get("section"),
            "source_type": source_type
        })
    collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=vectors)

def _normalize_where(where: dict | None) -> dict | None:
    if not where: return None
    if any(k.startswith("$") for k in where.keys()): return where
    clauses = [{k: {"$eq": v}} for k,v in where.items()]
    if not clauses: return None
    if len(clauses)==1: return clauses[0]
    return {"$and": clauses}

def search_by_embedding(qvec: list[float], top_k: int = 8, where: dict | None = None):
    filt = _normalize_where(where)
    kwargs = {
        "query_embeddings": [qvec],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if filt is not None: kwargs["where"]=filt

    res = collection.query(**kwargs)

    rows=[]
    ids = res.get("ids", [[]])[0] if res.get("ids") else []
    docs = res.get("documents", [[]])[0] if res.get("documents") else []
    metas = res.get("metadatas", [[]])[0] if res.get("metadatas") else []
    dists = res.get("distances", [[]])[0] if res.get("distances") else []

    n = min(len(ids), len(docs), len(metas), len(dists))
    for i in range(n):
        rows.append({
            "id": ids[i],
            "text": docs[i],
            "metadata": metas[i],
            "distance": dists[i],
        })
    return rows
