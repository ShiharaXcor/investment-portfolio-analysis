from backend.rag_agent.embedder import embed_texts
from backend.rag_agent.store import search_by_embedding

def format_citation(meta:dict) -> str:
    name = meta.get("document_name","Unknown")
    section = meta.get("section")
    page = meta.get("page")
    if section and page: return f"({name}, {section}/Page {page})"
    if section: return f"({name}, {section})"
    if page: return f"({name}, Page {page})"
    return f"({name})"

def retrieve(question:str, top_k:int=8, where:dict|None=None):
    qvec = embed_texts([question])[0]
    hits = search_by_embedding(qvec, top_k=top_k, where=where)
    blocks=[]
    cits=[]
    for i,h in enumerate(hits,1):
        blocks.append(f"[{i}] {h['text']}")
        cits.append(format_citation(h["metadata"]))
    context = "\n\n".join(blocks)
    return context, cits, hits
