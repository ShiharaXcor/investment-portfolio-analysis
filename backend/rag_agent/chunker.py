import tiktoken
from typing import List, Dict, Any

def chunk_text(text:str, max_tokens:int=900, overlap:int=100) -> List[str]:
    enc = tiktoken.get_encoding("cl100k_base")
    toks = enc.encode(text)
    step = max_tokens - overlap
    chunks = []
    for i in range(0, len(toks), step):
        piece = toks[i:i+max_tokens]
        if not piece: break
        chunks.append(enc.decode(piece))
    return chunks

def chunk_from_pages(pages:List[Dict[str,Any]], max_tokens=900, overlap=100) -> List[Dict[str,Any]]:
    out=[]
    for pg in pages:
        parts = chunk_text(pg["text"], max_tokens=max_tokens, overlap=overlap)
        for idx, t in enumerate(parts, 1):
            out.append({"text": t, "metadata": {"page": pg["page"], "section": pg.get("section")}})
    return out
