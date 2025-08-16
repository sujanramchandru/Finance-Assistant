from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

app = FastAPI()

index = None
documents = []

model = SentenceTransformer('all-MiniLM-L6-v2')

class DocsRequest(BaseModel):
    docs: list[str]
    
class QueryRequest(BaseModel):
    query: str
    top_k: int = 3  # default number of results


@app.on_event("startup")
def startup_event():
    global index, documents
    try:
        with open("documents.pkl", "rb") as f:
            documents = pickle.load(f)
        index = faiss.read_index("faiss.index")
        print("Loaded existing FAISS index and documents.")
    except Exception:
        dim = 384
        index = faiss.IndexFlatL2(dim)
        documents = []
        print("Initialized empty FAISS index.")

@app.post("/add_documents")
def add_documents(req: DocsRequest):
    docs = req.docs
    global index, documents
    embeddings = model.encode(docs, convert_to_numpy=True)
    if index.ntotal == 0:
        index = faiss.IndexFlatL2(embeddings.shape[1])
    if isinstance(embeddings, list):
        embeddings = np.array(embeddings)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)  # make it (1, d) if only one doc

    index.add(embeddings)
    documents.extend(docs)
    faiss.write_index(index, "faiss.index")
    with open("documents.pkl", "wb") as f:
        pickle.dump(documents, f)
    return {"message": f"Added {len(docs)} documents."}


@app.post("/query")
def query(req: QueryRequest):
    if index.ntotal == 0:
        raise HTTPException(status_code=400, detail="No documents in index.")
    query_embedding = model.encode([req.query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, req.top_k)
    results = [documents[idx] for idx in indices[0] if idx < len(documents)]
    return {"results": results}


