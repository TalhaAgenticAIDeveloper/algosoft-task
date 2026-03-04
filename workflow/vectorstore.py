"""
Vector store for LO semantic search
Loads lo_with_chunks.json
Creates FAISS index
"""

import json
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load LO data
with open("lo_with_chunks.json", "r", encoding="utf-8") as f:
    lo_data = json.load(f)

texts = [lo["description"] for lo in lo_data]
metadatas = lo_data

# Create vector store
vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)


def search_los(query: str, k: int = 5):
    """
    Perform semantic search on LO descriptions
    """
    docs = vectorstore.similarity_search(query, k=k)

    results = []
    for doc in docs:
        meta = doc.metadata
        results.append({
            "lo_id": meta["lo_id"],
            "domain": meta["domain"],
            "subdomain": meta["subdomain"],
            "description": meta["description"],
            "chunks": meta["chunks"]
        })

    return results