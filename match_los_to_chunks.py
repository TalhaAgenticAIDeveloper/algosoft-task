"""
LO-to-Chunk Semantic Matching Script
=====================================
Loads learning outcomes and text chunks, computes embeddings using
SentenceTransformers, and maps each LO to its top-matching chunks
via cosine similarity.
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ── Configuration ──────────────────────────────────────────────────
LO_FILE = "clean_los.json"
CHUNKS_FILE = "chunks_clean.json"
OUTPUT_FILE = "lo_with_chunks.json"
MODEL_NAME = "all-MiniLM-L6-v2"       # fast, high-quality sentence embeddings
SIMILARITY_THRESHOLD = 0.25            # minimum cosine similarity to keep (tuned for short-query-to-long-document matching)
TOP_K = 3                              # max chunks per LO
# ───────────────────────────────────────────────────────────────────


def load_json(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: list, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main() -> None:
    # Step 1 – Load data
    print("[1/5] Loading JSON files...")
    los = load_json(LO_FILE)
    chunks = load_json(CHUNKS_FILE)
    print(f"      Loaded {len(los)} learning outcomes and {len(chunks)} chunks.")

    # Step 2 – Generate embeddings with the SAME model
    print(f"[2/5] Loading SentenceTransformer model: {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    lo_descriptions = [lo["description"] for lo in los]
    chunk_contents = [ch["content"] for ch in chunks]

    print("      Encoding LO descriptions...")
    lo_embeddings = model.encode(lo_descriptions, show_progress_bar=True,
                                  convert_to_numpy=True)

    print("      Encoding chunk contents...")
    chunk_embeddings = model.encode(chunk_contents, show_progress_bar=True,
                                     convert_to_numpy=True)

    # Step 3 – Cosine similarity & top-K selection
    print(f"[3/5] Computing cosine similarity (threshold={SIMILARITY_THRESHOLD}) ...")
    sim_matrix = cosine_similarity(lo_embeddings, chunk_embeddings)  # shape: (n_los, n_chunks)

    chunk_ids = [ch["chunk_id"] for ch in chunks]

    results = []
    for idx, lo in enumerate(los):
        scores = sim_matrix[idx]

        # Get indices sorted by descending similarity
        top_indices = np.argsort(scores)[::-1][:TOP_K]

        # Keep only those above the threshold
        matched_chunk_ids = [
            chunk_ids[i]
            for i in top_indices
            if scores[i] >= SIMILARITY_THRESHOLD
        ]

        results.append({
            "lo_id": lo["lo_id"],
            "domain": lo["domain"],
            "subdomain": lo["subdomain"],
            "description": lo["description"],
            "chunks": matched_chunk_ids,
        })

    # Step 4 & 5 – Build output and save
    matched = sum(1 for r in results if r["chunks"])
    print(f"[4/5] {matched}/{len(results)} LOs matched at least one chunk.")

    save_json(results, OUTPUT_FILE)
    print(f"[5/5] Saved → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
