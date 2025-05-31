import numpy as np
import cohere
import os
import json

# Load embeddings
embeddings = np.load("embeddings.npy", allow_pickle=True)

# Initialize Cohere client
co = cohere.Client(os.getenv("COHERE_API_KEY"))

# Query
query = input("Enter query text: ")  # e.g., "Sample text"
try:
    query_embedding = co.embed(
        texts=[query],
        model="embed-english-v3.0",
        input_type="search_query"
    ).embeddings[0]
except cohere.CohereAPIError as e:
    print(f"Cohere API error: {e}")
    exit(1)

# Compute cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

similarities = [
    {
        "chunk_id": emb["chunk_id"],
        "text": emb["text"],
        "metadata": emb["metadata"],
        "similarity": cosine_similarity(query_embedding, emb["embedding"])
    }
    for emb in embeddings
]

# Sort by similarity
similarities = sorted(similarities, key=lambda x: x["similarity"], reverse=True)[:5]

# Output
print(json.dumps(similarities, indent=2))
with open("query_results.json", "w") as f:
    json.dump(similarities, f, indent=2)

