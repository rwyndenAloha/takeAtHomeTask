import cohere
import json
import os
import subprocess
import numpy as np

# Run testSearch.sh
result = subprocess.run(["./testSearch.sh"], capture_output=True, text=True)
data = json.loads(result.stdout)

# Initialize Cohere client
co = cohere.Client(os.getenv("COHERE_API_KEY"))

# Extract texts
texts = [item["text"] for item in data]

# Generate embeddings
try:
    response = co.embed(
        texts=texts,
        model="embed-english-v3.0",
        input_type="search_document"
    )
except cohere.CohereAPIError as e:
    print(f"Cohere API error: {e}")
    exit(1)

# Add embeddings and save
embeddings = []
for i, item in enumerate(data):
    item["embedding"] = response.embeddings[i]
    embeddings.append({
        "chunk_id": item["chunk_id"],
        "embedding": item["embedding"],
        "text": item["text"],
        "metadata": item["metadata"]
    })

# Save to JSON and NumPy
with open("embedded_output.json", "w") as f:
    json.dump(data, f, indent=2)
np.save("embeddings.npy", embeddings)
print(json.dumps(data, indent=2))
