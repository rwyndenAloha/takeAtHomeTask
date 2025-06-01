import cohere
import json
import os
import subprocess
import numpy as np
import re

# Run testSearch.sh
result = subprocess.run(["./testSearch.sh"], capture_output=True, text=True)
if result.returncode != 0:
    print(f"Error: testSearch.sh failed with: {result.stderr}", file=os.stderr)
    exit(1)

# Extract JSON from testSearch.sh output
json_match = re.search(r'\[.*\]', result.stdout, re.DOTALL)
if not json_match:
    print("Error: No JSON found in testSearch.sh output", file=os.stderr)
    exit(1)

try:
    data = json.loads(json_match.group(0))
except json.JSONDecodeError as e:
    print(f"Error: Invalid JSON in testSearch.sh output: {e}", file=os.stderr)
    exit(1)

# Initialize Cohere client
api_key = os.getenv("COHERE_API_KEY")
if not api_key:
    print("Error: COHERE_API_KEY environment variable not set", file=os.stderr)
    exit(1)
co = cohere.Client(api_key)

# Extract texts
texts = [item["text"] for item in data if "text" in item]
if not texts:
    print("Error: No texts found in testSearch.sh output", file=os.stderr)
    exit(1)

# Generate embeddings
try:
    response = co.embed(
        texts=texts,
        model="embed-english-v3.0",
        input_type="search_document"
    )
except cohere.CohereAPIError as e:
    print(f"Cohere API error: {e}", file=os.stderr)
    exit(1)

# Add embeddings and save
embeddings = []
for i, item in enumerate(data):
    if i < len(response.embeddings):
        item["embedding"] = response.embeddings[i]
        embeddings.append({
            "chunk_id": item.get("chunk_id", ""),
            "embedding": item["embedding"],
            "text": item.get("text", ""),
            "metadata": item.get("metadata", {})
        })

# Save to JSON and NumPy
with open("embedded_output.json", "w") as f:
    json.dump(data, f, indent=2)
np.save("embeddings.npy", np.array(embeddings, dtype=object))
print(json.dumps(data, indent=2))
