#!/usr/bin/env python3
import subprocess
import json
import numpy as np
import cohere
import os
import sys

# Initialize Cohere client
co = cohere.Client(os.getenv("COHERE_API_KEY"))

# Get LIBRARY_ID from environment
LIBRARY_ID = os.getenv("LIBRARY_ID")
if not LIBRARY_ID:
    print("Error: LIBRARY_ID environment variable not set", file=sys.stderr)
    exit(1)

# Run testSearch.sh with LIBRARY_ID
try:
    result = subprocess.run(
        ["./testSearch.sh"],
        capture_output=True,
        text=True,
        env={**os.environ, "LIBRARY_ID": LIBRARY_ID},
        check=True
    )
    search_output = result.stdout
except subprocess.CalledProcessError as e:
    print(f"Error running testSearch.sh for library {LIBRARY_ID}: {e}", file=sys.stderr)
    exit(1)

# Parse search output
try:
    chunks = json.loads(search_output)
    if not isinstance(chunks, list):
        chunks = []
except json.JSONDecodeError:
    print(f"Error: Invalid JSON from testSearch.sh for library {LIBRARY_ID}", file=sys.stderr)
    exit(1)

# Extract texts
texts = [chunk["text"] for chunk in chunks if "text" in chunk]
if not texts:
    print(f"Error: No texts found in testSearch.sh output for library {LIBRARY_ID}", file=sys.stderr)
    exit(1)

# Generate embeddings
try:
    response = co.embed(
        texts=texts,
        model="embed-english-v3.0",
        input_type="search_document"
    )
    embeddings = response.embeddings
except cohere.CohereAPIError as e:
    print(f"Cohere API error: {e}", file=sys.stderr)
    exit(1)

# Save embeddings
output = [
    {
        "chunk_id": chunk["chunk_id"],
        "text": chunk["text"],
        "metadata": chunk["metadata"],
        "embedding": embedding
    }
    for chunk, embedding in zip(chunks, embeddings)
]
with open("embedded_output.json", "a") as f:  # Append to accumulate embeddings
    json.dump(output, f, indent=2)
    f.write("\n")
np.save("embeddings.npy", output)
print(f"Successfully generated embeddings for library {LIBRARY_ID}")
