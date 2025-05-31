import cohere
import json
import os
import subprocess

# Run testSearch.sh and capture output
result = subprocess.run(["./testSearch.sh"], capture_output=True, text=True)
data = json.loads(result.stdout)

# Initialize Cohere client
co = cohere.Client(os.getenv("COHERE_API_KEY"))

# Extract texts
texts = [item["text"] for item in data]

# Generate embeddings
response = co.embed(
    texts=texts,
    model="embed-english-v3.0",
    input_type="search_document"
)

# Add embeddings to data
for i, item in enumerate(data):
    item["embedding"] = response.embeddings[i]

# Save or print
with open("embedded_output.json", "w") as f:
    json.dump(data, f, indent=2)
print(json.dumps(data, indent=2))
EOF
