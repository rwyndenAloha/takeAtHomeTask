import cohere
import json
import os
import subprocess

# Run testSearch.sh
result = subprocess.run(["./testSearch.sh"], capture_output=True, text=True)
data = json.loads(result.stdout)

# Initialize Cohere client
co = cohere.Client(os.getenv("COHERE_API_KEY") or "YOUR_NEW_COHERE_API_KEY")

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

# Add embeddings to data
for i, item in enumerate(data):
    item["embedding"] = response.embeddings[i]

# Save output
with open("embedded_output.json", "w") as f:
    json.dump(data, f, indent=2)
print(json.dumps(data, indent=2))
