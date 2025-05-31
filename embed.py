import cohere
import json
import os
import subprocess
import argparse

# Run testSearch.sh
parser = argparse.ArgumentParser()
parser.add_argument("--input", default="./testSearch.sh", help="Input script or JSON file")
parser.add_argument("--output", default="embedded_output.json", help="Output JSON file")
args = parser.parse_args()

# Run script or read file
if args.input.endswith(".sh"):
    result = subprocess.run([args.input], capture_output=True, text=True)
    data = json.loads(result.stdout)
else:
    with open(args.input) as f:
        data = json.load(f)

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
