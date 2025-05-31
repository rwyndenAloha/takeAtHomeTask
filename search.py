import cohere
import json

# Example search logic
def search(query):
    co = cohere.Client("YOUR_COHERE_API_KEY")
    # Simulate search or embedding
    response = co.embed(texts=[query], model="embed-english-v3.0")
    return [{"document_id": "e1ee86e0-433b-4714-a131-3a1c4a58911a", "text": query, ...}]

# Output example
print(json.dumps(search("Sample text")))

