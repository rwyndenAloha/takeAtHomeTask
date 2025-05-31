import cohere
import json
import os

co = cohere.Client(os.getenv("COHERE_API_KEY"))

def search(query):
    # Existing search logic
    data = [{"document_id": "e1ee86e0-433b-4714-a131-3a1c4a58911a", "text": query, "metadata": {"source": "example"}}]
    texts = [item["text"] for item in data]

    try:
        response = co.embed(texts=texts, model="embed-english-v3.0")
    except cohere.CohereAPIError as e:
        print(f"Cohere API error: {e}")
        exit(1)

    response = co.embed(texts=texts, model="embed-english-v3.0", input_type="search_document")
    for i, item in enumerate(data):
        item["embedding"] = response.embeddings[i]
    return data

# Example
result = search("Sample text")
print(json.dumps(result, indent=2))

