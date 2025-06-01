from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
from typing import List
import numpy as np
from services import VectorDBService  # Import VectorDBService

# Existing app setup
app = FastAPI()
db = VectorDBService()  # Singleton instance

# Request model for the new endpoint
class ChunkSearchRequest(BaseModel):
    query_embedding: List[float]
    start_date: str
    name_contains: str

# Response model
class ChunkSearchResult(BaseModel):
    library_id: str
    document_id: str
    chunk_id: str
    text: str
    metadata: dict
    created_at: str
    similarity: float

# New endpoint
@app.post("/chunks/search/", response_model=List[ChunkSearchResult])
async def search_chunks(request: ChunkSearchRequest):
    try:
        # Parse start_date
        try:
            start_date = datetime.fromisoformat(request.start_date.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use ISO 8601 (e.g., 2025-06-01T00:00:00Z)")

        # Convert query embedding to numpy array
        query_embedding = np.array(request.query_embedding)

        # Search chunks using VectorDBService
        results = db.search_chunks_after_date(
            query_embedding=query_embedding,
            start_date=start_date,
            name_contains=request.name_contains.lower()
        )

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

# Add to services.py (if not already present)
"""
def search_chunks_after_date(self, query_embedding: np.ndarray, start_date: datetime, name_contains: str) -> List[dict]:
    results = []
    for library_id, library in self.libraries.items():
        for document in library.documents:
            for chunk in document.chunks:
                # Filter by date
                chunk_date = datetime.fromisoformat(chunk.created_at.replace("Z", "+00:00"))
                if chunk_date <= start_date:
                    continue
                # Filter by metadata source
                if name_contains not in chunk.metadata.get("source", "").lower():
                    continue
                # Perform similarity search
                chunk_embedding = np.array(chunk.embedding)
                if chunk_embedding.shape != query_embedding.shape:
                    continue  # Skip mismatched embeddings
                # Compute cosine similarity
                similarity = np.dot(chunk_embedding, query_embedding) / (
                    np.linalg.norm(chunk_embedding) * np.linalg.norm(query_embedding)
                )
                if similarity > 0.5:  # Threshold for relevance
                    results.append({
                        "library_id": library_id,
                        "document_id": document.id,
                        "chunk_id": chunk.id,
                        "text": chunk.text,
                        "metadata": chunk.metadata,
                        "created_at": chunk.created_at,
                        "similarity": float(similarity)
                    })
    return results
"""
