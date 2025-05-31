from fastapi import FastAPI, HTTPException
from models import Library, Document, Chunk
from services import VectorDBService
from typing import List, Dict

app = FastAPI(title="Vector Database API")

db_service = VectorDBService()

@app.post("/libraries/", response_model=Library)
async def create_library(library: Library):
    return db_service.create_library(library)

@app.get("/libraries", response_model=List[Dict])
@app.get("/libraries/", response_model=List[Dict])  # Add this
async def get_libraries():
    try:
        libraries = db_service.get_libraries()
        return libraries
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/libraries/{library_id}", response_model=Library)
async def get_library(library_id: str):
    library = db_service.get_library(library_id)
    if not library:
        raise HTTPException(status_code=404, detail="Library not found")
    return library

@app.put("/libraries/{library_id}", response_model=Library)
async def update_library(library_id: str, update_data: Dict[str, Dict[str, str]]):
    metadata = update_data.get("metadata", {})
    library = db_service.update_library(library_id, metadata)
    if not library:
        raise HTTPException(status_code=404, detail="Library not found")
    return library

@app.delete("/libraries/{library_id}")
async def delete_library(library_id: str):
    if not db_service.delete_library(library_id):
        raise HTTPException(status_code=404, detail="Library not found")
    return {"message": "Library deleted"}

@app.post("/libraries/{library_id}/documents/", response_model=Document)
async def add_document(library_id: str, document: Document):
    doc = db_service.add_document(library_id, document)
    if not doc:
        raise HTTPException(status_code=404, detail="Library not found")
    return doc

@app.get("/libraries/{library_id}/documents/{document_id}", response_model=Document)
async def get_document(library_id: str, document_id: str):
    doc = db_service.get_document(library_id, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc

@app.put("/libraries/{library_id}/documents/{document_id}", response_model=Document)
async def update_document(library_id: str, document_id: str, update_data: Dict[str, Dict[str, str]]):
    metadata = update_data.get("metadata", {})
    doc = db_service.update_document(library_id, document_id, metadata)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc

@app.delete("/libraries/{library_id}/documents/{document_id}")
async def delete_document(library_id: str, document_id: str):
    if not db_service.delete_document(library_id, document_id):
        raise HTTPException(status_code=404, detail="Document not found")
    return {"message": "Document deleted"}

@app.post("/libraries/{library_id}/documents/{document_id}/chunks/", response_model=Chunk)
async def add_chunk(library_id: str, document_id: str, chunk: Chunk):
    chunk = db_service.add_chunk(library_id, document_id, chunk)
    if not chunk:
        raise HTTPException(status_code=404, detail="Library or Document not found")
    return chunk

@app.get("/libraries/{library_id}/documents/{document_id}/chunks/{chunk_id}", response_model=Chunk)
async def get_chunk(library_id: str, document_id: str, chunk_id: str):
    chunk = db_service.get_chunk(library_id, document_id, chunk_id)
    if not chunk:
        raise HTTPException(status_code=404, detail="Chunk not found")
    return chunk

@app.put("/libraries/{library_id}/documents/{document_id}/chunks/{chunk_id}", response_model=Chunk)
async def update_chunk(library_id: str, document_id: str, chunk_id: str, chunk: Chunk):
    chunk = db_service.update_chunk(library_id, document_id, chunk_id, chunk.text, chunk.embedding)
    if not chunk:
        raise HTTPException(status_code=404, detail="Chunk not found")
    return chunk

@app.delete("/libraries/{library_id}/documents/{document_id}/chunks/{chunk_id}")
async def delete_chunk(library_id: str, document_id: str, chunk_id: str):
    if not db_service.delete_chunk(library_id, document_id, chunk_id):
        raise HTTPException(status_code=404, detail="Chunk not found")
    return {"message": "Chunk deleted"}

@app.post("/libraries/{library_id}/search/")
async def search(library_id: str, query: Dict[str, List[float]], k: int = 5):
    if "embedding" not in query:
        raise HTTPException(status_code=400, detail="Query embedding required")
    results = db_service.search(library_id, query["embedding"], k)
    return results

