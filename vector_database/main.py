from fastapi import FastAPI, Depends, HTTPException
from typing import List, Dict
from uuid import uuid4
from datetime import datetime, timezone
from services import VectorDBService
from models import Library, LibraryCreate, Document, DocumentCreate, Chunk, ChunkCreate, ChunkUpdate

app = FastAPI()

def get_db():
    return VectorDBService()

@app.post("/libraries/", response_model=Library)
async def create_library(library: LibraryCreate, db: VectorDBService = Depends(get_db)):
    return db.create_library(Library(id=str(uuid4()), documents=library.documents, metadata=library.metadata, created_at=datetime.now(timezone.utc)))

@app.get("/libraries/", response_model=List[Dict])
async def get_libraries(db: VectorDBService = Depends(get_db)):
    return db.get_libraries()

@app.get("/libraries/{library_id}", response_model=Library)
async def get_library(library_id: str, db: VectorDBService = Depends(get_db)):
    library = db.get_library(library_id)
    if not library:
        raise HTTPException(status_code=404, detail="Library not found")
    return library

@app.put("/libraries/{library_id}", response_model=Library)
async def update_library(library_id: str, metadata: Dict[str, str], db: VectorDBService = Depends(get_db)):
    library = db.update_library(library_id, metadata)
    if not library:
        raise HTTPException(status_code=404, detail="Library not found")
    return library

@app.delete("/libraries/{library_id}")
async def delete_library(library_id: str, db: VectorDBService = Depends(get_db)):
    if db.delete_library(library_id):
        return {"message": "Library deleted"}
    raise HTTPException(status_code=404, detail="Library not found")

@app.post("/libraries/{library_id}/documents/", response_model=Document)
async def add_document(library_id: str, document: DocumentCreate, db: VectorDBService = Depends(get_db)):
    new_document = db.add_document(library_id, Document(id=str(uuid4()), chunks=document.chunks, metadata=document.metadata, created_at=datetime.now(timezone.utc)))
    if not new_document:
        raise HTTPException(status_code=404, detail="Library not found")
    return new_document

@app.get("/libraries/{library_id}/documents/{document_id}", response_model=Document)
async def get_document(library_id: str, document_id: str, db: VectorDBService = Depends(get_db)):
    document = db.get_document(library_id, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document

@app.put("/libraries/{library_id}/documents/{document_id}", response_model=Document)
async def update_document(library_id: str, document_id: str, metadata: Dict[str, str], db: VectorDBService = Depends(get_db)):
    document = db.update_document(library_id, document_id, metadata)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document

@app.delete("/libraries/{library_id}/documents/{document_id}")
async def delete_document(library_id: str, document_id: str, db: VectorDBService = Depends(get_db)):
    if db.delete_document(library_id, document_id):
        return {"message": "Document deleted"}
    raise HTTPException(status_code=404, detail="Document not found")

@app.post("/libraries/{library_id}/documents/{document_id}/chunks/", response_model=Chunk)
async def add_chunk(library_id: str, document_id: str, chunk: ChunkCreate, db: VectorDBService = Depends(get_db)):
    new_chunk = db.add_chunk(library_id, document_id, Chunk(id=str(uuid4()), text=chunk.text, embedding=chunk.embedding, metadata=chunk.metadata, created_at=datetime.now(timezone.utc)))
    if not new_chunk:
        raise HTTPException(status_code=404, detail="Library or document not found")
    return new_chunk

@app.get("/libraries/{library_id}/documents/{document_id}/chunks/{chunk_id}", response_model=Chunk)
async def get_chunk(library_id: str, document_id: str, chunk_id: str, db: VectorDBService = Depends(get_db)):
    chunk = db.get_chunk(library_id, document_id, chunk_id)
    if not chunk:
        raise HTTPException(status_code=404, detail="Chunk not found")
    return chunk

@app.put("/libraries/{library_id}/documents/{document_id}/chunks/{chunk_id}", response_model=Chunk)
async def update_chunk(library_id: str, document_id: str, chunk_id: str, chunk: ChunkUpdate, db: VectorDBService = Depends(get_db)):
    updated_chunk = db.update_chunk(library_id, document_id, chunk_id, chunk.text, chunk.embedding)
    if not updated_chunk:
        raise HTTPException(status_code=404, detail="Chunk not found")
    return updated_chunk

@app.delete("/libraries/{library_id}/documents/{document_id}/chunks/{chunk_id}")
async def delete_chunk(library_id: str, document_id: str, chunk_id: str, db: VectorDBService = Depends(get_db)):
    if db.delete_chunk(library_id, document_id, chunk_id):
        return {"message": "Chunk deleted"}
    raise HTTPException(status_code=404, detail="Chunk not found")

@app.post("/libraries/{library_id}/search/")
async def search(library_id: str, query: Dict[str, List[float]], db: VectorDBService = Depends(get_db)):
    if "embedding" not in query:
        raise HTTPException(status_code=400, detail="Query embedding required")
    results = db.search(library_id, query["embedding"])
    return results
