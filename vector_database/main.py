from fastapi import FastAPI, Depends, HTTPException, Request
from typing import List, Dict
from uuid import uuid4
from datetime import datetime, timezone
from pydantic import BaseModel
import numpy as np
from services import VectorDBService
from models import Library, LibraryCreate, Document, DocumentCreate, Chunk, ChunkCreate, ChunkUpdate
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(root_path="/api/v1")  # Add API prefix
db = VectorDBService()  # Singleton instance

def get_db():
    return db

@app.post("/libraries/", response_model=Library)
async def create_library(request: Request, library: LibraryCreate, db: VectorDBService = Depends(get_db)):
    raw_body = await request.body()
    logger.debug(f"Raw create_library request body: {raw_body.decode()}")
    logger.debug(f"Parsed create_library request: {library.dict()}")
    documents = []
    for doc in library.documents:
        chunks = []
        for chunk in doc.chunks:
            # Validate embedding
            if chunk.embedding:
                try:
                    embedding = [float(x) for x in chunk.embedding]
                    logger.debug(f"Validated embedding for chunk: {embedding}")
                except (ValueError, TypeError) as e:
                    logger.error(f"Invalid embedding in chunk: {chunk.embedding}, error: {e}")
                    raise HTTPException(status_code=400, detail=f"Invalid embedding in chunk: {chunk.embedding}")
            else:
                embedding = []
            chunks.append(Chunk(
                id=str(uuid4()),
                text=chunk.text,
                embedding=embedding,
                metadata=chunk.metadata,
                created_at=datetime.now(timezone.utc)  # Use datetime object
            ))
        documents.append(Document(
            id=str(uuid4()),
            chunks=chunks,
            metadata=doc.metadata,
            created_at=datetime.now(timezone.utc)
        ))
    return db.create_library(Library(
        id=str(uuid4()),
        documents=documents,
        metadata=library.metadata,
        created_at=datetime.now(timezone.utc)
    ))

@app.get("/libraries/", response_model=List[Dict])
async def get_libraries(db: VectorDBService = Depends(get_db)):
    logger.debug("Received get_libraries request")
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
        raise HTTPException(status_code=400, detail="Library not found")
    return library

@app.delete("/libraries/{library_id}")
async def delete_library(library_id: str, db: VectorDBService = Depends(get_db)):
    if db.delete_library(library_id):
        return {"message": "Library deleted"}
    raise HTTPException(status_code=404, detail="Library not found")

@app.post("/libraries/{library_id}/documents/", response_model=Document)
async def add_document(request: Request, library_id: str, document: DocumentCreate, db: VectorDBService = Depends(get_db)):
    raw_body = await request.body()
    logger.debug(f"Raw add_document request body: {raw_body.decode()}")
    logger.debug(f"Parsed add_document request for library {library_id}: {document.dict()}")
    chunks = []
    for chunk in document.chunks:
        # Validate embedding
        if chunk.embedding:
            try:
                embedding = [float(x) for x in chunk.embedding]
                logger.debug(f"Validated embedding for chunk: {embedding}")
            except (ValueError, TypeError) as e:
                logger.error(f"Invalid embedding in chunk: {chunk.embedding}, error: {e}")
                raise HTTPException(status_code=400, detail=f"Invalid embedding in chunk: {chunk.embedding}")
        else:
            embedding = []
        chunks.append(Chunk(
            id=str(uuid4()),
            text=chunk.text,
            embedding=embedding,
            metadata=chunk.metadata,
            created_at=datetime.now(timezone.utc)
        ))
    new_document = db.add_document(
        library_id,
        Document(
            id=str(uuid4()),
            chunks=chunks,
            metadata=document.metadata,
            created_at=datetime.now(timezone.utc)
        )
    )
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
async def add_chunk(request: Request, library_id: str, document_id: str, chunk: ChunkCreate, db: VectorDBService = Depends(get_db)):
    raw_body = await request.body()
    logger.debug(f"Raw add_chunk request body: {raw_body.decode()}")
    logger.debug(f"Parsed add_chunk request for document {document_id}: {chunk.dict()}")
    # Validate embedding
    if chunk.embedding:
        try:
            embedding = [float(x) for x in chunk.embedding]
            logger.debug(f"Validated embedding for chunk: {embedding}")
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid embedding in chunk: {chunk.embedding}, error: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid embedding in chunk: {chunk.embedding}")
    else:
        embedding = []
    new_chunk = db.add_chunk(
        library_id,
        document_id,
        Chunk(
            id=str(uuid4()),
            text=chunk.text,
            embedding=embedding,
            metadata=chunk.metadata,
            created_at=datetime.now(timezone.utc)
        )
    )
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
async def update_chunk(request: Request, library_id: str, document_id: str, chunk_id: str, chunk: ChunkUpdate, db: VectorDBService = Depends(get_db)):
    raw_body = await request.body()
    logger.debug(f"Raw update_chunk request body: {raw_body.decode()}")
    logger.debug(f"Parsed update_chunk request for chunk {chunk_id}: {chunk.dict()}")
    # Validate embedding
    if chunk.embedding:
        try:
            embedding = [float(x) for x in chunk.embedding]
            logger.debug(f"Validated embedding for chunk: {embedding}")
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid embedding in chunk: {chunk.embedding}, error: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid embedding in chunk: {chunk.embedding}")
    else:
        embedding = []
    updated_chunk = db.update_chunk(
        library_id,
        document_id,
        chunk_id,
        chunk.text or "",
        embedding
    )
    if not updated_chunk:
        raise HTTPException(status_code=404, detail="Chunk not found")
    return updated_chunk

@app.delete("/libraries/{library_id}/documents/{document_id}/chunks/{chunk_id}")
async def delete_chunk(library_id: str, document_id: str, chunk_id: str, db: VectorDBService = Depends(get_db)):
    if db.delete_chunk(library_id, document_id, chunk_id):
        return {"message": "Chunk deleted"}
    raise HTTPException(status_code=404, detail="Chunk not found")

@app.post("/libraries/{library_id}/search/")
async def search(request: Request, library_id: str, query: Dict[str, List[float]], db: VectorDBService = Depends(get_db)):
    raw_body = await request.body()
    logger.debug(f"Raw search request body: {raw_body.decode()}")
    logger.debug(f"Parsed search request for library {library_id}: {query}")
    if "embedding" not in query:
        raise HTTPException(status_code=400, detail="Query embedding required")
    # Validate query embedding
    try:
        embedding = [float(x) for x in query["embedding"]]
        logger.debug(f"Validated query embedding: {embedding}")
    except (ValueError, TypeError) as e:
        logger.error(f"Invalid query embedding: {query['embedding']}, error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid query embedding: {query['embedding']}")
    results = db.search(library_id, embedding)
    return results

# New endpoint for searching chunks across all libraries
class ChunkSearchRequest(BaseModel):
    query_embedding: List[float]
    start_date: str
    name_contains: str

    def __init__(self, **data):
        super().__init__(**data)
        # Convert query_embedding elements to float if they are strings
        try:
            self.query_embedding = [float(x) for x in self.query_embedding]
            logger.debug(f"Validated query_embedding in ChunkSearchRequest: {self.query_embedding}")
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid query_embedding: {self.query_embedding}, error: {e}")
            raise ValueError(f"Invalid query embedding: {self.query_embedding}")

class ChunkSearchResult(BaseModel):
    library_id: str
    document_id: str
    chunk_id: str
    text: str
    metadata: dict
    created_at: str
    similarity: float

@app.post("/chunks/search/", response_model=List[ChunkSearchResult])
async def search_chunks(request: Request, body: ChunkSearchRequest, db: VectorDBService = Depends(get_db)):
    raw_body = await request.body()
    logger.debug(f"Raw chunk search request body: {raw_body.decode()}")
    logger.debug(f"Parsed chunk search request: query_embedding={body.query_embedding}, start_date={body.start_date}, name_contains={body.name_contains}")
    try:
        # Parse start_date
        try:
            start_date = datetime.fromisoformat(body.start_date.replace("Z", "+00:00"))
        except ValueError:
            logger.error(f"Invalid date format: {body.start_date}")
            raise HTTPException(status_code=400, detail="Invalid date format. Use ISO 8601 (e.g., 2025-06-01T00:00:00Z)")

        # Convert query embedding to numpy array
        try:
            query_embedding = np.array(body.query_embedding, dtype=np.float64)
            logger.debug(f"Converted query_embedding to numpy array: {query_embedding}")
        except Exception as e:
            logger.error(f"Failed to convert query_embedding to numpy array: {body.query_embedding}, error: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid query embedding: {body.query_embedding}")

        # Search chunks using VectorDBService
        results = db.search_chunks_after_date(
            query_embedding=query_embedding,
            start_date=start_date,
            name_contains=body.name_contains.lower()
        )

        logger.debug(f"Returning {len(results)} search results")
        return results

    except ValueError as ve:
        logger.error(f"ValueError in search_chunks: {str(ve)}")
        raise HTTPException(status_code=400, detail=f"Invalid query embedding: {str(ve)}")
    except Exception as e:
        logger.error(f"Exception in search_chunks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

