from fastapi import FastAPI, Depends, HTTPException, Request
from typing import List, Dict
from uuid import uuid4
from datetime import datetime, timezone
from pydantic import BaseModel
import numpy as np
from services import VectorDBService
from models import Library, LibraryCreate, Document, DocumentCreate, Chunk, ChunkCreate, ChunkUpdate, ChunkSearchResult
from kubernetes import config
import logging
import os

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(root_path="/api/v1")
db = VectorDBService(pod_name=os.getenv("POD_NAME", "vector-db-0"), namespace=os.getenv("NAMESPACE", "default"))

def get_db():
    return db

runningInKubes = False
try:
    config.load_incluster_config()
    runningInKubes = True
except:
    # Do nothing this means we're testing in docker without kubes
    runningInKubes = False
    pass

@app.post("/libraries/", response_model=Library)
async def create_library(request: Request, library: LibraryCreate, db: VectorDBService = Depends(get_db)):
    if (not db.is_leader and runningInKubes):
        raise HTTPException(status_code=403, detail="Write operation not allowed on follower")
    raw_body = await request.body()
    logger.debug(f"Raw create_library request body: {raw_body.decode()}")
    documents = []
    for doc in library.documents:
        chunks = []
        for chunk in doc.chunks:
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
    if (not db.is_leader and runningInKubes):
        raise HTTPException(status_code=403, detail="Write operation not allowed on follower")
    library = db.update_library(library_id, metadata)
    if not library:
        raise HTTPException(status_code=404, detail="Library not found")
    return library

@app.delete("/libraries/{library_id}")
async def delete_library(library_id: str, db: VectorDBService = Depends(get_db)):
    if (not db.is_leader and runningInKubes):
        raise HTTPException(status_code=403, detail="Write operation not allowed on follower")
    if db.delete_library(library_id):
        return {"message": "Library deleted"}
    raise HTTPException(status_code=404, detail="Library not found")

@app.post("/libraries/{library_id}/documents/", response_model=Document)
async def add_document(request: Request, library_id: str, document: DocumentCreate, db: VectorDBService = Depends(get_db)):
    if (not db.is_leader and runningInKubes):
        raise HTTPException(status_code=403, detail="Write operation not allowed on follower")
    raw_body = await request.body()
    logger.debug(f"Raw add_document request body: {raw_body.decode()}")
    chunks = []
    for chunk in document.chunks:
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
    if (not db.is_leader and runningInKubes):
        raise HTTPException(status_code=403, detail="Write operation not allowed on follower")
    document = db.update_document(library_id, document_id, metadata)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document

@app.delete("/libraries/{library_id}/documents/{document_id}")
async def delete_document(library_id: str, document_id: str, db: VectorDBService = Depends(get_db)):
    if (not db.is_leader and runningInKubes):
        raise HTTPException(status_code=403, detail="Write operation not allowed on follower")
    if db.delete_document(library_id, document_id):
        return {"message": "Document deleted"}
    raise HTTPException(status_code=404, detail="Document not found")

@app.post("/libraries/{library_id}/documents/{document_id}/chunks/", response_model=Chunk)
async def add_chunk(request: Request, library_id: str, document_id: str, chunk: ChunkCreate, db: VectorDBService = Depends(get_db)):
    if (not db.is_leader and runningInKubes):
        raise HTTPException(status_code=403, detail="Write operation not allowed on follower")
    raw_body = await request.body()
    logger.debug(f"Raw add_chunk request body: {raw_body.decode()}")
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
    if (not db.is_leader and runningInKubes):
        raise HTTPException(status_code=403, detail="Write operation not allowed on follower")
    raw_body = await request.body()
    logger.debug(f"Raw update_chunk request body: {raw_body.decode()}")
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
    if (not db.is_leader and runningInKubes):
        raise HTTPException(status_code=403, detail="Write operation not allowed on follower")
    if db.delete_chunk(library_id, document_id, chunk_id):
        return {"message": "Chunk deleted"}
    raise HTTPException(status_code=404, detail="Chunk not found")

@app.post("/libraries/{library_id}/search/")
async def search(request: Request, library_id: str, query: Dict[str, List[float]], db: VectorDBService = Depends(get_db)):
    raw_body = await request.body()
    logger.debug(f"Raw search request body: {raw_body.decode()}")
    if "embedding" not in query:
        raise HTTPException(status_code=400, detail="Query embedding required")
    try:
        embedding = [float(x) for x in query["embedding"]]
        logger.debug(f"Validated query embedding: {embedding}")
    except (ValueError, TypeError) as e:
        logger.error(f"Invalid query embedding: {query['embedding']}, error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid query embedding: {query['embedding']}")
    results = db.search(library_id, embedding)
    return results

class ChunkSearchRequest(BaseModel):
    query_embedding: List[float]
    start_date: str
    name_contains: str

    def __init__(self, **data):
        super().__init__(**data)
        try:
            self.query_embedding = [float(x) for x in self.query_embedding]
            logger.debug(f"Validated query_embedding in ChunkSearchRequest: {self.query_embedding}")
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid query_embedding: {self.query_embedding}, error: {e}")
            raise ValueError(f"Invalid query embedding: {self.query_embedding}")

@app.post("/chunks/search/", response_model=List[ChunkSearchResult])
async def search_chunks(request: Request, body: ChunkSearchRequest, db: VectorDBService = Depends(get_db)):
    raw_body = await request.body()
    logger.debug(f"Raw chunk search request body: {raw_body.decode()}")
    try:
        if body.query_embedding is None or not body.query_embedding:
            logger.error("Query embedding is None or empty")
            raise HTTPException(status_code=400, detail="Query embedding cannot be None or empty")
        try:
            start_date = datetime.fromisoformat(body.start_date.replace("Z", "+00:00"))
        except ValueError:
            logger.error(f"Invalid date format: {body.start_date}")
            raise HTTPException(status_code=400, detail="Invalid date format. Use ISO 8601")
        try:
            query_embedding = np.array(body.query_embedding, dtype=np.float32)
            logger.debug(f"Converted query_embedding to numpy array: {query_embedding}")
        except Exception as e:
            logger.error(f"Failed to convert query_embedding: {body.query_embedding}, error: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid query embedding: {str(e)}")
        results = db.search_chunks_after_date(
            query_embedding=query_embedding,
            start_date=start_date,
            name_contains=body.name_contains.lower() if body.name_contains else ""
        )
        logger.debug(f"Returning {len(results)} search results")
        return results
    except ValueError as ve:
        logger.error(f"ValueError in search_chunks: {str(ve)}")
        raise HTTPException(status_code=400, detail=f"Invalid query embedding: {str(ve)}")
    except Exception as e:
        logger.error(f"Exception in search_chunks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/replicate/")
async def replicate_event(event: Dict, db: VectorDBService = Depends(get_db)):
    logger.debug(f"Received replication event: {event}")
    try:
        db.apply_event(event)
        return {"message": "Event applied"}
    except Exception as e:
        logger.error(f"Failed to apply replication event: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to apply event: {str(e)}")

