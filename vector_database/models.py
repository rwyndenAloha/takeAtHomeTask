from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime
import uuid

class Chunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    embedding: List[float]
    metadata: Dict[str, str] = {}
    created_at: datetime = Field(default_factory=datetime.utcnow)

class ChunkCreate(BaseModel):
    text: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, str] = {}

class ChunkUpdate(BaseModel):
    text: Optional[str] = None
    embedding: Optional[List[float]] = None

class Document(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    chunks: List[Chunk] = []
    metadata: Dict[str, str] = {}
    created_at: datetime = Field(default_factory=datetime.utcnow)

class DocumentCreate(BaseModel):
    chunks: List[ChunkCreate] = []
    metadata: Dict[str, str] = {}

class Library(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    documents: List[Document] = []
    metadata: Dict[str, str] = {}
    created_at: datetime = Field(default_factory=datetime.utcnow)

class LibraryCreate(BaseModel):
    documents: List[DocumentCreate] = []
    metadata: Dict[str, str] = {}
