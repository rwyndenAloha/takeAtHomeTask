from pydantic import BaseModel, Field, field_serializer
from typing import List, Dict, Optional
from datetime import datetime
import uuid

class Chunk(BaseModel):
    id: str
    text: str
    embedding: List[float]
    metadata: Dict[str, str] = {}
    created_at: datetime

    @field_serializer('created_at')
    def serialize_created_at(self, created_at: datetime, _info):
        return created_at.isoformat()

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
    created_at: datetime

    @field_serializer('created_at')
    def serialize_created_at(self, created_at: datetime, _info):
        return created_at.isoformat()

class DocumentCreate(BaseModel):
    chunks: List[ChunkCreate] = []
    metadata: Dict[str, str] = {}

class Library(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    documents: List[Document] = []
    metadata: Dict[str, str] = {}
    created_at: datetime

    @field_serializer('created_at')
    def serialize_created_at(self, created_at: datetime, _info):
        return created_at.isoformat()

class LibraryCreate(BaseModel):
    documents: List[DocumentCreate] = []
    metadata: Dict[str, str] = {}
