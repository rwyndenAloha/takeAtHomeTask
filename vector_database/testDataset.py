from services import VectorDBService
from datetime import datetime
db = VectorDBService("test", "default")
print(f"Libraries: {len(db.libraries)}")
for lib_id, library in db.libraries.items():
    print(f"Library {lib_id}: {len(library.documents)} documents")
    for doc in library.documents:
        print(f"  Document {doc.id}: {len(doc.chunks)} chunks")
        for chunk in doc.chunks:
            print(f"    Chunk {chunk.id}: embedding={len(chunk.embedding) if chunk.embedding else 'None'}, created_at={chunk.created_at}, metadata={chunk.metadata}")
print("Name Index:", {name: len(chunks) for name, chunks in db.name_index.items()})

