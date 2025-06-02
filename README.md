To run define: export COHERE_API_KEY='A1XXXXXXXXXXX'
To run docker cd vector_database and ./run.sh

Requirements Fulfilled:

1. Chunk, Doc and Library classes are defined in vector_database/models.py
2. FlatIndex and BallTree index are defined in vector_database/indexing.py
   (see comment at the top of that file for an analysis of them).
   Note: The FlatIndex is custom coded from scratch (not using scikit.learn)
3. This repo provides concurrency (thread safety).
   (see top of vector_database/indexing.py and vector_database/services.py)
   (also see "CONCURRENCY HANDLING summary below in this file)
4. See vector_database/services.py for decoupling API endpoints.
   CRUD fully implemented. See "SERVICES DECOUPLING summary below in this file
5. Endpoints are defined in vector_database/main.py
6. See vector_database/DOCKERFILE for image def

Quick test of server side search:
  a) ./testCreateLibrary.sh
  b) ./testLibrList.sh (you will see your new library)
  c) ./testSearch.sh (just enter "Sample")

Extra:

7. Metadata filtering is included ... see testMetadataFiltering.sh

ExtraExtra:

8. Added client-side query: See queryFromClient.py (comments at top also)
The cosine similarity in queryFromClient.py adds client-side semantic search to the system complementing the serverside vector search. Uses Cohere as well for offline searching ability.  Gives it ability to run offline...

Quick test of this off-line search: type: "python3 queryFromClient.py"
  (and use "Sample")

NOTE: See movie for walkthrough
NOTE: I also have a programming handle 'jojapoppa' with another repo.  Ask and I'll give you that path too...

TABLE OF CONTENTS
├── embedded_output.json
├── embeddings.npy
├── embed.py Generates embeddings to embedded_output.json & embeddings.npy.
├── LICENSE
├── queryFromClient.py Adds client-side query with cosine similarity
├── query_results.json
├── README.md
├── search.py
├── sshVMUbuntu.sh
├── testChunkCRUD.sh
├── prepCohereOnClient.sh
├── testCreateLibrary.sh
├── testDeleteLibrary.sh
├── testDocumentCRUD.sh
├── testLibrList.sh
├── testMsgBodyParam.sh
├── testSearch.sh   Outputs JSON (e.g., [{"text":"Sample text",...}])
├── testUpdateLibraryMetadata.sh
└── vector_database
    ├── Dockerfile
    ├── get-docker.sh
    ├── indexing.py        server indexing (fully coded - no libraries)
    ├── main.py            server endpoints
    ├── models.py          server classes
    ├── requirements.txt
    ├── run.sh
    ├── services.py        server services decoupling 
    └── stop.sh

CONCURRENCY HANDLING

 a) Fulfills Requirement: The current code fully meets the requirement, as:
No Data Races: threading.RLock in BallTreeIndex (indexing.py) and VectorDBService (services.py) ensures thread-safe reads/writes to self.libraries and self.indexes. This protects against races in single-threaded asyncio (coroutine concurrency) or potential multi-threaded/worker setups.

 b) Data Structures/Algorithms: Custom BallTreeIndex (O(log n * d) search), FlatIndex (O(n*d)), and dictionary (self.libraries) are thread-safe with RLock, suitable for the system’s scale.

 c) RLock: Ensures safety with recursive locking, balancing performance and consistency.

 d) Custom BallTreeIndex: Optimizes search for small datasets, with locks for concurrency.

 e) In-Memory Storage: Fast but ephemeral, sufficient for testing.

SERVICES DECOUPLING

 a) VectorDBService: Encapsulates all CRUD logic (create_library, add_document, search, etc.), separate from API endpoints.

 b) main.py: Endpoints call VectorDBService methods via dependency injection

 c) async def create_library(library: LibraryCreate, db: VectorDBService = Depends(get_db)): return db.create_library(...)

 d) Decoupling: API endpoints handle HTTP concerns (request validation, response formatting), while VectorDBService manages data operations, achieving clear separation.

 e) Decoupling Fully fulfilled, as all implemented endpoints use VectorDBService, and services.py contains the business logic.

 f) CRUD implemented with Create/Read/Update/Delete: Fully implemented with endpoints, Service methods, and tests

Thanks!
Rob Wynden, PhD

