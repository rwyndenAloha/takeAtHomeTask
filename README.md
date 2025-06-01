To run define: export COHERE_API_KEY='A1XXXXXXXXXXX'

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

Extra:

7. See query.py (comments at top also)
The cosine similarity in query.py adds client-side semantic search to the system complementing the serverside vector search. Uses Cohere as well for offline searching ability.

To query serverside (original requirement just run testSearch.sh.

Note: See movie for walkthrough

Table of Contents
testSearch.sh: Outputs JSON (e.g., [{"text":"Sample text",...}]).
embed.py: Generates embeddings, saves to embedded_output.json & embeddings.npy.
Prototype scripts created as examples that work:
search.py: functional as a test for Cohere embeddings
query.py: Queries embeddings with cosine similarity. Adds a client side query capability to the system
Other test scripts TBD
requirements.txt: Includes cohere, numpy, possibly requests.

CONCURRENCY HANDLING

Fulfills Requirement: The current code fully meets the requirement, as:
No Data Races: threading.RLock in BallTreeIndex (indexing.py) and VectorDBService (services.py) ensures thread-safe reads/writes to self.libraries and self.indexes. This protects against races in single-threaded asyncio (coroutine concurrency) or potential multi-threaded/worker setups.

Data Structures/Algorithms: Custom BallTreeIndex (O(log n * d) search), FlatIndex (O(n*d)), and dictionary (self.libraries) are thread-safe with RLock, suitable for the system’s scale.

Design Choices:
RLock: Ensures safety with recursive locking, balancing performance and consistency.
Custom BallTreeIndex: Optimizes search for small datasets, with locks for concurrency.
In-Memory Storage: Fast but ephemeral, sufficient for testing.

SERVICES DECOUPLING

Implementation:
VectorDBService: Encapsulates all CRUD logic (create_library, add_document, search, etc.), separate from API endpoints.

main.py: Endpoints call VectorDBService methods via dependency injection

async def create_library(library: LibraryCreate, db: VectorDBService = Depends(get_db)): return db.create_library(...)

Decoupling: API endpoints handle HTTP concerns (request validation, response formatting), while VectorDBService manages data operations, achieving clear separation.

Decoupling Fully fulfilled, as all implemented endpoints use VectorDBService, and services.py contains the business logic.

CRUD implemented with Create/Read/Update/Delete: Fully implemented with endpoints, Service methods, and tests

Thanks!
Rob

