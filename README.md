# takeAtHomeTask overview info

Note: See movie for overview

Table of Contents
testSearch.sh: Outputs JSON (e.g., [{"text":"Sample text",...}]).

embed.py: Generates embeddings, saves to embedded_output.json & embeddings.npy.

Prototype scripts created as examples that work:

search.py: functional as a test for Cohere embeddings
query.py: Queries embeddings with cosine similarity. Adds a client side query capability to the system

Other test scripts TBD

requirements.txt: Includes cohere, numpy, possibly requests.

CONCURRENCY HANDLING:

Fulfills Requirement: The current code fully meets the requirement, as:
No Data Races: threading.RLock in BallTreeIndex (indexing.py) and VectorDBService (services.py) ensures thread-safe reads/writes to self.libraries and self.indexes. This protects against races in single-threaded asyncio (coroutine concurrency) or potential multi-threaded/worker setups.

Data Structures/Algorithms: Custom BallTreeIndex (O(log n * d) search), FlatIndex (O(n*d)), and dictionary (self.libraries) are thread-safe with RLock, suitable for the system’s scale.

Design Choices:
RLock: Ensures safety with recursive locking, balancing performance and consistency.
Custom BallTreeIndex: Optimizes search for small datasets, with locks for concurrency.
In-Memory Storage: Fast but ephemeral, sufficient for testing.

