"""
Concurrency handling

VectorDBService: self.lock (RLock) wraps all methods, ensuring thread-safe access to self.libraries and self.indexes.

Overview of thread safety

VectorDBService uses RLock for all methods, protecting self.libraries (dictionary updates, list appends) and self.indexes (index creation, updates, queries). This covers:
Reads: get_library, search (accessing self.libraries, calling index.search).
Writes: create_library, add_document, update_library (modifying self.libraries, self.indexes).

Asyncio Safety: RLock is thread-safe, and since FastAPI uses asyncio, the lock ensures coroutines don’t race (e.g., concurrent POST /documents/ and POST /search/).

Persistence: Saves self.libraries to vector_db_state.json in a PersistentVolume.
Leader Election: Uses Kubernetes Lease for leader selection.
Replication: Leader sends write events to followers via HTTP /replicate/.
Failover: Followers monitor lease and promote to leader if needed.

Thread Safety: self.lock (RLock) ensures safe access to self.libraries and self.indexes..
"""

from typing import List, Optional, Dict
import threading
import numpy as np
from models import Library, Document, Chunk, ChunkSearchResult
from indexing import FlatIndex, BallTreeIndex
from datetime import datetime
import json
import os
import logging
import requests
from kubernetes import client, config
from kubernetes.client.rest import ApiException
import time

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

runningInKubes = False

class VectorDBService:
    def __init__(self, pod_name: str, namespace: str = "default"):
        self.libraries: Dict[str, Library] = {}
        self.indexes: Dict[str, BallTreeIndex] = {}  # Per-library indexes
        self.global_index = BallTreeIndex()  # For cross-library searches
        self.name_index = {}  # Map name to list of (library_id, chunk_id)
        self.lock = threading.RLock()
        self.pod_name = pod_name
        self.namespace = namespace
        self.is_leader = False
        self.leader_lease_name = "vector-db-leader"
        self.state_file = f"/data/{pod_name}/vector_db_state.json"
        self.event_log = []  # In-memory event log for replication
        self.followers = []  # List of follower pod URLs
        self.pending_events = []  # Batch replication events

        # Create data directory
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)

        # Load state from disk
        self.load_state()

        # Start leader election in a separate thread if in Kubernetes
        try:
            config.load_incluster_config()
            global runningInKubes
            runningInKubes = True
            threading.Thread(target=self.run_leader_election, daemon=True).start()
        except:
            logger.info("Not running in Kubernetes, skipping leader election")
            self.is_leader = True

    def run_leader_election(self):
        try:
            config.load_incluster_config()
        except:
            config.load_kube_config()
        coordination_api = client.CoordinationV1Api()
        lease = client.V1Lease(
            metadata=client.V1ObjectMeta(name=self.leader_lease_name, namespace=self.namespace),
            spec=client.V1LeaseSpec(
                holder_identity=self.pod_name,
                lease_duration_seconds=15,
                renew_time=datetime.utcnow()
            )
        )

        while True:
            try:
                try:
                    current_lease = coordination_api.read_namespaced_lease(self.leader_lease_name, self.namespace)
                    if current_lease.spec.holder_identity == self.pod_name:
                        current_lease.spec.renew_time = datetime.utcnow()
                        coordination_api.replace_namespaced_lease(self.leader_lease_name, self.namespace, current_lease)
                        self.is_leader = True
                        self.update_followers()
                    elif (datetime.utcnow() - current_lease.spec.renew_time.replace(tzinfo=None)).total_seconds() > 15:
                        current_lease.spec.holder_identity = self.pod_name
                        current_lease.spec.renew_time = datetime.utcnow()
                        coordination_api.replace_namespaced_lease(self.leader_lease_name, self.namespace, current_lease)
                        self.is_leader = True
                        self.update_followers()
                    else:
                        self.is_leader = False
                except ApiException as e:
                    if e.status == 404:
                        coordination_api.create_namespaced_lease(self.namespace, lease)
                        self.is_leader = True
                        self.update_followers()
                    else:
                        logger.error(f"Lease error: {e}")
                time.sleep(5)
            except Exception as e:
                logger.error(f"Leader election error: {e}")
                time.sleep(5)

    def update_followers(self):
        if not self.is_leader and runningInKubes:
            self.followers = []
            return
        followers = []
        num_replicas = 3
        for i in range(num_replicas):
            pod_hostname = f"vector-db-{i}"
            if pod_hostname != self.pod_name:
                followers.append(f"http://{pod_hostname}.vector-db.default.svc.cluster.local:8000/api/v1/replicate/")
        self.followers = followers
        logger.debug(f"Updated followers: {self.followers}")

    def save_state(self):
        with self.lock:
            try:
                libraries_json = {k: v.dict(exclude_unset=True) for k, v in self.libraries.items()}
                with open(self.state_file, "w") as f:
                    json.dump(libraries_json, f, indent=2)
                logger.debug(f"Saved state to {self.state_file}")
            except Exception as e:
                logger.error(f"Failed to save state: {e}")

    def load_state(self):
        with self.lock:
            try:
                if os.path.exists(self.state_file):
                    with open(self.state_file, "r") as f:
                        data = json.load(f)
                    self.libraries = {k: Library.parse_obj(v) for k, v in data.items()}
                    chunk_count = 0
                    for library_id, library in self.libraries.items():
                        embeddings = []
                        chunk_ids = []
                        metadata = []
                        created_at = []
                        for doc in library.documents:
                            for chunk in doc.chunks:
                                chunk_count += 1
                                if chunk.embedding:
                                    embeddings.append(chunk.embedding)
                                    chunk_ids.append(chunk.id)
                                    metadata.append(chunk.metadata)
                                    created_at.append(chunk.created_at)
                                    if 'name' in chunk.metadata:
                                        name = chunk.metadata['name'].lower()
                                        if name not in self.name_index:
                                            self.name_index[name] = []
                                        self.name_index[name].append((library_id, chunk.id))
                        self.indexes[library_id] = BallTreeIndex() if embeddings else FlatIndex()
                        if embeddings:
                            self.indexes[library_id].batch_add(embeddings, chunk_ids, metadata, created_at)
                            self.global_index.batch_add(embeddings, chunk_ids, metadata, created_at)
                    logger.debug(f"Loaded state from {self.state_file}: {len(self.libraries)} libraries, {chunk_count} chunks, {len(self.name_index)} name_index entries")
            except Exception as e:
                logger.error(f"Failed to load state: {e}")

    def replicate_event(self, event: Dict):
        if not self.is_leader:
            return
        with self.lock:
            self.event_log.append(event)
            self.pending_events.append(event)
            if len(self.pending_events) >= 10:  # Batch replication
                for follower in self.followers:
                    try:
                        response = requests.post(follower, json={"events": self.pending_events}, timeout=5)
                        response.raise_for_status()
                    except requests.RequestException as e:
                        logger.error(f"Failed to replicate to {follower}: {e}")
                self.pending_events = []

    def apply_event(self, event: Dict):
        with self.lock:
            events = event.get("events", [event])
            for evt in events:
                action = evt.get("action")
                data = evt.get("data")
                if action == "create_library":
                    library = Library.parse_obj(data)
                    self.libraries[library.id] = library
                    embeddings = []
                    chunk_ids = []
                    metadata = []
                    created_at = []
                    for doc in library.documents:
                        for chunk in doc.chunks:
                            if chunk.embedding:
                                embeddings.append(chunk.embedding)
                                chunk_ids.append(chunk.id)
                                metadata.append(chunk.metadata)
                                created_at.append(chunk.created_at)
                    self.indexes[library.id] = BallTreeIndex() if embeddings else FlatIndex()
                    if embeddings:
                        self.indexes[library.id].batch_add(embeddings, chunk_ids, metadata, created_at)
                        self.global_index.batch_add(embeddings, chunk_ids, metadata, created_at)
                elif action == "add_document":
                    library_id = data["library_id"]
                    document = Document.parse_obj(data["document"])
                    if library_id in self.libraries:
                        self.libraries[library_id].documents.append(document)
                        embeddings = []
                        chunk_ids = []
                        metadata = []
                        created_at = []
                        for chunk in document.chunks:
                            if chunk.embedding:
                                embeddings.append(chunk.embedding)
                                chunk_ids.append(chunk.id)
                                metadata.append(chunk.metadata)
                                created_at.append(chunk.created_at)
                        if embeddings:
                            self.indexes[library_id].batch_add(embeddings, chunk_ids, metadata, created_at)
                            self.global_index.batch_add(embeddings, chunk_ids, metadata, created_at)
                elif action == "update_library":
                    library_id = data["library_id"]
                    metadata = data["metadata"]
                    if library_id in self.libraries:
                        self.libraries[library_id].metadata.update(metadata)
                elif action == "delete_library":
                    library_id = data["library_id"]
                    if library_id in self.libraries:
                        del self.libraries[library_id]
                        del self.indexes[library_id]
                        self.global_index = BallTreeIndex()  # Rebuild global index
                        for lib_id, lib in self.libraries.items():
                            embeddings = []
                            chunk_ids = []
                            metadata = []
                            created_at = []
                            for doc in lib.documents:
                                for chunk in doc.chunks:
                                    if chunk.embedding:
                                        embeddings.append(chunk.embedding)
                                        chunk_ids.append(chunk.id)
                                        metadata.append(chunk.metadata)
                                        created_at.append(chunk.created_at)
                            if embeddings:
                                self.global_index.batch_add(embeddings, chunk_ids, metadata, created_at)
                elif action == "add_chunk":
                    library_id = data["library_id"]
                    document_id = data["document_id"]
                    chunk = Chunk.parse_obj(data["chunk"])
                    if library_id in self.libraries:
                        for doc in self.libraries[library_id].documents:
                            if doc.id == document_id:
                                doc.chunks.append(chunk)
                                if chunk.embedding:
                                    self.indexes[library_id].add(chunk.embedding, chunk.id, chunk.metadata, chunk.created_at)
                                    self.global_index.add(chunk.embedding, chunk.id, chunk.metadata, chunk.created_at)
                                break
                # Add other actions (update_document, delete_document, update_chunk, delete_chunk) similarly
                self.save_state()

    def create_library(self, library: Library) -> Library:
        with self.lock:
            if not self.is_leader and runningInKubes:
                raise Exception("Write operation not allowed on follower")
            embeddings = []
            chunk_ids = []
            metadata = []
            created_at = []
            for doc in library.documents:
                for chunk in doc.chunks:
                    if chunk.embedding:
                        embeddings.append(chunk.embedding)
                        chunk_ids.append(chunk.id)
                        metadata.append(chunk.metadata)
                        created_at.append(chunk.created_at)
                        if 'name' in chunk.metadata:
                            name = chunk.metadata['name'].lower()
                            if name not in self.name_index:
                                self.name_index[name] = []
                            self.name_index[name].append((library.id, chunk.id))
            if embeddings and not all(len(e) == len(embeddings[0]) for e in embeddings):
                raise ValueError("All embeddings must have the same dimensionality")
            self.libraries[library.id] = library
            self.indexes[library.id] = BallTreeIndex() if embeddings else FlatIndex()
            if embeddings:
                self.indexes[library.id].batch_add(embeddings, chunk_ids, metadata, created_at)
                self.global_index.batch_add(embeddings, chunk_ids, metadata, created_at)
            self.save_state()
            self.replicate_event({"action": "create_library", "data": library.dict()})
            logger.info(f"Created library {library.id} with {len(embeddings)} embeddings")
            return library

    def get_library(self, library_id: str) -> Optional[Library]:
        with self.lock:
            return self.libraries.get(library_id)

    def get_libraries(self) -> List[dict]:
        with self.lock:
            return [{"id": k, "documents": v.documents, "metadata": v.metadata} for k, v in self.libraries.items()]

    def update_library(self, library_id: str, metadata: Dict[str, str]) -> Optional[Library]:
        with self.lock:
            if not self.is_leader and runningInKubes:
                raise Exception("Write operation not allowed on follower")
            if library_id in self.libraries:
                self.libraries[library_id].metadata.update(metadata)
                self.save_state()
                self.replicate_event({"action": "update_library", "data": {"library_id": library_id, "metadata": metadata}})
                return self.libraries[library_id]
            return None

    def delete_library(self, library_id: str) -> bool:
        with self.lock:
            if not self.is_leader and runningInKubes:
                raise Exception("Write operation not allowed on follower")
            if library_id in self.libraries:
                del self.libraries[library_id]
                del self.indexes[library_id]
                self.global_index = BallTreeIndex()
                for lib_id, lib in self.libraries.items():
                    embeddings = []
                    chunk_ids = []
                    metadata = []
                    created_at = []
                    for doc in lib.documents:
                        for chunk in doc.chunks:
                            if chunk.embedding:
                                embeddings.append(chunk.embedding)
                                chunk_ids.append(chunk.id)
                                metadata.append(chunk.metadata)
                                created_at.append(chunk.created_at)
                    if embeddings:
                        self.global_index.batch_add(embeddings, chunk_ids, metadata, created_at)
                self.save_state()
                self.replicate_event({"action": "delete_library", "data": {"library_id": library_id}})
                return True
            return False

    def add_document(self, library_id: str, document: Document) -> Optional[Document]:
        with self.lock:
            if not self.is_leader and runningInKubes:
                raise Exception("Write operation not allowed on follower")
            if library_id not in self.libraries:
                return None
            self.libraries[library_id].documents.append(document)
            embeddings = []
            chunk_ids = []
            metadata = []
            created_at = []
            for chunk in document.chunks:
                if chunk.embedding:
                    embeddings.append(chunk.embedding)
                    chunk_ids.append(chunk.id)
                    metadata.append(chunk.metadata)
                    created_at.append(chunk.created_at)
            if embeddings:
                self.indexes[library_id].batch_add(embeddings, chunk_ids, metadata, created_at)
                self.global_index.batch_add(embeddings, chunk_ids, metadata, created_at)
            self.save_state()
            self.replicate_event({"action": "add_document", "data": {"library_id": library_id, "document": document.dict()}})
            return document

    def get_document(self, library_id: str, document_id: str) -> Optional[Document]:
        with self.lock:
            if library_id in self.libraries:
                for doc in self.libraries[library_id].documents:
                    if doc.id == document_id:
                        return doc
            return None

    def update_document(self, library_id: str, document_id: str, metadata: Dict[str, str]) -> Optional[Document]:
        with self.lock:
            if not self.is_leader and runningInKubes:
                raise Exception("Write operation not allowed on follower")
            if library_id in self.libraries:
                for doc in self.libraries[library_id].documents:
                    if doc.id == document_id:
                        doc.metadata.update(metadata)
                        self.save_state()
                        self.replicate_event({"action": "update_document", "data": {"library_id": library_id, "document_id": document_id, "metadata": metadata}})
                        return doc
            return None

    def delete_document(self, library_id: str, document_id: str) -> bool:
        with self.lock:
            if not self.is_leader and runningInKubes:
                raise Exception("Write operation not allowed on follower")
            if library_id in self.libraries:
                library = self.libraries[library_id]
                for i, doc in enumerate(library.documents):
                    if doc.id == document_id:
                        library.documents.pop(i)
                        self.indexes[library_id] = BallTreeIndex()
                        self.global_index = BallTreeIndex()
                        for lib_id, lib in self.libraries.items():
                            embeddings = []
                            chunk_ids = []
                            metadata = []
                            created_at = []
                            for doc in lib.documents:
                                for chunk in doc.chunks:
                                    if chunk.embedding:
                                        embeddings.append(chunk.embedding)
                                        chunk_ids.append(chunk.id)
                                        metadata.append(chunk.metadata)
                                        created_at.append(chunk.created_at)
                            if embeddings:
                                self.indexes[lib_id].batch_add(embeddings, chunk_ids, metadata, created_at)
                                self.global_index.batch_add(embeddings, chunk_ids, metadata, created_at)
                        self.save_state()
                        self.replicate_event({"action": "delete_document", "data": {"library_id": library_id, "document_id": document_id}})
                        return True
            return False

    def add_chunk(self, library_id: str, document_id: str, chunk: Chunk) -> Optional[Chunk]:
        with self.lock:
            if not self.is_leader and runningInKubes:
                raise Exception("Write operation not allowed on follower")
            if library_id in self.libraries:
                for doc in self.libraries[library_id].documents:
                    if doc.id == document_id:
                        doc.chunks.append(chunk)
                        if chunk.embedding:
                            self.indexes[library_id].add(chunk.embedding, chunk.id, chunk.metadata, chunk.created_at)
                            self.global_index.add(chunk.embedding, chunk.id, chunk.metadata, chunk.created_at)
                            if 'name' in chunk.metadata:
                                name = chunk.metadata['name'].lower()
                                if name not in self.name_index:
                                    self.name_index[name] = []
                                self.name_index[name].append((library_id, chunk.id))
                        self.save_state()
                        self.replicate_event({"action": "add_chunk", "data": {"library_id": library_id, "document_id": document_id, "chunk": chunk.dict()}})
                        return chunk
            return None

    def get_chunk(self, library_id: str, document_id: str, chunk_id: str) -> Optional[Chunk]:
        with self.lock:
            if library_id in self.libraries:
                for doc in self.libraries[library_id].documents:
                    if doc.id == document_id:
                        for chunk in doc.chunks:
                            if chunk.id == chunk_id:
                                return chunk
            return None

    def update_chunk(self, library_id: str, document_id: str, chunk_id: str, text: str, embedding: List[float]) -> Optional[Chunk]:
        with self.lock:
            if not self.is_leader and runningInKubes:
                raise Exception("Write operation not allowed on follower")
            if library_id in self.libraries:
                for doc in self.libraries[library_id].documents:
                    if doc.id == document_id:
                        for chunk in doc.chunks:
                            if chunk.id == chunk_id:
                                chunk.text = text
                                chunk.embedding = embedding
                                self.indexes[library_id] = BallTreeIndex()
                                self.global_index = BallTreeIndex()
                                for lib_id, lib in self.libraries.items():
                                    embeddings = []
                                    chunk_ids = []
                                    metadata = []
                                    created_at = []
                                    for doc in lib.documents:
                                        for c in doc.chunks:
                                            if c.embedding:
                                                embeddings.append(c.embedding)
                                                chunk_ids.append(c.id)
                                                metadata.append(c.metadata)
                                                created_at.append(c.created_at)
                                    if embeddings:
                                        self.indexes[lib_id].batch_add(embeddings, chunk_ids, metadata, created_at)
                                        self.global_index.batch_add(embeddings, chunk_ids, metadata, created_at)
                                self.save_state()
                                self.replicate_event({"action": "update_chunk", "data": {"library_id": library_id, "document_id": document_id, "chunk_id": chunk_id, "text": text, "embedding": embedding}})
                                return chunk
            return None

    def delete_chunk(self, library_id: str, document_id: str, chunk_id: str) -> bool:
        with self.lock:
            if not self.is_leader and runningInKubes:
                raise Exception("Write operation not allowed on follower")
            if library_id in self.libraries:
                for doc in self.libraries[library_id].documents:
                    if doc.id == document_id:
                        for i, chunk in enumerate(doc.chunks):
                            if chunk.id == chunk_id:
                                doc.chunks.pop(i)
                                self.indexes[library_id] = BallTreeIndex()
                                self.global_index = BallTreeIndex()
                                for lib_id, lib in self.libraries.items():
                                    embeddings = []
                                    chunk_ids = []
                                    metadata = []
                                    created_at = []
                                    for doc in lib.documents:
                                        for c in doc.chunks:
                                            if c.embedding:
                                                embeddings.append(c.embedding)
                                                chunk_ids.append(c.id)
                                                metadata.append(c.metadata)
                                                created_at.append(c.created_at)
                                    if embeddings:
                                        self.indexes[lib_id].batch_add(embeddings, chunk_ids, metadata, created_at)
                                        self.global_index.batch_add(embeddings, chunk_ids, metadata, created_at)
                                self.save_state()
                                self.replicate_event({"action": "delete_chunk", "data": {"library_id": library_id, "document_id": document_id, "chunk_id": chunk_id}})
                                return True
            return False

    def search(self, library_id: str, query_embedding: List[float], k: int = 5) -> List[Dict]:
        with self.lock:
            logger.debug(f"Search in library {library_id} with query_embedding={query_embedding}")
            if library_id not in self.libraries:
                logger.warning(f"Library {library_id} not found")
                return []
            index = self.indexes.get(library_id)
            if not index or isinstance(index, FlatIndex):
                logger.warning(f"No valid index for library {library_id}")
                return []
            chunk_ids = index.search(query_embedding, k)
            results = []
            for chunk_id, similarity in chunk_ids:
                for doc in self.libraries[library_id].documents:
                    for chunk in doc.chunks:
                        if chunk.id == chunk_id:
                            results.append({
                                "document_id": doc.id,
                                "chunk_id": chunk.id,
                                "text": chunk.text,
                                "metadata": chunk.metadata,
                                "similarity": similarity
                            })
            logger.debug(f"Search in library {library_id} returned {len(results)} results")
            return results

    def search_chunks_after_date(self, query_embedding: np.ndarray, start_date: datetime, name_contains: str) -> List[ChunkSearchResult]:
        with self.lock:
            logger.debug(f"Starting search with {len(self.libraries)} libraries")
            if not self.libraries:
                logger.warning("No libraries exist in VectorDBService")
                return []
            if query_embedding is None:
                logger.error("Query embedding is None")
                raise ValueError("Query embedding cannot be None")
            query_embedding = query_embedding.astype(np.float32)
            # Log all chunks
            for library_id, library in self.libraries.items():
                for doc in library.documents:
                    for chunk in doc.chunks:
                        logger.debug(f"Chunk {chunk.id}: text={chunk.text}, metadata={chunk.metadata}, created_at={chunk.created_at}, embedding={chunk.embedding}")

            # Validate embedding dimension
            expected_dim = None
            chunk_count = 0
            for library_id, library in self.libraries.items():
                for doc in library.documents:
                    for chunk in doc.chunks:
                        chunk_count += 1
                        if chunk.embedding:
                            expected_dim = len(chunk.embedding)
                            logger.debug(f"Found chunk with embedding dimension: {expected_dim}")
                            break
                    if expected_dim is not None:
                        break
                if expected_dim is not None:
                    break
            if chunk_count == 0:
                logger.warning("No chunks found in dataset")
                return []
            if expected_dim is None:
                logger.warning("No chunks with embeddings found, skipping embedding validation")
                # Return chunks based on metadata/source and date only
                results = []
                for library_id, library in self.libraries.items():
                    for doc in library.documents:
                        for chunk in doc.chunks:
                            if chunk.created_at >= start_date:
                                if not name_contains or \
                                   (name_contains and 'source' in chunk.metadata and name_contains.lower() in chunk.metadata['source'].lower()) or \
                                   (name_contains and name_contains.lower() in chunk.text.lower()):
                                    results.append(ChunkSearchResult(
                                        library_id=library_id,
                                        document_id=doc.id,
                                        chunk_id=chunk.id,
                                        text=chunk.text,
                                        metadata=chunk.metadata,
                                        created_at=chunk.created_at.isoformat(),
                                        similarity=0.0  # No embedding comparison
                                    ))
                logger.debug(f"Search returned {len(results)} results")
                return results
            if len(query_embedding) != expected_dim:
                logger.error(f"Embedding dimension mismatch: query {len(query_embedding)}, expected {expected_dim}")
                raise ValueError(f"Query embedding dimension must be {expected_dim}")
            # Pre-filter by name_contains
            valid_chunk_ids = None
            if name_contains:
                logger.debug(f"Filtering by name_contains: {name_contains}")
                valid_chunk_ids = set()
                name_contains_lower = name_contains.lower()
                for library_id, library in self.libraries.items():
                    for doc in library.documents:
                        for chunk in doc.chunks:
                            if ('source' in chunk.metadata and name_contains_lower in chunk.metadata['source'].lower()) or \
                               (name_contains_lower in chunk.text.lower()):
                                valid_chunk_ids.add(chunk.id)
                logger.debug(f"Found {len(valid_chunk_ids)} chunks matching name_contains")
            results = self.global_index.search(query_embedding.tolist(), k=10, start_date=start_date, name_contains=None)
            output = []
            for chunk_id, similarity in results:
                if valid_chunk_ids is not None and chunk_id not in valid_chunk_ids:
                    continue
                for library_id, library in self.libraries.items():
                    for doc in library.documents:
                        for chunk in doc.chunks:
                            if chunk.id == chunk_id:
                                output.append(ChunkSearchResult(
                                    library_id=library_id,
                                    document_id=doc.id,
                                    chunk_id=chunk.id,
                                    text=chunk.text,
                                    metadata=chunk.metadata,
                                    created_at=chunk.created_at.isoformat(),
                                    similarity=similarity
                                ))
            logger.debug(f"Search returned {len(output)} results")
            return output

