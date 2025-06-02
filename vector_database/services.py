"""
Concurrency handling (also see indexing.py for bulk of it...)

VectorDBService: self.lock (RLock) wraps all methods, ensuring thread-safe access to self.libraries and self.indexes.

Overview of thread safety

VectorDBService uses RLock for all methods, protecting self.libraries (dictionary updates, list appends) and self.indexes (index creation, updates, queries). This covers:
Reads: get_library, search (accessing self.libraries, calling index.query).
Writes: create_library, add_document, update_library (modifying self.libraries, self.indexes).

Asyncio Safety: RLock is thread-safe, and since FastAPI uses asyncio, the lock ensures coroutines don’t race (e.g., concurrent POST /documents/ and POST /search/)

VectorDBService added with persistence and leader-follower architecture.

Persistence: Saves self.libraries to vector_db_state.json in a PersistentVolume.
Leader Election: Uses Kubernetes Lease for leader selection.

Replication: Leader sends write events to followers via HTTP /replicate/.
Failover: Followers monitor lease and promote to leader if needed.
Thread Safety: self.lock (RLock) ensures safe access to self.libraries and self.indexes..
"""

from typing import List, Optional, Dict
import threading
import numpy as np
from models import Library, Document, Chunk
from heapq import heappush, heappop
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

class FlatIndex:
    def __init__(self):
        pass
    def query(self, embedding: List[float], k: int = 5) -> List[int]:
        return []

class BallTreeNode:
    def __init__(self, points: np.ndarray, indices: List[int]):
        self.points = points
        self.indices = indices
        self.centroid = np.mean(points, axis=0)
        self.radius = np.max(np.sqrt(np.sum((points - self.centroid) ** 2, axis=1))) if len(points) > 0 else 0.0
        self.left: Optional[BallTreeNode] = None
        self.right: Optional[BallTreeNode] = None

class BallTreeIndex:
    def __init__(self, embeddings: List[List[float]] = None):
        self.root = None
        self.embedding_ids: List[int] = []
        self.chunk_ids: List[str] = []
        if embeddings and len(embeddings) > 0:
            embeddings_array = np.array(embeddings)
            if embeddings_array.ndim == 1:
                embeddings_array = embeddings_array.reshape(1, -1)
            self.embedding_ids = list(range(len(embeddings_array)))
            self.root = self._build_tree(embeddings_array, self.embedding_ids)

    def _build_tree(self, points: np.ndarray, indices: List[int], depth: int = 0) -> Optional[BallTreeNode]:
        if len(points) == 0:
            return None
        if len(points) <= 10:
            return BallTreeNode(points, indices)
        centroid = np.mean(points, axis=0)
        distances = np.sqrt(np.sum((points - centroid) ** 2, axis=1))
        pivot_idx = np.argmax(distances)
        pivot = points[pivot_idx]
        distances_to_pivot = np.sqrt(np.sum((points - pivot) ** 2, axis=1))
        median_dist = np.median(distances_to_pivot)
        left_mask = distances_to_pivot <= median_dist
        right_mask = ~left_mask
        left_points = points[left_mask]
        right_points = points[right_mask]
        left_indices = [indices[i] for i in range(len(indices)) if left_mask[i]]
        right_indices = [indices[i] for i in range(len(indices)) if right_mask[i]]
        node = BallTreeNode(points, indices)
        node.left = self._build_tree(left_points, left_indices, depth + 1)
        node.right = self._build_tree(right_points, right_indices, depth + 1)
        return node

    def add_embeddings(self, embeddings: List[List[float]], chunk_ids: List[str]) -> None:
        if not embeddings:
            return
        embeddings_array = np.array(embeddings)
        if embeddings_array.ndim == 1:
            embeddings_array = embeddings_array.reshape(1, -1)
        new_ids = list(range(len(self.embedding_ids), len(self.embedding_ids) + len(embeddings_array)))
        self.embedding_ids.extend(new_ids)
        self.chunk_ids.extend(chunk_ids)
        all_points = np.vstack([self.root.points, embeddings_array]) if self.root else embeddings_array
        self.root = self._build_tree(all_points, self.embedding_ids)

    def query(self, embedding: List[float], k: int = 5) -> List[str]:
        if self.root is None:
            return []
        embedding_array = np.array(embedding)
        if embedding_array.ndim == 1:
            embedding_array = embedding_array.reshape(1, -1)
        heap = []
        def search(node: Optional[BallTreeNode]):
            if node is None:
                return
            dist_to_centroid = np.sqrt(np.sum((embedding_array - node.centroid) ** 2))
            if len(heap) < k or dist_to_centroid - node.radius < heap[0][0]:
                if node.left is None and node.right is None:
                    for idx, point in zip(node.indices, node.points):
                        dist = np.sqrt(np.sum((embedding_array - point) ** 2))
                        if len(heap) < k:
                            heappush(heap, (-dist, idx))
                        elif dist < -heap[0][0]:
                            heappop(heap)
                            heappush(heap, (-dist, idx))
                else:
                    search(node.left)
                    search(node.right)
        search(self.root)
        return [self.chunk_ids[idx] for _, idx in sorted(heap, reverse=True)][:k]

class VectorDBService:
    def __init__(self, pod_name: str, namespace: str = "default"):
        self.libraries: Dict[str, Library] = {}
        self.indexes: Dict[str, FlatIndex | BallTreeIndex] = {}
        self.lock = threading.RLock()
        self.pod_name = pod_name
        self.namespace = namespace
        self.is_leader = False
        self.leader_lease_name = "vector-db-leader"
        self.state_file = f"/data/{pod_name}/vector_db_state.json"
        self.event_log = []  # In-memory event log for replication
        self.followers = []  # List of follower pod URLs

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
            # Running in Docker or local, skip leader election
            logger.info("Not running in Kubernetes, skipping leader election")
            self.is_leader = True  # Default to leader for non-Kubernetes

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
                # Try to acquire or renew lease
                try:
                    current_lease = coordination_api.read_namespaced_lease(self.leader_lease_name, self.namespace)
                    if current_lease.spec.holder_identity == self.pod_name:
                        # Renew lease
                        current_lease.spec.renew_time = datetime.utcnow()
                        coordination_api.replace_namespaced_lease(self.leader_lease_name, self.namespace, current_lease)
                        self.is_leader = True
                        self.update_followers()
                    elif (datetime.utcnow() - current_lease.spec.renew_time.replace(tzinfo=None)).total_seconds() > 15:
                        # Steal lease if expired
                        current_lease.spec.holder_identity = self.pod_name
                        current_lease.spec.renew_time = datetime.utcnow()
                        coordination_api.replace_namespaced_lease(self.leader_lease_name, self.namespace, current_lease)
                        self.is_leader = True
                        self.update_followers()
                    else:
                        self.is_leader = False
                except ApiException as e:
                    if e.status == 404:
                        # Create lease if it doesn't exist
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
        # Discover followers via Kubernetes service DNS
        followers = []
        num_replicas = 3  # Adjust based on StatefulSet replicas
        for i in range(num_replicas):
            pod_hostname = f"vector-db-{i}"
            if pod_hostname != self.pod_name:
                followers.append(f"http://{pod_hostname}.vector-db.default.svc.cluster.local:8000/api/v1/replicate/")
        self.followers = followers
        logger.debug(f"Updated followers: {self.followers}")

    def save_state(self):
        with self.lock:
            try:
                # Serialize libraries directly using Pydantic's dict() method
                libraries_json = {k: v.dict(exclude_unset=True) for k, v in self.libraries.items()}
                with open(self.state_file, "w") as f:
                    json.dump(libraries_json, f, indent=2)
                logger.debug(f"Saved state to {self.state_file}")
            except Exception as e:
                logger.error(f"Failed to save state: {e}")
                raise

    def load_state(self):
        with self.lock:
            try:
                if os.path.exists(self.state_file):
                    with open(self.state_file, "r") as f:
                        data = json.load(f)
                    self.libraries = {k: Library.parse_obj(v) for k, v in data.items()}
                    # Rebuild indexes
                    for library_id, library in self.libraries.items():
                        embeddings = []
                        chunk_ids = []
                        for doc in library.documents:
                            for chunk in doc.chunks:
                                if chunk.embedding:
                                    embedding = [float(x) for x in chunk.embedding]
                                    embeddings.append(embedding)
                                    chunk_ids.append(chunk.id)
                        self.indexes[library_id] = BallTreeIndex(embeddings) if embeddings else FlatIndex()
                        if embeddings:
                            self.indexes[library_id].chunk_ids = chunk_ids
                    logger.debug(f"Loaded state from {self.state_file}")
            except Exception as e:
                logger.error(f"Failed to load state: {e}")

    def replicate_event(self, event: Dict):
        if not self.is_leader:
            return
        with self.lock:
            self.event_log.append(event)
            for follower in self.followers:
                try:
                    response = requests.post(follower, json=event, timeout=5)
                    response.raise_for_status()
                except requests.RequestException as e:
                    logger.error(f"Failed to replicate to {follower}: {e}")

    def apply_event(self, event: Dict):
        with self.lock:
            action = event.get("action")
            data = event.get("data")
            if action == "create_library":
                library = Library.parse_obj(data)
                self.libraries[library.id] = library
                embeddings = []
                chunk_ids = []
                for doc in library.documents:
                    for chunk in doc.chunks:
                        if chunk.embedding:
                            embedding = [float(x) for x in chunk.embedding]
                            embeddings.append(embedding)
                            chunk_ids.append(chunk.id)
                self.indexes[library.id] = BallTreeIndex(embeddings) if embeddings else FlatIndex()
                if embeddings:
                    self.indexes[library.id].chunk_ids = chunk_ids
            elif action == "add_document":
                library_id = data["library_id"]
                document = Document.parse_obj(data["document"])
                if library_id in self.libraries:
                    self.libraries[library_id].documents.append(document)
                    embeddings = []
                    chunk_ids = []
                    for chunk in document.chunks:
                        if chunk.embedding:
                            embedding = [float(x) for x in chunk.embedding]
                            embeddings.append(embedding)
                            chunk_ids.append(chunk.id)
                    if embeddings:
                        self.indexes[library_id].add_embeddings(embeddings, chunk_ids)
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
            elif action == "update_document":
                library_id = data["library_id"]
                document_id = data["document_id"]
                metadata = data["metadata"]
                if library_id in self.libraries:
                    for doc in self.libraries[library_id].documents:
                        if doc.id == document_id:
                            doc.metadata.update(metadata)
                            break
            elif action == "delete_document":
                library_id = data["library_id"]
                document_id = data["document_id"]
                if library_id in self.libraries:
                    library = self.libraries[library_id]
                    for i, doc in enumerate(library.documents):
                        if doc.id == document_id:
                            library.documents.pop(i)
                            embeddings = []
                            chunk_ids = []
                            for doc in library.documents:
                                for chunk in doc.chunks:
                                    if chunk.embedding:
                                        embedding = [float(x) for x in chunk.embedding]
                                        embeddings.append(embedding)
                                        chunk_ids.append(chunk.id)
                            self.indexes[library_id] = BallTreeIndex(embeddings) if embeddings else FlatIndex()
                            if embeddings:
                                self.indexes[library_id].chunk_ids = chunk_ids
                            break
            elif action == "add_chunk":
                library_id = data["library_id"]
                document_id = data["document_id"]
                chunk = Chunk.parse_obj(data["chunk"])
                if library_id in self.libraries:
                    for doc in self.libraries[library_id].documents:
                        if doc.id == document_id:
                            doc.chunks.append(chunk)
                            if chunk.embedding:
                                embedding = [float(x) for x in chunk.embedding]
                                self.indexes[library_id].add_embeddings([embedding], [chunk.id])
                            break
            self.save_state()

    def create_library(self, library: Library) -> Library:
        with self.lock:
            if not self.is_leader and runningInKubes:
                raise Exception("Write operation not allowed on follower")
            embeddings = []
            chunk_ids = []
            for doc in library.documents:
                for chunk in doc.chunks:
                    if chunk.embedding:
                        embedding = [float(x) for x in chunk.embedding]
                        embeddings.append(embedding)
                        chunk_ids.append(chunk.id)
            if embeddings and not all(len(e) == len(embeddings[0]) for e in embeddings):
                raise ValueError("All embeddings must have the same dimensionality")
            self.libraries[library.id] = library
            index = BallTreeIndex(embeddings) if embeddings else FlatIndex()
            if embeddings:
                index.chunk_ids = chunk_ids
            self.indexes[library.id] = index
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
            for chunk in document.chunks:
                if chunk.embedding:
                    embedding = [float(x) for x in chunk.embedding]
                    embeddings.append(embedding)
                    chunk_ids.append(chunk.id)
            if embeddings:
                self.indexes[library_id].add_embeddings(embeddings, chunk_ids)
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
                        embeddings = []
                        chunk_ids = []
                        for doc in library.documents:
                            for chunk in doc.chunks:
                                if chunk.embedding:
                                    embedding = [float(x) for x in chunk.embedding]
                                    embeddings.append(embedding)
                                    chunk_ids.append(chunk.id)
                        self.indexes[library_id] = BallTreeIndex(embeddings) if embeddings else FlatIndex()
                        if embeddings:
                            self.indexes[library_id].chunk_ids = chunk_ids
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
                            embedding = [float(x) for x in chunk.embedding]
                            self.indexes[library_id].add_embeddings([embedding], [chunk.id])
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
                                chunk.embedding = [float(x) for x in embedding]
                                embeddings = []
                                chunk_ids = []
                                for d in self.libraries[library_id].documents:
                                    for c in d.chunks:
                                        if c.embedding:
                                            embedding = [float(x) for x in c.embedding]
                                            embeddings.append(embedding)
                                            chunk_ids.append(c.id)
                                self.indexes[library_id] = BallTreeIndex(embeddings) if embeddings else FlatIndex()
                                if embeddings:
                                    self.indexes[library_id].chunk_ids = chunk_ids
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
                                embeddings = []
                                chunk_ids = []
                                for d in self.libraries[library_id].documents:
                                    for c in d.chunks:
                                        if c.embedding:
                                            embedding = [float(x) for x in c.embedding]
                                            embeddings.append(embedding)
                                            chunk_ids.append(c.id)
                                self.indexes[library_id] = BallTreeIndex(embeddings) if embeddings else FlatIndex()
                                if embeddings:
                                    self.indexes[library_id].chunk_ids = chunk_ids
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
            chunk_ids = index.query(query_embedding, k)
            results = []
            for chunk_id in chunk_ids:
                for doc in self.libraries[library_id].documents:
                    for chunk in doc.chunks:
                        if chunk.id == chunk_id:
                            results.append({
                                "document_id": doc.id,
                                "chunk_id": chunk.id,
                                "text": chunk.text,
                                "metadata": chunk.metadata
                            })
            logger.debug(f"Search in library {library_id} returned {len(results)} results")
            return results

    def search_chunks_after_date(self, query_embedding: np.ndarray, start_date: datetime, name_contains: str) -> List[dict]:
        with self.lock:
            results = []
            logger.debug(f"Search parameters: query_embedding={query_embedding.tolist()}, start_date={start_date}, name_contains={name_contains}")
            if not self.libraries:
                logger.warning("No libraries exist in VectorDBService")
                return []
            logger.debug(f"Processing {len(self.libraries)} libraries")
            for library_id, library in self.libraries.items():
                logger.debug(f"Processing library {library_id} with {len(library.documents)} documents")
                if not library.documents:
                    logger.debug(f"No documents in library {library_id}")
                    continue
                for document in library.documents:
                    logger.debug(f"Processing document {document.id} with {len(document.chunks)} chunks")
                    if not document.chunks:
                        logger.debug(f"No chunks in document {document.id}")
                        continue
                    for chunk in document.chunks:
                        # Validate chunk data
                        if not hasattr(chunk, 'id') or not hasattr(chunk, 'embedding') or not hasattr(chunk, 'created_at'):
                            logger.error(f"Invalid chunk data: {chunk}")
                            continue
                        # Filter by date
                        try:
                            chunk_date = chunk.created_at if isinstance(chunk.created_at, datetime) else datetime.fromisoformat(chunk.created_at.replace("Z", "+00:00"))
                        except ValueError as e:
                            logger.error(f"Invalid created_at for chunk {chunk.id}: {chunk.created_at}, error: {e}")
                            continue
                        if chunk_date <= start_date:
                            logger.debug(f"Chunk {chunk.id} skipped due to date {chunk_date} <= {start_date}")
                            continue
                        # Filter by metadata source
                        source = chunk.metadata.get("source", "").lower()
                        if not isinstance(source, str):
                            logger.error(f"Invalid metadata source for chunk {chunk.id}: {source}")
                            continue
                        if name_contains not in source:
                            logger.debug(f"Chunk {chunk.id} skipped due to source '{source}' not containing '{name_contains}'")
                            continue
                        # Perform similarity search
                        if not chunk.embedding:
                            logger.warning(f"Empty embedding for chunk {chunk.id}")
                            continue
                        try:
                            logger.debug(f"Processing chunk {chunk.id} with embedding: {chunk.embedding}")
                            chunk_embedding = np.array([float(x) for x in chunk.embedding], dtype=np.float64)
                        except (ValueError, TypeError) as e:
                            logger.error(f"Invalid embedding for chunk {chunk.id}: {chunk.embedding}, error: {e}")
                            continue
                        if chunk_embedding.shape != query_embedding.shape:
                            logger.warning(f"Shape mismatch for chunk {chunk.id}: chunk {chunk_embedding.shape}, query {query_embedding.shape}")
                            continue
                        try:
                            similarity = np.dot(chunk_embedding, query_embedding) / (
                                np.linalg.norm(chunk_embedding) * np.linalg.norm(query_embedding)
                            )
                            if not np.isfinite(similarity):
                                logger.warning(f"Non-finite similarity for chunk {chunk.id}")
                                continue
                        except Exception as e:
                            logger.error(f"Similarity computation failed for chunk {chunk.id}: {e}")
                            continue
                        if similarity > 0.5:
                            results.append({
                                "library_id": library_id,
                                "document_id": document.id,
                                "chunk_id": chunk.id,
                                "text": chunk.text,
                                "metadata": chunk.metadata,
                                "created_at": chunk.created_at.isoformat() + "Z" if isinstance(chunk.created_at, datetime) else chunk.created_at,
                                "similarity": float(similarity)
                            })
                            logger.debug(f"Added result for chunk {chunk.id} with similarity {similarity}")
            logger.debug(f"Search returned {len(results)} results")
            return results

