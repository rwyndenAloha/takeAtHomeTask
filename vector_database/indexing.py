"""
Indexing Algorithms Supported

FlatIndex: Brute-force k-NN search.
Space Complexity: O(n*d), where n is the number of chunks, d is embedding dimension
Time Complexity: O(n*d) for search, O(1) for insertion.
Choice: Simple, exact results, suitable for small datasets.

BallTreeIndex: Hierarchical partitioning for approximate k-NN.
Space Complexity: O(n*d).
Time Complexity: O(log n * d) for search (average), O(log n) for insertion.
Choice: Faster for larger datasets, balances speed and accuracy.

Thread Safety Choice

RLock: Used in VectorDBService and indexes to ensure thread-safe reads and writes.
Design Choice: RLock allows recursive locking within the same thread, suitable for nested operations (e.g., updating a chunk within a document). It prevents data races while maintaining performance for read-heavy workloads.

Thread Safety Mechanisms

threading.RLock:
indexing.py:
FlatIndex: self.lock (RLock) protects add and search, ensuring thread-safe list appends and distance computations.
BallTreeIndex: self.lock protects add (tree insertion) and search (tree traversal), preventing races during node updates or queries.
"""

import numpy as np
from typing import List, Tuple, Optional
import threading
import heapq
from datetime import datetime

class FlatIndex:
    """Flat index using vectorized brute-force k-NN search"""
    def __init__(self):
        self.embeddings = None  # Single NumPy array for all embeddings
        self.chunk_ids = []
        self.metadata = []  # Store chunk metadata for filtering
        self.created_at = []  # Store chunk creation times
        self.lock = threading.RLock()
        self.n = 0

    def add(self, embedding: List[float], chunk_id: str, metadata: dict, created_at: datetime) -> None:
        """Add a chunk with embedding, ID, metadata, and creation time."""
        with self.lock:
            embedding = np.array(embedding, dtype=np.float32)
            if self.embeddings is None:
                self.embeddings = embedding.reshape(1, -1)
            else:
                self.embeddings = np.vstack((self.embeddings, embedding))
            self.chunk_ids.append(chunk_id)
            self.metadata.append(metadata)
            self.created_at.append(created_at)
            self.n += 1

    def batch_add(self, embeddings: List[List[float]], chunk_ids: List[str], metadata: List[dict], created_at: List[datetime]) -> None:
        """Add multiple chunks efficiently."""
        with self.lock:
            embeddings = np.array(embeddings, dtype=np.float32)
            if self.embeddings is None:
                self.embeddings = embeddings
            else:
                self.embeddings = np.vstack((self.embeddings, embeddings))
            self.chunk_ids.extend(chunk_ids)
            self.metadata.extend(metadata)
            self.created_at.extend(created_at)
            self.n += len(embeddings)

    def search(self, query: List[float], k: int, start_date: Optional[datetime] = None, name_contains: Optional[str] = None) -> List[Tuple[str, float]]:
        """Search for k-nearest chunks with optional date and name filtering."""
        with self.lock:
            if self.n == 0:
                return []
            query = np.array(query, dtype=np.float32)
            # Vectorized Euclidean distance
            distances = np.sqrt(np.sum((self.embeddings - query) ** 2, axis=1))
            # Apply filters
            valid_indices = np.arange(self.n)
            if start_date:
                valid_indices = valid_indices[np.array([self.created_at[i] >= start_date for i in valid_indices])]
            if name_contains:
                valid_indices = valid_indices[np.array(['name' in self.metadata[i] and name_contains.lower() in self.metadata[i]['name'].lower() for i in valid_indices])]
            if len(valid_indices) == 0:
                return []
            # Filter distances and IDs
            distances = distances[valid_indices]
            chunk_ids = [self.chunk_ids[i] for i in valid_indices]
            # Get top-k
            top_k_indices = np.argpartition(distances, min(k, len(distances)))[:k]
            top_k = [(chunk_ids[i], float(distances[i])) for i in top_k_indices]
            top_k.sort(key=lambda x: x[1])
            return top_k

class BallTreeIndex:
    """Ball tree index for approximate k-NN search with priority queue"""
    class Node:
        def __init__(self, point, chunk_id, radius, metadata, created_at, left=None, right=None):
            self.point = point
            self.chunk_id = chunk_id
            self.radius = radius
            self.metadata = metadata
            self.created_at = created_at
            self.left = left
            self.right = right

    def __init__(self):
        self.lock = threading.RLock()
        self.root = None
        self.points = []
        self.chunk_ids = []

    def add(self, embedding: List[float], chunk_id: str, metadata: dict, created_at: datetime) -> None:
        """Add a chunk with embedding, ID, metadata, and creation time."""
        with self.lock:
            point = np.array(embedding, dtype=np.float32)
            self.points.append(point)
            self.chunk_ids.append(chunk_id)
            if len(self.points) == 1:
                self.root = self.Node(point, chunk_id, 0.0, metadata, created_at)
            else:
                self._insert(point, chunk_id, metadata, created_at, self.root)

    def batch_add(self, embeddings: List[List[float]], chunk_ids: List[str], metadata: List[dict], created_at: List[datetime]) -> None:
        """Add multiple chunks efficiently."""
        with self.lock:
            for emb, cid, meta, c_at in zip(embeddings, chunk_ids, metadata, created_at):
                self.add(emb, cid, meta, c_at)

    def _insert(self, point, chunk_id, metadata, created_at, node):
        dist = np.linalg.norm(point - node.point)
        if dist < node.radius:
            if node.left is None:
                node.left = self.Node(point, chunk_id, 0.0, metadata, created_at)
            else:
                self._insert(point, chunk_id, metadata, created_at, node.left)
        else:
            if node.right is None:
                node.right = self.Node(point, chunk_id, 0.0, metadata, created_at)
            else:
                self._insert(point, chunk_id, metadata, created_at, node.right)
            node.radius = max(node.radius, dist)

    def search(self, query: List[float], k: int, start_date: Optional[datetime] = None, name_contains: Optional[str] = None) -> List[Tuple[str, float]]:
        """Search for k-nearest chunks with optional date and name filtering."""
        with self.lock:
            if not self.root:
                return []
            query = np.array(query, dtype=np.float32)
            heap = []
            self._search(query, k, self.root, heap, start_date, name_contains)
            results = [(chunk_id, -dist) for dist, chunk_id in heap]  # Correct unpacking, negate distance
            results.sort(key=lambda x: x[1])  # Sort by distance (ascending)
            return results[:k]

    def _search(self, query, k, node, heap, start_date, name_contains):
        if node is None:
            return
        # Apply filters
        if start_date and node.created_at < start_date:
            return
        if name_contains and ('name' not in node.metadata or name_contains.lower() not in node.metadata['name'].lower()):
            return
        dist = float(np.linalg.norm(query - node.point))
        heapq.heappush(heap, (-dist, node.chunk_id))  # Negative distance for min-heap
        if len(heap) > k:
            heapq.heappop(heap)
        max_dist = -heap[0][0] if heap else float('inf')
        if node.left:
            self._search(query, k, node.left, heap, start_date, name_contains)
        if node.right and dist - node.radius < max_dist:
            self._search(query, k, node.right, heap, start_date, name_contains)

