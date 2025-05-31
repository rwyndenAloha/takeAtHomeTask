/*
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
*/

import numpy as np from typing import List, Tuple import threading

class FlatIndex: """Flat index using brute-force k-NN search""" def init(self): self.embeddings = [] self.chunk_ids = [] self.lock = threading.RLock()

def add(self, embedding: List[float], chunk_id: str) -> None:
    with self.lock:
        self.embeddings.append(np.array(embedding))
        self.chunk_ids.append(chunk_id)

def search(self, query: List[float], k: int) -> List[Tuple[str, float]]:
    with self.lock:
        if not self.embeddings:
            return []
        query = np.array(query)
        distances = [np.linalg.norm(emb - query) for emb in self.embeddings]
        sorted_indices = np.argsort(distances)[:k]
        return [(self.chunk_ids[i], float(distances[i])) for i in sorted_indices]

class BallTreeIndex: """Ball tree index for approximate k-NN search""" def init(self): self.lock = threading.RLock() self.root = None self.chunk_ids = [] self.points = []

class Node:
    def __init__(self, point, chunk_id, radius, left=None, right=None):
        self.point = point
        self.chunk_id = chunk_id
        self.radius = radius
        self.left = left
        self.right = right

def add(self, embedding: List[float], chunk_id: str) -> None:
    with self.lock:
        point = np.array(embedding)
        self.points.append(point)
        self.chunk_ids.append(chunk_id)
        if len(self.points) == 1:
            self.root = self.Node(point, chunk_id, 0.0)
        else:
            self._insert(point, chunk_id, self.root)

def _insert(self, point, chunk_id, node):
    dist = np.linalg.norm(point - node.point)
    if dist < node.radius:
        if node.left is None:
            node.left = self.Node(point, chunk_id, 0.0)
        else:
            self._insert(point, chunk_id, node.left)
    else:
        if node.right is None:
            node.right = self.Node(point, chunk_id, 0.0)
        else:
            self._insert(point, chunk_id, node.right)
        node.radius = max(node.radius, dist)

def search(self, query: List[float], k: int) -> List[Tuple[str, float]]:
    with self.lock:
        query = np.array(query)
        results = []
        self._search(query, k, self.root, results)
        results.sort(key=lambda x: x[1])
        return results[:k]

def _search(self, query, k, node, results):
    if node is None:
        return
    dist = float(np.linalg.norm(query - node.point))
    results.append((node.chunk_id, dist))
    if node.left:
        self._search(query, k, node.left, results)
    if node.right and dist - node.radius < min([r[1] for r in results] + [float('inf')]):
        self._search(query, k, node.right, results)

