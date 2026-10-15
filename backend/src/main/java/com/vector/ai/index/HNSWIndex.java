package com.vector.ai.index;

import com.vector.ai.distance.CosineDistance;
import com.vector.ai.model.SearchResult;
import com.vector.ai.model.VectorRecord;
import org.springframework.context.annotation.Primary;
import org.springframework.stereotype.Component;

import java.util.*;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Hierarchical Navigable Small World (HNSW) vector index.
 *
 * Algorithm characteristics:
 * - Multi-layer graph where upper layers have sparser connections for long-distance skips
 * - Layer 0 contains all nodes with up to 2*M connections
 * - Upper layers have up to M connections
 * - searchLayer uses a MAX-heap for candidates (best-first exploration)
 *   and a MIN-heap for results (tracking top-ef nearest with fast eviction of the worst)
 * - Thread-safety via ReentrantReadWriteLock (read locks for search/size, write lock for insert)
 */
@Component("hnswIndex")
@Primary
public class HNSWIndex implements VectorIndex {

    private final int M;
    private final int efConstruction;
    private final int efSearch;
    private final double mL;

    public HNSWIndex() {
        this(16, 128, 100);
    }

    public HNSWIndex(int M, int efConstruction, int efSearch) {
        this.M = M;
        this.efConstruction = efConstruction;
        this.efSearch = efSearch;
        this.mL = 1.0 / Math.log(M);
    }

    private final Map<String, Node> nodes = new HashMap<>();
    private Node entryPoint = null;
    private int maxLevel = -1;
    private final Random random = new Random();

    private final ReentrantReadWriteLock rwLock = new ReentrantReadWriteLock();
    private final Lock readLock = rwLock.readLock();
    private final Lock writeLock = rwLock.writeLock();

    public static class Node {
        public final VectorRecord record;
        public final int level;
        public final Map<Integer, Set<Node>> connections = new HashMap<>();

        public Node(VectorRecord record, int level) {
            this.record = record;
            this.level = level;
            for (int l = 0; l <= level; l++) {
                connections.put(l, new HashSet<>());
            }
        }
    }

    @Override
    public void insert(VectorRecord record) {
        if (record == null) {
            throw new IllegalArgumentException("Record cannot be null");
        }

        writeLock.lock();
        try {
            if (nodes.containsKey(record.id())) {
                throw new IllegalArgumentException("Record with id '" + record.id() + "' already exists in index");
            }

            int nodeLevel = randomLevel();
            Node newNode = new Node(record, nodeLevel);
            nodes.put(record.id(), newNode);

            if (entryPoint == null) {
                entryPoint = newNode;
                maxLevel = nodeLevel;
                return;
            }

            Node curr = entryPoint;
            float[] query = record.embedding();

            // 1. Greedily find closest entry point from top level down to nodeLevel + 1
            for (int l = maxLevel; l > nodeLevel; l--) {
                boolean changed = true;
                while (changed) {
                    changed = false;
                    Set<Node> neighbors = curr.connections.get(l);
                    if (neighbors != null) {
                        Node bestNeighbor = null;
                        float bestDist = CosineDistance.cosineSimilarity(query, curr.record.embedding());
                        for (Node neighbor : neighbors) {
                            float dist = CosineDistance.cosineSimilarity(query, neighbor.record.embedding());
                            if (dist > bestDist) {
                                bestDist = dist;
                                bestNeighbor = neighbor;
                            }
                        }
                        if (bestNeighbor != null) {
                            curr = bestNeighbor;
                            changed = true;
                        }
                    }
                }
            }

            // 2. From min(maxLevel, nodeLevel) down to 0, find neighbors and connect with pruning
            int startLevel = Math.min(maxLevel, nodeLevel);
            for (int l = startLevel; l >= 0; l--) {
                List<SearchResult> candidates = searchLayer(curr, query, efConstruction, l);
                int maxConn = (l == 0) ? 2 * M : M;

                int limit = Math.min(candidates.size(), maxConn);
                for (int i = 0; i < limit; i++) {
                    SearchResult candidate = candidates.get(i);
                    Node neighbor = nodes.get(candidate.record().id());
                    if (neighbor != null && neighbor != newNode) {
                        newNode.connections.get(l).add(neighbor);
                        neighbor.connections.get(l).add(newNode);

                        // Prune neighbor connections if exceeding max allowed
                        pruneConnections(neighbor, l, maxConn);
                    }
                }

                // Prune new node's own connections
                pruneConnections(newNode, l, maxConn);

                if (!candidates.isEmpty()) {
                    curr = nodes.get(candidates.get(0).record().id());
                }
            }

            if (nodeLevel > maxLevel) {
                maxLevel = nodeLevel;
                entryPoint = newNode;
            }
        } finally {
            writeLock.unlock();
        }
    }

    /**
     * Searches a single layer for the nearest neighbors of the query embedding.
     * Candidates = MAX-heap (best explored first).
     * Results = MIN-heap (worst on top, polled when size > ef).
     */
    private List<SearchResult> searchLayer(Node entryPointNode, float[] query, int ef, int level) {
        Set<String> visited = new HashSet<>();
        visited.add(entryPointNode.record.id());

        // MAX-heap: Highest similarity score has highest priority
        PriorityQueue<SearchResult> candidates = new PriorityQueue<>(
                (a, b) -> Float.compare(b.similarityScore(), a.similarityScore())
        );

        // MIN-heap: Lowest similarity score on top for O(1) eviction
        PriorityQueue<SearchResult> results = new PriorityQueue<>(
                (a, b) -> Float.compare(a.similarityScore(), b.similarityScore())
        );

        float initialScore = CosineDistance.cosineSimilarity(query, entryPointNode.record.embedding());
        SearchResult entryResult = new SearchResult(entryPointNode.record, initialScore);
        candidates.offer(entryResult);
        results.offer(entryResult);

        while (!candidates.isEmpty()) {
            SearchResult currentCandidate = candidates.poll();
            SearchResult worstResult = results.peek();

            if (currentCandidate.similarityScore() < worstResult.similarityScore() && results.size() >= ef) {
                break;
            }

            Node currentNode = nodes.get(currentCandidate.record().id());
            if (currentNode == null) continue;

            Set<Node> neighbors = currentNode.connections.get(level);
            if (neighbors != null) {
                for (Node neighbor : neighbors) {
                    if (visited.add(neighbor.record.id())) {
                        float score = CosineDistance.cosineSimilarity(query, neighbor.record.embedding());
                        SearchResult neighborResult = new SearchResult(neighbor.record, score);

                        if (results.size() < ef || score > results.peek().similarityScore()) {
                            candidates.offer(neighborResult);
                            results.offer(neighborResult);
                            if (results.size() > ef) {
                                results.poll(); // Evict worst candidate
                            }
                        }
                    }
                }
            }
        }

        List<SearchResult> list = new ArrayList<>(results);
        list.sort((a, b) -> Float.compare(b.similarityScore(), a.similarityScore()));
        return list;
    }

    private void pruneConnections(Node node, int level, int maxAllowed) {
        Set<Node> neighbors = node.connections.get(level);
        if (neighbors == null || neighbors.size() <= maxAllowed) {
            return;
        }

        float[] targetEmbedding = node.record.embedding();
        List<Node> sorted = new ArrayList<>(neighbors);
        sorted.sort((n1, n2) -> {
            float s1 = CosineDistance.cosineSimilarity(targetEmbedding, n1.record.embedding());
            float s2 = CosineDistance.cosineSimilarity(targetEmbedding, n2.record.embedding());
            return Float.compare(s2, s1); // Highest similarity first
        });

        neighbors.clear();
        for (int i = 0; i < maxAllowed; i++) {
            neighbors.add(sorted.get(i));
        }
    }

    @Override
    public List<SearchResult> search(float[] queryEmbedding, int topK) {
        if (queryEmbedding == null) {
            throw new IllegalArgumentException("Query embedding cannot be null");
        }
        if (topK <= 0) {
            return Collections.emptyList();
        }

        readLock.lock();
        try {
            if (entryPoint == null) {
                return Collections.emptyList();
            }

            Node curr = entryPoint;
            for (int l = maxLevel; l > 0; l--) {
                boolean changed = true;
                while (changed) {
                    changed = false;
                    Set<Node> neighbors = curr.connections.get(l);
                    if (neighbors != null) {
                        Node bestNeighbor = null;
                        float bestDist = CosineDistance.cosineSimilarity(queryEmbedding, curr.record.embedding());
                        for (Node neighbor : neighbors) {
                            float dist = CosineDistance.cosineSimilarity(queryEmbedding, neighbor.record.embedding());
                            if (dist > bestDist) {
                                bestDist = dist;
                                bestNeighbor = neighbor;
                            }
                        }
                        if (bestNeighbor != null) {
                            curr = bestNeighbor;
                            changed = true;
                        }
                    }
                }
            }

            List<SearchResult> candidates = searchLayer(curr, queryEmbedding, Math.max(efSearch, topK), 0);
            if (candidates.size() > topK) {
                return candidates.subList(0, topK);
            }
            return candidates;
        } finally {
            readLock.unlock();
        }
    }

    @Override
    public int size() {
        readLock.lock();
        try {
            return nodes.size();
        } finally {
            readLock.unlock();
        }
    }

    /**
     * Clears all nodes from the index.
     */
    public void clear() {
        writeLock.lock();
        try {
            nodes.clear();
            entryPoint = null;
            maxLevel = -1;
        } finally {
            writeLock.unlock();
        }
    }

    /**
     * Returns a snapshot list of all stored VectorRecords.
     */
    public List<VectorRecord> getAllRecords() {
        readLock.lock();
        try {
            return nodes.values().stream().map(n -> n.record).toList();
        } finally {
            readLock.unlock();
        }
    }

    private int randomLevel() {
        // Use 1.0 - random.nextDouble() to strictly avoid log(0)
        double r = 1.0 - random.nextDouble();
        return (int) (-Math.log(r) * mL);
    }
}
