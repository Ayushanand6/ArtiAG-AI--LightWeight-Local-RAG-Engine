package com.vector.ai.index;

import com.vector.ai.distance.CosineDistance;
import com.vector.ai.model.SearchResult;
import com.vector.ai.model.VectorRecord;
import org.springframework.stereotype.Component;

import java.util.*;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * Linear scan (brute-force) vector index.
 * Evaluates cosine similarity across every stored vector.
 * Acts as the exact ground-truth baseline for benchmarking approximate nearest neighbor (ANN) search.
 */
@Component("linearScanIndex")
public class LinearScanIndex implements VectorIndex {

    private final List<VectorRecord> records = new CopyOnWriteArrayList<>();

    @Override
    public void insert(VectorRecord record) {
        if (record == null) {
            throw new IllegalArgumentException("Record cannot be null");
        }
        records.add(record);
    }

    @Override
    public List<SearchResult> search(float[] queryEmbedding, int topK) {
        if (records.isEmpty() || topK <= 0) {
            return Collections.emptyList();
        }

        // Min-heap keeps the topK best results
        PriorityQueue<SearchResult> minHeap = new PriorityQueue<>(Comparator.comparingDouble(SearchResult::similarityScore));

        for (VectorRecord record : records) {
            float score = CosineDistance.cosineSimilarity(queryEmbedding, record.embedding());
            minHeap.offer(new SearchResult(record, score));
            if (minHeap.size() > topK) {
                minHeap.poll();
            }
        }

        List<SearchResult> results = new ArrayList<>(minHeap);
        results.sort((a, b) -> Float.compare(b.similarityScore(), a.similarityScore()));
        return results;
    }

    @Override
    public int size() {
        return records.size();
    }
}
