package com.vector.ai.index;

import com.vector.ai.model.SearchResult;
import com.vector.ai.model.VectorRecord;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

class HNSWIndexTest {

    @Test
    @DisplayName("Should reject duplicate IDs")
    void shouldRejectDuplicateIds() {
        HNSWIndex index = new HNSWIndex();
        float[] vec = new float[]{1.0f, 0.0f, 0.0f};
        index.insert(new VectorRecord("doc1", vec, "payload 1"));

        assertThrows(IllegalArgumentException.class, () ->
                index.insert(new VectorRecord("doc1", vec, "payload duplicate"))
        );
    }

    @Test
    @DisplayName("HNSW recall vs LinearScan on random vectors should exceed 85%")
    void shouldAchieveHighRecallAgainstLinearScan() {
        int dim = 16;
        int numVectors = 200;
        int numQueries = 30;
        int topK = 5;

        Random random = new Random(42);

        HNSWIndex hnsw = new HNSWIndex();
        LinearScanIndex linear = new LinearScanIndex();

        for (int i = 0; i < numVectors; i++) {
            float[] vec = generateRandomVector(dim, random);
            VectorRecord record = new VectorRecord("id_" + i, vec, "payload_" + i);
            hnsw.insert(record);
            linear.insert(record);
        }

        assertEquals(numVectors, hnsw.size());
        assertEquals(numVectors, linear.size());

        int totalHits = 0;
        int totalExpected = numQueries * topK;

        for (int q = 0; q < numQueries; q++) {
            float[] query = generateRandomVector(dim, random);

            List<SearchResult> groundTruth = linear.search(query, topK);
            List<SearchResult> hnswResults = hnsw.search(query, topK);

            Set<String> groundTruthIds = new HashSet<>();
            for (SearchResult r : groundTruth) {
                groundTruthIds.add(r.record().id());
            }

            for (SearchResult r : hnswResults) {
                if (groundTruthIds.contains(r.record().id())) {
                    totalHits++;
                }
            }
        }

        double recall = (double) totalHits / totalExpected;
        System.out.println("HNSW vs LinearScan Recall@" + topK + ": " + String.format("%.2f%%", recall * 100));

        // High recall expectation: > 85%
        assertTrue(recall >= 0.85, "Recall should be at least 85%, got: " + (recall * 100) + "%");
    }

    private float[] generateRandomVector(int dim, Random random) {
        float[] v = new float[dim];
        float sumSq = 0.0f;
        for (int i = 0; i < dim; i++) {
            v[i] = (float) random.nextGaussian();
            sumSq += v[i] * v[i];
        }
        float norm = (float) Math.sqrt(sumSq);
        if (norm > 0) {
            for (int i = 0; i < dim; i++) {
                v[i] /= norm;
            }
        }
        return v;
    }
}
