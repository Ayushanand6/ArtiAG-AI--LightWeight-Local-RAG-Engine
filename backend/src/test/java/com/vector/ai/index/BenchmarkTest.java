package com.vector.ai.index;

import com.vector.ai.model.SearchResult;
import com.vector.ai.model.VectorRecord;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.assertTrue;

class BenchmarkTest {

    @Test
    @DisplayName("Benchmark HNSW vs LinearScan at 1,000 and 10,000 vectors")
    void runComprehensiveBenchmark() {
        int dim = 128;
        int[] datasetSizes = new int[]{1000, 10000};
        int numQueries = 100;
        int topK = 10;

        System.out.println("==========================================================================================");
        System.out.println("                   ARTIAG VECTOR SEARCH BENCHMARK: HNSW vs LINEAR SCAN                     ");
        System.out.println("==========================================================================================");
        System.out.printf("%-10s | %-12s | %-15s | %-15s | %-10s | %-10s%n",
                "Vectors", "Queries", "LinearScan (ms)", "HNSW (ms)", "Speedup", "Recall@10");
        System.out.println("------------------------------------------------------------------------------------------");

        for (int n : datasetSizes) {
            Random random = new Random(42 + n);

            HNSWIndex hnsw = new HNSWIndex();
            LinearScanIndex linear = new LinearScanIndex();

            // Populate indices
            for (int i = 0; i < n; i++) {
                float[] v = generateNormalizedVector(dim, random);
                VectorRecord rec = new VectorRecord("id_" + i, v, "payload_" + i);
                hnsw.insert(rec);
                linear.insert(rec);
            }

            // Generate query vectors
            List<float[]> queries = new ArrayList<>();
            for (int q = 0; q < numQueries; q++) {
                queries.add(generateNormalizedVector(dim, random));
            }

            // Warmup
            for (int w = 0; w < 10; w++) {
                linear.search(queries.get(w % queries.size()), topK);
                hnsw.search(queries.get(w % queries.size()), topK);
            }

            // Benchmark LinearScan
            long tLinearStart = System.nanoTime();
            List<List<SearchResult>> groundTruths = new ArrayList<>();
            for (float[] q : queries) {
                groundTruths.add(linear.search(q, topK));
            }
            long tLinearDuration = System.nanoTime() - tLinearStart;
            double avgLinearMs = (double) tLinearDuration / (numQueries * 1_000_000.0);

            // Benchmark HNSW
            long tHnswStart = System.nanoTime();
            List<List<SearchResult>> hnswResults = new ArrayList<>();
            for (float[] q : queries) {
                hnswResults.add(hnsw.search(q, topK));
            }
            long tHnswDuration = System.nanoTime() - tHnswStart;
            double avgHnswMs = (double) tHnswDuration / (numQueries * 1_000_000.0);

            // Calculate Recall@K
            int totalHits = 0;
            int totalPossible = numQueries * topK;
            for (int q = 0; q < numQueries; q++) {
                Set<String> gtIds = new HashSet<>();
                for (SearchResult sr : groundTruths.get(q)) {
                    gtIds.add(sr.record().id());
                }
                for (SearchResult sr : hnswResults.get(q)) {
                    if (gtIds.contains(sr.record().id())) {
                        totalHits++;
                    }
                }
            }

            double recall = (double) totalHits / totalPossible;
            double speedup = (double) tLinearDuration / Math.max(1, tHnswDuration);

            System.out.printf("%-10d | %-12d | %-15.3f | %-15.3f | %-9.1fx | %-9.2f%%%n",
                    n, numQueries, avgLinearMs, avgHnswMs, speedup, recall * 100.0);

            assertTrue(recall >= 0.80, "Recall for " + n + " vectors should be at least 80%");
        }
        System.out.println("==========================================================================================");
    }

    private float[] generateNormalizedVector(int dim, Random random) {
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
