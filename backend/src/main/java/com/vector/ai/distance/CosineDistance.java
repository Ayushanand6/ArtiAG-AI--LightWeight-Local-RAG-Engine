package com.vector.ai.distance;

/**
 * Utility for vector distance and similarity calculations.
 */
public class CosineDistance {

    /**
     * Calculates cosine similarity between two float vectors.
     * Formula: dot(v1, v2) / (||v1|| * ||v2||)
     * Handles zero vectors by returning 0.0f.
     * Throws IllegalArgumentException on null inputs or length mismatch.
     */
    public static float cosineSimilarity(float[] v1, float[] v2) {
        if (v1 == null || v2 == null) {
            throw new IllegalArgumentException("Vector arrays cannot be null");
        }
        if (v1.length != v2.length) {
            throw new IllegalArgumentException("Vector length mismatch: " + v1.length + " vs " + v2.length);
        }

        float dotProduct = 0.0f;
        float normA = 0.0f;
        float normB = 0.0f;

        for (int i = 0; i < v1.length; i++) {
            dotProduct += v1[i] * v2[i];
            normA += v1[i] * v1[i];
            normB += v2[i] * v2[i];
        }

        if (normA <= 0.0f || normB <= 0.0f) {
            // Zero vector: cosine similarity is mathematically undefined, return 0.0f to avoid NaN
            return 0.0f;
        }

        return (float) (dotProduct / (Math.sqrt(normA) * Math.sqrt(normB)));
    }
}
