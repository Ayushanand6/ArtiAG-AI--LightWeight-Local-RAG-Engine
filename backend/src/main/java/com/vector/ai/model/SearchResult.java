package com.vector.ai.model;

/**
 * Result of a similarity search containing the matched vector record and its cosine similarity score.
 * Implements Comparable by similarityScore (natural order is ascending, which makes PriorityQueue a min-heap).
 */
public record SearchResult(
    VectorRecord record,
    float similarityScore
) implements Comparable<SearchResult> {

    @Override
    public int compareTo(SearchResult other) {
        return Float.compare(this.similarityScore, other.similarityScore);
    }
}
