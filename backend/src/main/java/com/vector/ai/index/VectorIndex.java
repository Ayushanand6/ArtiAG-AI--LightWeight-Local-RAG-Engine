package com.vector.ai.index;

import com.vector.ai.model.SearchResult;
import com.vector.ai.model.VectorRecord;
import java.util.List;

/**
 * Common contract for vector indexing and similarity search.
 */
public interface VectorIndex {

    /**
     * Inserts a vector record into the index.
     */
    void insert(VectorRecord record);

    /**
     * Finds the top-K most similar records to the given query embedding.
     */
    List<SearchResult> search(float[] queryEmbedding, int topK);

    /**
     * Returns the total number of indexed vectors.
     */
    int size();
}
