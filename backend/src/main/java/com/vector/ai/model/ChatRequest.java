package com.vector.ai.model;

/**
 * Chat query request payload with default topK = 3.
 */
public record ChatRequest(
    String query,
    Integer topK
) {
    public ChatRequest {
        if (topK == null || topK <= 0) {
            topK = 3;
        }
    }
}
