package com.vector.ai.model;

/**
 * Response payload for document ingestion.
 */
public record IngestResponse(
    String status,
    int chunksIndexed,
    int totalChunks
) {}
