package com.vector.ai.model;

/**
 * Ingestion request payload.
 */
public record IngestRequest(
    String id,
    String text
) {}
