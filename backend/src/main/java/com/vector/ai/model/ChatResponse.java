package com.vector.ai.model;

import java.util.List;

/**
 * Chat response containing the generated answer from the LLM and the matched context documents.
 */
public record ChatResponse(
    String answer,
    List<SearchResult> matches
) {}
