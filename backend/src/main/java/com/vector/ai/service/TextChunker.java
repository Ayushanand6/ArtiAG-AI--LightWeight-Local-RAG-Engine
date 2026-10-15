package com.vector.ai.service;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

/**
 * Service for chunking text using a word-based sliding window approach with overlap.
 */
@Service
public class TextChunker {

    private final int chunkSize;
    private final int chunkOverlap;

    public TextChunker(
            @Value("${rag.chunk-size:200}") int chunkSize,
            @Value("${rag.chunk-overlap:40}") int chunkOverlap) {
        if (chunkSize <= 0) {
            throw new IllegalArgumentException("chunkSize must be greater than 0, got: " + chunkSize);
        }
        if (chunkOverlap < 0) {
            throw new IllegalArgumentException("chunkOverlap cannot be negative, got: " + chunkOverlap);
        }
        if (chunkOverlap >= chunkSize) {
            throw new IllegalArgumentException("chunkOverlap (" + chunkOverlap + ") must be less than chunkSize (" + chunkSize + ")");
        }
        this.chunkSize = chunkSize;
        this.chunkOverlap = chunkOverlap;
    }

    /**
     * Splits input text into overlapping chunks based on word counts.
     */
    public List<String> chunkText(String text) {
        if (text == null || text.isBlank()) {
            return List.of();
        }

        String[] words = text.trim().split("\\s+");
        if (words.length == 0 || (words.length == 1 && words[0].isEmpty())) {
            return List.of();
        }

        List<String> chunks = new ArrayList<>();
        int step = chunkSize - chunkOverlap;

        for (int i = 0; i < words.length; i += step) {
            int end = Math.min(i + chunkSize, words.length);
            StringBuilder sb = new StringBuilder();
            for (int j = i; j < end; j++) {
                if (j > i) {
                    sb.append(" ");
                }
                sb.append(words[j]);
            }
            chunks.add(sb.toString());
            if (end == words.length) {
                break;
            }
        }

        return chunks;
    }

    public int getChunkSize() {
        return chunkSize;
    }

    public int getChunkOverlap() {
        return chunkOverlap;
    }
}
