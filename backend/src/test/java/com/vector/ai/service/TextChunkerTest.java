package com.vector.ai.service;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class TextChunkerTest {

    @Test
    @DisplayName("Should return empty list for null or empty text")
    void shouldHandleNullOrEmptyText() {
        TextChunker chunker = new TextChunker(200, 40);
        assertTrue(chunker.chunkText(null).isEmpty());
        assertTrue(chunker.chunkText("").isEmpty());
        assertTrue(chunker.chunkText("    ").isEmpty());
    }

    @Test
    @DisplayName("Should return single chunk when text has fewer words than chunkSize")
    void shouldReturnSingleChunkForShortText() {
        TextChunker chunker = new TextChunker(200, 40);
        String text = "Java 21 introduces virtual threads and records.";
        List<String> chunks = chunker.chunkText(text);

        assertEquals(1, chunks.size());
        assertEquals(text, chunks.get(0));
    }

    @Test
    @DisplayName("Should properly split with overlap")
    void shouldSplitWithOverlap() {
        // chunkSize = 10, overlap = 3, step = 7
        TextChunker chunker = new TextChunker(10, 3);
        // 17 words: w0 to w16
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 17; i++) {
            if (i > 0) sb.append(" ");
            sb.append("w").append(i);
        }
        List<String> chunks = chunker.chunkText(sb.toString());

        // Chunk 0: w0 to w9 (10 words)
        // Chunk 1: w7 to w16 (10 words, starting at 10 - 3 = 7)
        assertEquals(2, chunks.size());
        assertTrue(chunks.get(0).startsWith("w0") && chunks.get(0).endsWith("w9"));
        assertTrue(chunks.get(1).startsWith("w7") && chunks.get(1).endsWith("w16"));
    }

    @Test
    @DisplayName("Should throw IllegalArgumentException when overlap >= chunkSize")
    void shouldRejectInvalidOverlap() {
        assertThrows(IllegalArgumentException.class, () -> new TextChunker(50, 50));
        assertThrows(IllegalArgumentException.class, () -> new TextChunker(50, 60));
        assertThrows(IllegalArgumentException.class, () -> new TextChunker(-1, 0));
        assertThrows(IllegalArgumentException.class, () -> new TextChunker(50, -5));
    }
}
