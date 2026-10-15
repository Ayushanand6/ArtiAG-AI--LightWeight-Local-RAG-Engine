package com.vector.ai.service;

import com.vector.ai.index.VectorIndex;
import com.vector.ai.model.ChatResponse;
import com.vector.ai.model.SearchResult;
import com.vector.ai.model.VectorRecord;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.List;

/**
 * Core orchestration service for document ingestion and retrieval-augmented generation (RAG).
 */
@Service
public class RagService {

    private static final Logger log = LoggerFactory.getLogger(RagService.class);

    private final VectorIndex vectorIndex;
    private final OllamaClientService ollamaClientService;
    private final TextChunker textChunker;
    private final float minSimilarity;

    public RagService(
            VectorIndex vectorIndex,
            OllamaClientService ollamaClientService,
            TextChunker textChunker,
            @Value("${rag.min-similarity:0.55}") float minSimilarity) {
        this.vectorIndex = vectorIndex;
        this.ollamaClientService = ollamaClientService;
        this.textChunker = textChunker;
        this.minSimilarity = minSimilarity;
    }

    public int ingestDocument(String id, String text) {
        if (id == null || id.isBlank()) {
            throw new IllegalArgumentException("Document id cannot be null or empty");
        }
        if (text == null || text.isBlank()) {
            throw new IllegalArgumentException("Document text cannot be null or empty");
        }

        List<String> chunks = textChunker.chunkText(text);
        if (chunks.isEmpty()) {
            throw new IllegalArgumentException("Document text produced zero valid chunks");
        }

        for (int i = 0; i < chunks.size(); i++) {
            String chunk = chunks.get(i);
            float[] embedding = ollamaClientService.getEmbeddings(chunk);
            String chunkId = id + "#" + i;
            VectorRecord record = new VectorRecord(chunkId, embedding, chunk);
            vectorIndex.insert(record);
        }
        log.info("Ingested document '{}' with {} chunk(s). Current index size: {}", id, chunks.size(), vectorIndex.size());
        return chunks.size();
    }

    public ChatResponse ask(String query, int topK) {
        if (query == null || query.isBlank()) {
            throw new IllegalArgumentException("Query cannot be null or empty");
        }

        float[] queryEmbedding = ollamaClientService.getEmbeddings(query);
        List<SearchResult> matches = vectorIndex.search(queryEmbedding, topK);

        float topScore = matches.isEmpty() ? 0.0f : matches.get(0).similarityScore();
        log.info("Query: '{}' | Top similarity score: {} | Min threshold: {}", query, topScore, minSimilarity);

        // Filter results by minSimilarity threshold guardrail
        List<SearchResult> filteredMatches = matches.stream()
                .filter(m -> m.similarityScore() >= minSimilarity)
                .toList();

        // Guardrail: If no retrieved documents meet the similarity threshold, skip the LLM entirely
        if (filteredMatches.isEmpty()) {
            log.info("No matches met similarity threshold ({}). Bypassing LLM call.", minSimilarity);
            return new ChatResponse("I could not find the answer to this in the provided documents.", List.of());
        }

        StringBuilder contextBuilder = new StringBuilder();
        for (SearchResult match : filteredMatches) {
            contextBuilder.append("Document [").append(match.record().id()).append("]:\n")
                    .append(match.record().textPayload()).append("\n\n");
        }

        String prompt = """
                Answer the question based directly on the context provided below.
                Be direct, factual, and concise.
                If the answer is not mentioned in the context, reply exactly: "I could not find the answer to this in the provided documents."

                Context:
                %s

                Question: %s
                Answer:
                """.formatted(contextBuilder.toString(), query);

        String answer = ollamaClientService.generateAnswer(prompt);
        return new ChatResponse(answer, filteredMatches);
    }

    public int getIndexSize() {
        return vectorIndex.size();
    }
}
