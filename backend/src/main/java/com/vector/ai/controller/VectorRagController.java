package com.vector.ai.controller;

import com.vector.ai.model.ChatRequest;
import com.vector.ai.model.ChatResponse;
import com.vector.ai.model.IngestRequest;
import com.vector.ai.model.IngestResponse;
import com.vector.ai.service.IndexPersistenceService;
import com.vector.ai.service.RagService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

/**
 * REST API controller for document ingestion, query retrieval, and index persistence.
 */
@RestController
@RequestMapping("/api")
@CrossOrigin(origins = "*")
public class VectorRagController {

    private static final Logger log = LoggerFactory.getLogger(VectorRagController.class);

    private final RagService ragService;
    private final IndexPersistenceService persistenceService;

    public VectorRagController(RagService ragService, IndexPersistenceService persistenceService) {
        this.ragService = ragService;
        this.persistenceService = persistenceService;
    }

    @PostMapping("/documents/ingest")
    public ResponseEntity<IngestResponse> ingest(@RequestBody IngestRequest request) {
        long startTime = System.currentTimeMillis();

        if (request == null) {
            throw new IllegalArgumentException("Request body cannot be null");
        }
        if (request.id() == null || request.id().isBlank()) {
            throw new IllegalArgumentException("Document id cannot be null or empty");
        }
        if (request.text() == null || request.text().isBlank()) {
            throw new IllegalArgumentException("Document text cannot be null or empty");
        }

        int chunksIndexed = ragService.ingestDocument(request.id().trim(), request.text().trim());
        long elapsedMs = System.currentTimeMillis() - startTime;
        log.info("Ingest completed for document '{}': {} chunk(s) indexed in {} ms", request.id(), chunksIndexed, elapsedMs);

        return ResponseEntity.ok(new IngestResponse("SUCCESS", chunksIndexed, chunksIndexed));
    }

    @PostMapping("/chat/query")
    public ResponseEntity<ChatResponse> query(@RequestBody ChatRequest request) {
        long startTime = System.currentTimeMillis();

        if (request == null) {
            throw new IllegalArgumentException("Request body cannot be null");
        }
        if (request.query() == null || request.query().isBlank()) {
            throw new IllegalArgumentException("Query cannot be null or empty");
        }
        if (request.topK() != null && (request.topK() < 1 || request.topK() > 20)) {
            throw new IllegalArgumentException("topK must be between 1 and 20, got: " + request.topK());
        }

        int topK = (request.topK() != null) ? request.topK() : 3;
        ChatResponse response = ragService.ask(request.query().trim(), topK);

        long elapsedMs = System.currentTimeMillis() - startTime;
        log.info("Query '{}' processed in {} ms with {} match(es)", request.query(), elapsedMs, response.matches().size());

        return ResponseEntity.ok(response);
    }

    @PostMapping("/index/save")
    public ResponseEntity<Map<String, Object>> saveIndex() {
        long startTime = System.currentTimeMillis();
        int count = persistenceService.saveIndex();
        long elapsedMs = System.currentTimeMillis() - startTime;
        log.info("Persisted index with {} records in {} ms", count, elapsedMs);
        return ResponseEntity.ok(Map.of("status", "SUCCESS", "recordsSaved", count));
    }

    @PostMapping("/index/load")
    public ResponseEntity<Map<String, Object>> loadIndex() {
        long startTime = System.currentTimeMillis();
        int count = persistenceService.loadIndex();
        long elapsedMs = System.currentTimeMillis() - startTime;
        log.info("Loaded index with {} records in {} ms", count, elapsedMs);
        return ResponseEntity.ok(Map.of("status", "SUCCESS", "recordsLoaded", count));
    }

    @GetMapping("/status")
    public ResponseEntity<Map<String, Object>> getStatus() {
        return ResponseEntity.ok(Map.of(
                "status", "UP",
                "indexedVectors", ragService.getIndexSize(),
                "engine", "ArtiAG Lightweight HNSW",
                "embedModel", "nomic-embed-text",
                "chatModel", "llama3.2:1b"
        ));
    }
}
