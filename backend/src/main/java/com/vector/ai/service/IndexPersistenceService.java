package com.vector.ai.service;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.vector.ai.index.HNSWIndex;
import com.vector.ai.model.VectorRecord;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

/**
 * Service responsible for persisting and restoring indexed vector records to/from disk (data/index.json).
 *
 * Architecture Decision & Trade-off:
 * - We persist the raw VectorRecords (id, float[] embedding, textPayload).
 * - On startup, we rebuild the HNSW graph by sequentially inserting the records.
 * - Trade-off:
 *     Pros: File format is compact, portable, and decoupled from graph internal pointers/parameters (M, ef).
 *     Cons: Startup time includes graph reconstruction (negligible for tens of thousands of records).
 */
@Service
public class IndexPersistenceService {

    private static final Logger log = LoggerFactory.getLogger(IndexPersistenceService.class);
    private static final Path DATA_DIR = Paths.get("data");
    private static final Path INDEX_FILE = DATA_DIR.resolve("index.json");

    private final HNSWIndex hnswIndex;
    private final ObjectMapper objectMapper;

    public record StoredRecord(String id, float[] embedding, String textPayload) {}

    public IndexPersistenceService(HNSWIndex hnswIndex, ObjectMapper objectMapper) {
        this.hnswIndex = hnswIndex;
        this.objectMapper = objectMapper;
    }

    @PostConstruct
    public void onStartup() {
        loadIndex();
    }

    @PreDestroy
    public void onShutdown() {
        saveIndex();
    }

    public synchronized int saveIndex() {
        List<VectorRecord> records = hnswIndex.getAllRecords();
        if (records.isEmpty()) {
            log.info("Index is empty, skipping persistence.");
            return 0;
        }

        try {
            if (!Files.exists(DATA_DIR)) {
                Files.createDirectories(DATA_DIR);
            }

            List<StoredRecord> stored = records.stream()
                    .map(r -> new StoredRecord(r.id(), r.embedding(), r.textPayload()))
                    .toList();

            objectMapper.writerWithDefaultPrettyPrinter().writeValue(INDEX_FILE.toFile(), stored);
            log.info("Successfully persisted {} vector records to {}", stored.size(), INDEX_FILE);
            return stored.size();
        } catch (IOException e) {
            log.error("Failed to persist vector index to file {}: {}", INDEX_FILE, e.getMessage(), e);
            throw new IllegalStateException("Failed to persist vector index: " + e.getMessage(), e);
        }
    }

    public synchronized int loadIndex() {
        File file = INDEX_FILE.toFile();
        if (!file.exists() || file.length() == 0) {
            log.info("No persisted index file found at {}. Starting with empty index.", INDEX_FILE);
            return 0;
        }

        try {
            List<StoredRecord> stored = objectMapper.readValue(file, new TypeReference<List<StoredRecord>>() {});
            hnswIndex.clear();
            for (StoredRecord sr : stored) {
                VectorRecord record = new VectorRecord(sr.id(), sr.embedding(), sr.textPayload());
                hnswIndex.insert(record);
            }
            log.info("Successfully loaded {} vector records from {} and rebuilt HNSW graph", stored.size(), INDEX_FILE);
            return stored.size();
        } catch (IOException e) {
            log.error("Failed to load vector index from {}: {}", INDEX_FILE, e.getMessage(), e);
            throw new IllegalStateException("Failed to load vector index: " + e.getMessage(), e);
        }
    }
}
