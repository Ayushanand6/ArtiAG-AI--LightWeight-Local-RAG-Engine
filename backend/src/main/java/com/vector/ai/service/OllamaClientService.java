package com.vector.ai.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.net.ConnectException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.Map;

/**
 * Service to interact with the local Ollama instance for embeddings and completions.
 * Reads configurations via @Value from application.properties.
 * Uses a single centralized post() helper for robust HTTP calls and error handling.
 */
@Service
public class OllamaClientService {

    private final HttpClient httpClient;
    private final ObjectMapper objectMapper;
    private final String baseUrl;
    private final String embedModel;
    private final String chatModel;
    private final int timeoutSeconds;

    public OllamaClientService(
            @Value("${ollama.base-url}") String baseUrl,
            @Value("${ollama.embed-model}") String embedModel,
            @Value("${ollama.chat-model}") String chatModel,
            @Value("${ollama.timeout-seconds}") int timeoutSeconds,
            ObjectMapper objectMapper) {
        this.baseUrl = baseUrl;
        this.embedModel = embedModel;
        this.chatModel = chatModel;
        this.timeoutSeconds = timeoutSeconds;
        this.objectMapper = objectMapper;
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(timeoutSeconds))
                .build();
    }

    public float[] getEmbeddings(String text) {
        if (text == null || text.isBlank()) {
            throw new IllegalArgumentException("Text for embedding cannot be null or empty");
        }
        try {
            Map<String, Object> payload = Map.of(
                    "model", embedModel,
                    "prompt", text
            );
            String jsonRequest = objectMapper.writeValueAsString(payload);
            String responseBody = post("/api/embeddings", jsonRequest);

            JsonNode root = objectMapper.readTree(responseBody);
            JsonNode embeddingNode = root.get("embedding");
            if (embeddingNode == null || !embeddingNode.isArray() || embeddingNode.isEmpty()) {
                throw new IllegalStateException("Embedding not found or empty in Ollama response");
            }

            float[] embedding = new float[embeddingNode.size()];
            for (int i = 0; i < embeddingNode.size(); i++) {
                embedding[i] = (float) embeddingNode.get(i).asDouble();
            }
            return embedding;
        } catch (JsonProcessingException e) {
            throw new IllegalStateException("Failed to parse Ollama embedding response: " + e.getMessage(), e);
        }
    }

    public String generateAnswer(String prompt) {
        if (prompt == null || prompt.isBlank()) {
            throw new IllegalArgumentException("Prompt cannot be null or empty");
        }
        try {
            Map<String, Object> payload = Map.of(
                    "model", chatModel,
                    "prompt", prompt,
                    "stream", false
            );
            String jsonRequest = objectMapper.writeValueAsString(payload);
            String responseBody = post("/api/generate", jsonRequest);

            JsonNode root = objectMapper.readTree(responseBody);
            JsonNode responseNode = root.get("response");
            if (responseNode == null || responseNode.asText().isBlank()) {
                throw new IllegalStateException("LLM response not found or empty in Ollama output");
            }
            return responseNode.asText();
        } catch (JsonProcessingException e) {
            throw new IllegalStateException("Failed to parse Ollama generation response: " + e.getMessage(), e);
        }
    }

    /**
     * Centralized HTTP POST helper:
     * - Configures endpoint URI, request timeout, and JSON content-type header
     * - Checks HTTP status != 200
     * - Translates ConnectException -> "Is Ollama running?"
     * - Restores interrupt flag on InterruptedException
     */
    private String post(String endpoint, String jsonBody) {
        try {
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(baseUrl + endpoint))
                    .timeout(Duration.ofSeconds(timeoutSeconds))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(jsonBody))
                    .build();

            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
            if (response.statusCode() != 200) {
                throw new IllegalStateException("Ollama API failed with HTTP " + response.statusCode() + ": " + response.body());
            }
            return response.body();
        } catch (ConnectException e) {
            throw new IllegalStateException("Cannot connect to Ollama. Is Ollama running on " + baseUrl + "?", e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Request to Ollama was interrupted", e);
        } catch (IOException e) {
            if (e.getCause() instanceof ConnectException) {
                throw new IllegalStateException("Cannot connect to Ollama. Is Ollama running on " + baseUrl + "?", e);
            }
            throw new IllegalStateException("IO error communicating with Ollama: " + e.getMessage(), e);
        }
    }
}
