package com.vector.ai.model;

import com.fasterxml.jackson.annotation.JsonIgnore;
import java.util.Arrays;
import java.util.Objects;

/**
 * Represents an indexed vector document chunk.
 * The embedding vector is marked @JsonIgnore so that raw float arrays are not leaked in API responses.
 */
public record VectorRecord(
    String id,
    @JsonIgnore float[] embedding,
    String textPayload
) {
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof VectorRecord that)) return false;
        return Objects.equals(id, that.id) &&
               Arrays.equals(embedding, that.embedding) &&
               Objects.equals(textPayload, that.textPayload);
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(id, textPayload);
        result = 31 * result + Arrays.hashCode(embedding);
        return result;
    }
}
