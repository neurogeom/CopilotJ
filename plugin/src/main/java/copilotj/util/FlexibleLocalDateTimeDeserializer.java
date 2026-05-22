/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj.util;

import java.io.IOException;
import java.time.LocalDateTime;
import java.time.OffsetDateTime;
import java.time.ZoneId;
import java.time.format.DateTimeParseException;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JsonDeserializer;

/**
 * Deserializes a datetime string into {@link LocalDateTime}, tolerating both
 * naive ("2026-06-18T10:00:00") and offset-aware ("2026-06-18T10:00:00Z",
 * "2026-06-18T10:00:00+00:00") ISO 8601 forms. This mirrors what LLM-driven
 * MCP clients tend to emit, whereas Jackson's default
 * {@code LocalDateTimeDeserializer} rejects any offset.
 * <p>
 * Offset-aware values are converted to the system-local timeline so they
 * compare correctly against operation timestamps recorded with
 * {@link LocalDateTime#now()}.
 */
public class FlexibleLocalDateTimeDeserializer extends JsonDeserializer<LocalDateTime> {

  @Override
  public LocalDateTime deserialize(final JsonParser p, final DeserializationContext ctxt) throws IOException {
    final String text = p.getValueAsString();
    if (text == null) {
      return null;
    }
    final String trimmed = text.trim();
    if (trimmed.isEmpty()) {
      return null;
    }

    // Offset-aware (trailing 'Z' or an explicit +HH:MM/-HH:MM) → instant → system-local.
    try {
      return OffsetDateTime.parse(trimmed)
          .toInstant()
          .atZone(ZoneId.systemDefault())
          .toLocalDateTime();
    } catch (final DateTimeParseException notOffsetAware) {
      // Fall back to a naive local datetime; let this throw if it is genuinely invalid.
      return LocalDateTime.parse(trimmed);
    }
  }
}
