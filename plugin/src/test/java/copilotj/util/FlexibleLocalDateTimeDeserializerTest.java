/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj.util;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.time.LocalDateTime;
import java.time.OffsetDateTime;
import java.time.ZoneId;

import org.junit.jupiter.api.Test;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.module.SimpleModule;

/**
 * Unit tests for {@link FlexibleLocalDateTimeDeserializer}. Exercised through a
 * real {@link ObjectMapper} (no mock) so the production {@code JsonParser} path
 * runs. Expected offset-aware values are computed via the same chain the
 * implementation uses, so the assertions are zone-agnostic.
 */
public class FlexibleLocalDateTimeDeserializerTest {

  /** Plain static holder (NOT a record — tests compile under Java 8). */
  public static final class Holder {
    @JsonProperty("t")
    public LocalDateTime t;
  }

  private static ObjectMapper mapper() {
    final ObjectMapper m = new ObjectMapper();
    final SimpleModule module = new SimpleModule();
    module.addDeserializer(LocalDateTime.class, new FlexibleLocalDateTimeDeserializer());
    m.registerModule(module);
    return m;
  }

  private static LocalDateTime parse(final String jsonValue) throws Exception {
    return mapper().readValue("{\"t\":\"" + jsonValue + "\"}", Holder.class).t;
  }

  private static LocalDateTime localAtSystemZone(final String offsetDateTime) {
    return OffsetDateTime.parse(offsetDateTime).toInstant().atZone(ZoneId.systemDefault()).toLocalDateTime();
  }

  @Test
  public void parsesNaiveLocalDateTime() throws Exception {
    assertEquals(LocalDateTime.of(2026, 6, 18, 10, 0, 0), parse("2026-06-18T10:00:00"));
  }

  @Test
  public void parsesUtcZ() throws Exception {
    final String s = "2026-06-18T10:00:00Z";
    assertEquals(localAtSystemZone(s), parse(s));
  }

  @Test
  public void parsesExplicitPositiveOffset() throws Exception {
    final String s = "2026-06-18T18:00:00+08:00";
    assertEquals(localAtSystemZone(s), parse(s));
  }

  @Test
  public void parsesNegativeOffset() throws Exception {
    final String s = "2026-06-18T05:00:00-05:00";
    assertEquals(localAtSystemZone(s), parse(s));
  }

  @Test
  public void trimsWhitespace() throws Exception {
    assertEquals(LocalDateTime.of(2026, 6, 18, 10, 0, 0), parse("  2026-06-18T10:00:00  "));
  }

  @Test
  public void returnsNullForJsonNull() throws Exception {
    final Holder h = mapper().readValue("{\"t\":null}", Holder.class);
    assertNull(h.t);
  }

  @Test
  public void returnsNullForEmptyAfterTrim() throws Exception {
    assertNull(parse("   "));
  }

  @Test
  public void rejectsDateOnly() {
    // No time component -> LocalDateTime.parse throws, propagating out of readValue.
    assertThrows(Exception.class, () -> parse("2026-06-18"));
  }

  @Test
  public void rejectsGarbage() {
    assertThrows(Exception.class, () -> parse("not-a-date"));
  }
}
