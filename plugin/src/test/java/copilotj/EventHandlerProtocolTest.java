/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.time.LocalDateTime;
import java.util.LinkedHashMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import copilotj.ImagejListener.HistoryRequest;
import copilotj.ScreenCapturer.CaptureImageRequest;
import copilotj.ScriptRunner.PluginRequest;
import copilotj.ScriptRunner.Result;
import copilotj.ScriptRunner.ScriptRequest;
import copilotj.SnapshotManager.ActionRequest;
import copilotj.SnapshotManager.CompareRequest;
import copilotj.Summerizer.EnvironmentSummary;
import copilotj.util.Trie;

/**
 * Protocol tests for {@link EventHandler}'s JSON wire format. Exercises the exact
 * ObjectMapper configuration production uses via the package-private
 * {@link EventHandler#configureMapper(ObjectMapper)} seam, WITHOUT ever constructing an
 * EventHandler — its constructor cascades into ImageJ statics and cannot run headlessly.
 * The dispatch {@code handle(Payload)} is covered by Layer C.
 */
public class EventHandlerProtocolTest {

  private final ObjectMapper mapper = EventHandler.configureMapper(new ObjectMapper());

  /** Serialize to a tree for stable, substring-free key assertions. */
  private JsonNode ser(final Object value) throws Exception {
    return mapper.readTree(mapper.writeValueAsString(value));
  }

  // ===== Payload =====

  @Test
  public void payloadSerializesSnakeCaseKeys() throws Exception {
    final EventHandler.Payload p = new EventHandler.Payload();
    p.id = "id-1";
    p.event_id = "eid-1";
    p.event = "take_snapshot";
    final JsonNode tree = ser(p);
    assertEquals("id-1", tree.get("id").asText());
    assertEquals("eid-1", tree.get("event_id").asText());
    assertEquals("take_snapshot", tree.get("event").asText());
    assertFalse(tree.has("data") && !tree.get("data").isNull(), "absent data omitted");
  }

  @Test
  public void payloadRoundTripsThroughSnakeCase() throws Exception {
    final String json = "{\"event\":\"run_script\",\"event_id\":\"e1\",\"id\":\"i1\","
        + "\"data\":{\"x\":1},\"err\":null}";
    final EventHandler.Payload p = mapper.readValue(json, EventHandler.Payload.class);
    assertEquals("run_script", p.event);
    assertEquals("e1", p.event_id);
    assertEquals("i1", p.id);
    assertNotNull(p.data);
    assertEquals(1, p.data.get("x").asInt());
    assertNull(p.err);
  }

  // ===== IdChanged =====

  @Test
  public void idChangedRoundTrip() throws Exception {
    final EventHandler.IdChanged ic = mapper.readValue("{\"id\":\"abc\"}", EventHandler.IdChanged.class);
    assertEquals("abc", ic.id);
  }

  // ===== SnapshotManager DTOs =====

  @Test
  public void compareRequestSnakeCase() throws Exception {
    final CompareRequest req = mapper.readValue("{\"id_early\":2,\"id_later\":5}", CompareRequest.class);
    assertEquals(Integer.valueOf(2), req.idEarly);
    assertEquals(Integer.valueOf(5), req.idLater);
    assertTrue(ser(req).has("id_early"));
    assertTrue(ser(req).has("id_later"));
  }

  @Test
  public void actionRequestSnakeCase() throws Exception {
    final ActionRequest req = mapper.readValue("{\"snapshot_id\":3,\"action_id\":7,\"parameters\":[\"a\",1]}",
        ActionRequest.class);
    assertEquals(3, req.snapshotId);
    assertEquals(7, req.actionId);
    assertEquals(2, req.parameters.size());
    assertEquals("a", req.parameters.get(0));
    assertEquals(Integer.valueOf(1), req.parameters.get(1));
    assertTrue(ser(req).has("snapshot_id"));
    assertTrue(ser(req).has("action_id"));
  }

  // ===== ScreenCapturer / ScriptRunner DTOs =====

  @Test
  public void captureImageRequestRoundTrip() throws Exception {
    final CaptureImageRequest req = mapper.readValue("{\"title\":\"foo.png\"}", CaptureImageRequest.class);
    assertEquals("foo.png", req.title);
  }

  @Test
  public void scriptRequestRoundTrip() throws Exception {
    final ScriptRequest req =
        mapper.readValue("{\"language\":\"groovy\",\"script\":\"print 1\",\"timeout\":30}", ScriptRequest.class);
    assertEquals("groovy", req.language);
    assertEquals("print 1", req.script);
    assertEquals(30, req.timeout);
  }

  @Test
  public void pluginRequestSnakeCaseIj1Style() throws Exception {
    final PluginRequest req = mapper.readValue("{\"plugin\":\"P\",\"ij1_style\":false}", PluginRequest.class);
    assertEquals("P", req.plugin);
    assertFalse(req.ij1Style);
    assertTrue(ser(req).has("ij1_style"));
  }

  @Test
  public void resultSerializesErrKey() throws Exception {
    final JsonNode tree = ser(new Result("boom"));
    assertEquals("boom", tree.get("err").asText());
  }

  // ===== Summerizer.EnvironmentSummary (serialize-only; ctor-based) =====

  @Test
  public void environmentSummarySerializesSnakeCase() throws Exception {
    final Map<String, Trie.SentenceTrie> plugins = new LinkedHashMap<>();
    final JsonNode tree = ser(new EnvironmentSummary("/ij", "/java", 21, plugins));
    assertEquals("/ij", tree.get("imagej_home").asText());
    assertEquals("/java", tree.get("java_home").asText());
    assertEquals(21, tree.get("java_version").asInt());
    assertTrue(tree.has("plugins"));
  }

  // ===== ImagejListener.HistoryRequest (FlexibleLocalDateTimeDeserializer) =====

  @Test
  public void historyRequestParsesIsoDate() throws Exception {
    final HistoryRequest req = mapper.readValue("{\"since\":\"2026-06-30T12:34:56\"}", HistoryRequest.class);
    assertNotNull(req.since);
    assertEquals(LocalDateTime.of(2026, 6, 30, 12, 34, 56), req.since);
  }

  @Test
  public void historyRequestParsesIsoWithFractional() throws Exception {
    // The flexible deserializer accepts fractional seconds (and other variants).
    final HistoryRequest req = mapper.readValue("{\"since\":\"2026-06-30T12:34:56.123\"}", HistoryRequest.class);
    assertNotNull(req.since);
  }

  // ===== connected-event construction (exercises the real buildConnectedPayload branch) =====

  @Test
  public void connectedPayloadWithIdIsNegotiateId() {
    final EventHandler.Payload p = EventHandler.buildConnectedPayload("plugin-42", mapper);
    assertEquals("negotiate_id", p.event);
    assertNotNull(p.data);
    assertEquals("plugin-42", p.data.get("id").asText());
  }

  @Test
  public void connectedPayloadWithoutIdIsQueryId() {
    final EventHandler.Payload p = EventHandler.buildConnectedPayload(null, mapper);
    assertEquals("query_id", p.event);
    assertNull(p.data);
  }

  @Test
  public void connectedPayloadWithEmptyIdIsQueryId() {
    // The branch treats an empty id the same as null (no negotiation).
    final EventHandler.Payload p = EventHandler.buildConnectedPayload("", mapper);
    assertEquals("query_id", p.event);
    assertNull(p.data);
  }
}
