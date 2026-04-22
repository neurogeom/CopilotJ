/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj;

import java.io.IOException;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.CopyOnWriteArrayList;

import org.scijava.Context;
import org.scijava.log.LogService;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.MapperFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.PropertyNamingStrategy;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;

import copilotj.awt.Action;
import copilotj.awt.Snapshot;
import copilotj.awt.window.IjImage;
import copilotj.util.JsonBase64ImageTruncator;

public class EventHandler {
  public interface IdListener {
    void onIdChanged(String newId);
  }

  public static class Payload {
    public String id;
    public String event_id;
    public String event;
    public JsonNode data;
    public String err;

    public Payload() {
    }

    private Payload(final String event, final ObjectNode data) {
      this.id = UUID.randomUUID().toString();
      this.event_id = UUID.randomUUID().toString();
      this.event = event;
      this.data = data;
      this.err = null;
    }

    private Payload(final Payload request, final JsonNode data) {
      this.id = UUID.randomUUID().toString();
      this.event_id = request.event_id;
      this.event = "resp:" + request.event;
      this.data = data;
      this.err = null;
    }

    private Payload(final Payload request, final String err) {
      this.id = UUID.randomUUID().toString();
      this.event_id = request.event_id;
      this.event = "resp:" + request.event;
      this.data = null;
      this.err = err;
    }
  }

  // --- Dependencies & State ---
  private final LogService log;
  private final ScreenCapturer screenCapturer;
  private final ScriptRunner scriptRunner;
  private final SnapshotManager snapshotManager;
  private final Summerizer summerizer;

  private final ObjectMapper objectMapper = new ObjectMapper()
      .registerModule(new JavaTimeModule())
      .configure(MapperFeature.SORT_PROPERTIES_ALPHABETICALLY, true)
      .configure(SerializationFeature.WRITE_DATES_AS_TIMESTAMPS, false)
      .setPropertyNamingStrategy(PropertyNamingStrategy.SNAKE_CASE);

  public EventHandler(final Context context, final LogService log, final boolean debug) {
    this.log = log;
    this.snapshotManager = new SnapshotManager(log, debug);
    this.screenCapturer = new ScreenCapturer(this.snapshotManager.getIdentifier());
    this.scriptRunner = new ScriptRunner(context, log);
    this.summerizer = new Summerizer(context, log, this.snapshotManager);
  }

  // --- ID Listener ---

  private String id;
  private final List<IdListener> idListeners = new CopyOnWriteArrayList<>(); // Thread-safe list

  public static class IdChanged {
    public String id;

    public IdChanged() {
    }

    private IdChanged(final String id) {
      this.id = id;
    }
  }

  public String getId() {
    return this.id;
  }

  public void addListener(final IdListener listener) {
    System.out.println("new listener: " + listener.getClass().getName());
    if (listener != null && !idListeners.contains(listener)) {
      idListeners.add(listener);
      listener.onIdChanged(this.id); // Immediately notify the new listener
    }
  }

  public void removeListener(final IdListener listener) {
    if (listener != null) {
      idListeners.remove(listener);
    }
  }

  public String newAuthEvent(final String accessToken) {
    final ObjectNode data = objectMapper.createObjectNode();
    data.put("access_token", accessToken);
    final Payload payload = new Payload("auth", data);
    try {
      return objectMapper.writeValueAsString(payload);
    } catch (final IOException e) {
      log.error("Error serializing auth event: " + e.getMessage());
      return null;
    }
  }

  public String newConnectedEvent() {
    final Payload payload = this.id != null && !this.id.isEmpty()
        ? new Payload("negotiate_id", objectMapper.valueToTree(new IdChanged(this.id)))
        : new Payload("query_id", null);

    try {
      final String respStr = objectMapper.writeValueAsString(payload);
      log.debug(new JsonBase64ImageTruncator.WithTitle("Response", respStr));
      return respStr;

    } catch (final IOException e) {
      log.error("Error serialize response: " + e.getMessage());
      return "Internal server error";
    }
  }

  private void onIdChanged(final String newId) {
    this.id = newId;
    for (final IdListener listener : idListeners) {
      try {
        listener.onIdChanged(this.id);
      } catch (final Exception e) {
        log.error("Error notifying IdListener: " + e.getMessage(), e);
      }
    }
  }

  // --- Message Handling ---

  public String handle(final String message) {
    Payload payload;
    try {
      payload = objectMapper.readValue(message, Payload.class);

    } catch (final IOException e) {
      final String err = "Error parsing request: " + e.getMessage();
      log.error(err);
      return err;
    }

    Payload resp;
    try {
      final JsonNode data = handle(payload);
      if (data == null) {
        return null;
      }
      resp = new Payload(payload, data);

    } catch (final IllegalArgumentException e) {
      e.printStackTrace();
      resp = new Payload(payload, "Error handling message: " + e.getMessage());
      log.debug("Invalid argument: " + resp.err);

    } catch (final NullPointerException e) {
      e.printStackTrace();
      resp = new Payload(payload, "Unexpected null value");
      log.error("Null pointer: " + e.getMessage());

    } catch (final Exception e) {
      e.printStackTrace();
      resp = new Payload(payload, "Unexpected error");
      log.error("Unexpected error: " + resp.err);
    }

    try {
      final String respStr = objectMapper.writeValueAsString(resp);
      log.debug(new JsonBase64ImageTruncator.WithTitle("Response", respStr));
      return respStr;

    } catch (final IOException e) {
      log.error("Error serialize response: " + e.getMessage());
      return "Internal server error";
    }
  }

  private JsonNode handle(final Payload payload) {
    switch (payload.event) {
      case "id_changed": {
        if (payload.err != null) {
          log.warn("Id changed: " + payload.err);
        }

        final String newId = payload.data.get("id").asText();
        if (newId == null || newId.isEmpty()) {
          log.warn("Received 'connected' event with empty or null ID.");
        }

        log.info("Connection ID received: " + newId);
        onIdChanged(newId); // Notify listeners about the new ID
        return null;
      }

      case "auth_ok": {
        log.info("Authentication successful");
        return null;
      }

      case "capture_screen": {
        final ScreenCapturer.ScreenPreviews image = screenCapturer.captureScreen();
        return objectMapper.valueToTree(image);
      }

      case "capture_image": {
        final ScreenCapturer.CaptureImageRequest request = objectMapper.convertValue(payload.data,
            ScreenCapturer.CaptureImageRequest.class);
        final IjImage.ImagePreview image = screenCapturer.captureImage(request);
        return objectMapper.valueToTree(image);
      }

      case "compare_snapshots": {
        final SnapshotManager.CompareRequest request = objectMapper.convertValue(payload.data,
            SnapshotManager.CompareRequest.class);
        final Snapshot.Difference diff = snapshotManager.compare(request);
        return objectMapper.valueToTree(diff);
      }

      case "get_operation_history": {
        final ImagejListener.HistoryRequest request = objectMapper.convertValue(payload.data,
            ImagejListener.HistoryRequest.class);
        final ImagejListener.HistoryResponse history = snapshotManager.getListener().getMessages(request);
        return objectMapper.valueToTree(history);
      }

      case "run_action": {
        final SnapshotManager.ActionRequest request = objectMapper.convertValue(payload.data,
            SnapshotManager.ActionRequest.class);
        final Action.Response result = snapshotManager.runAction(request);
        return objectMapper.valueToTree(result);
      }

      case "run_script": {
        final ScriptRunner.ScriptRequest request = objectMapper.convertValue(payload.data,
            ScriptRunner.ScriptRequest.class);
        final ScriptRunner.Result result = scriptRunner.runScript(request);
        return objectMapper.valueToTree(result);
      }

      case "summarise_environment": {
        final Summerizer.EnvironmentSummary summary = summerizer.summariseEnvironment();
        return objectMapper.valueToTree(summary);
      }

      case "take_snapshot": {
        final Snapshot snapshot = snapshotManager.capture();
        return objectMapper.valueToTree(snapshot);
      }

      default: {
        throw new IllegalArgumentException("Unhandled event: " + payload.event);
      }
    }
  }
}
