/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.mock;

import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.scijava.log.LogService;

/**
 * Tests {@link Connection}'s reconnect backoff curve and observer plumbing without
 * touching the network. State-transition / timer / WebSocketClient-callback logic is
 * Layer C (needs the real client or a factory seam).
 */
public class ConnectionTest {

  /** A listener that records every notification it receives. */
  static final class RecordingListener implements Connection.ConnectionStateListener {
    final List<Connection.State> states = new ArrayList<>();
    final List<String> messages = new ArrayList<>();

    @Override
    public void onStateChange(final Connection.State state, final String message) {
      states.add(state);
      messages.add(message);
    }
  }

  private static Connection newConnection() {
    return new Connection("http://server.example", mock(EventHandler.class), mock(LogService.class), 60);
  }

  // ===== backoff formula =====

  @Test
  public void backoffSecondsTable() {
    // k = RETRY_WAIT_INCREASED_AFTER = 3 → 2^max(retry-2, 0), capped at maxRetry.
    assertEquals(1, Connection.backoffSeconds(0, 60));
    assertEquals(1, Connection.backoffSeconds(1, 60));
    assertEquals(1, Connection.backoffSeconds(2, 60));
    assertEquals(2, Connection.backoffSeconds(3, 60));
    assertEquals(4, Connection.backoffSeconds(4, 60));
    assertEquals(8, Connection.backoffSeconds(5, 60));
    assertEquals(16, Connection.backoffSeconds(6, 60));
  }

  @Test
  public void backoffSecondsCappedAtMaxRetry() {
    assertEquals(10, Connection.backoffSeconds(6, 10), "16 capped to 10");
    assertEquals(4, Connection.backoffSeconds(6, 4), "16 capped to 4");
    assertEquals(1, Connection.backoffSeconds(0, 1), "already at floor");
  }

  // ===== observer plumbing =====

  @Test
  public void getStateDisconnectedAfterConstruction() {
    // Note: the ctor-time DISCONNECTED notification is a silent no-op (listeners list is
    // empty at that point); state is still queryable via getState().
    assertEquals(Connection.State.DISCONNECTED, newConnection().getState());
  }

  @Test
  public void registerImmediatelyNotifiesCurrentStateAndDedups() {
    final Connection c = newConnection();
    final RecordingListener l = new RecordingListener();
    c.registerStateListener(l);
    assertEquals(1, l.states.size(), "immediate notification of current state");
    assertEquals(Connection.State.DISCONNECTED, l.states.get(0));
    assertEquals("Currently disconnected.", l.messages.get(0));

    c.registerStateListener(l); // duplicate — must not double-add or re-notify
    assertEquals(1, l.states.size(), "duplicate register is a no-op");
  }

  @Test
  public void removedListenerNotNotifiedOnClose() {
    final Connection c = newConnection();
    final RecordingListener kept = new RecordingListener();
    final RecordingListener removed = new RecordingListener();
    c.registerStateListener(kept);
    c.registerStateListener(removed);
    c.removeStateListener(removed);
    c.close(); // drives a DISCONNECTED fanout to remaining listeners

    assertEquals(1, removed.states.size(), "removed listener only has its register-time notification");
    assertEquals(2, kept.states.size(), "kept listener: register + close");
    assertEquals(Connection.State.DISCONNECTED, kept.states.get(1));
    assertEquals("Disconnected by user.", kept.messages.get(1));
  }

  @Test
  public void closeNotifiesAllRegisteredListeners() {
    final Connection c = newConnection();
    final RecordingListener a = new RecordingListener();
    final RecordingListener b = new RecordingListener();
    c.registerStateListener(a);
    c.registerStateListener(b);
    c.close();

    // each: 1 register-time + 1 close = 2
    assertEquals(2, a.states.size());
    assertEquals(2, b.states.size());
  }
}
