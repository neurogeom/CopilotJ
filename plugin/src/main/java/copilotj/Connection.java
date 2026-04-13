/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj;

import java.net.URI;
import java.net.URISyntaxException;
import java.util.List;
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.CopyOnWriteArrayList;

import org.java_websocket.client.WebSocketClient;
import org.java_websocket.handshake.ServerHandshake;

import org.scijava.log.LogService;

class Connection {
  public enum State {
    DISCONNECTED,
    CONNECTING,
    CONNECTED,
    RECONNECTING,
    ERROR
  }

  public interface ConnectionStateListener {
    void onStateChange(State state, String message);
  }

  private final EventHandler handler;
  private final LogService log;

  private final int maxRetryWaitSecond;
  private final int retryWaitSecondIncreasedAfter = 3;

  private WebSocketClient webSocketClient;
  private int retryCount = 0;
  private Timer timer;

  private final String server;

  public Connection(final String serverURL, final EventHandler handler, final LogService log,
      final int maxRetryWaitSecond) {
    this.server = serverURL;
    this.handler = handler;
    this.log = log;
    this.maxRetryWaitSecond = maxRetryWaitSecond;
    Connection.notifyStateChange(this, State.DISCONNECTED, "Initialized");
  }

  public void connect() {
    Connection.notifyStateChange(this, State.CONNECTING, "Attempting to connect to " + server);
    this.webSocketClient = configure(); // WebSocketClient is not reusable
    if (webSocketClient == null) {
      final String errorMsg = "Failed to configure WebSocket client for " + server;
      log.error(errorMsg);
      Connection.notifyStateChange(this, State.ERROR, errorMsg);
      return;
    }
    webSocketClient.connect();
  }

  public void send(final String message) {
    if (webSocketClient == null) {
      log.error("WebSocket client is not connected");
      return;
    }
    webSocketClient.send(message);
  }

  public void close() {
    // Set state to DISCONNECTED before closing so that the async onClose callback
    // sees currentState == DISCONNECTED and skips handleReconnect().
    Connection.notifyStateChange(this, State.DISCONNECTED, "Disconnected by user.");

    if (webSocketClient != null) {
      webSocketClient.close();
      webSocketClient = null;
      log.debug("WebSocket client closed");
    }

    if (timer != null) {
      timer.cancel();
      timer = null;
    }

    log.debug("Connection closed: " + server);
  }

  private WebSocketClient configure() {
    try {
      final String addr = server.replace("http://", "ws://").replace("https://", "wss://");
      if (addr.endsWith("/")) {
        addr.substring(0, addr.length() - 1);
      }
      final URI uri = new URI(addr + "/api/plugins");

      return new WebSocketClient(uri) {
        @Override
        public void onOpen(final ServerHandshake handshakedata) {
          log.info("Connected to WebSocket server: " + uri);
          retryCount = 0;
          Connection.notifyStateChange(Connection.this, State.CONNECTED, "Connection established to " + uri);

          // send a ping to check if the connection is alive
          webSocketClient.sendPing();

          // Notify the handler of the new connection
          final String event = handler.newConnectedEvent();
          this.send(event);
        }

        @Override
        public void onMessage(final String message) {
          log.debug("Receive message: " + message);

          final Thread workerThread = new Thread(() -> {
            final String result = handler.handle(message);
            if (result == null) {
              return;
            }
            send(result);
          });

          workerThread.start();
        }

        @Override
        public void onClose(final int code, final String reason, final boolean remote) {
          final String closeReason = reason.isEmpty() ? "No reason provided (code: " + code + ")" : reason;
          if (currentState != State.DISCONNECTED) { // Avoid redundant logs/notifications if close() was called
            if (retryCount == 0) {
              log.info("WebSocket connection closed: " + closeReason);
            }
            Connection.notifyStateChange(Connection.this, State.DISCONNECTED, "Connection closed: " + closeReason);
            handleReconnect();
          } else {
            log.debug("WebSocket closed explicitly or already disconnected.");
          }
        }

        @Override
        public void onError(final Exception e) {
          String errorMsg;
          if (e.getMessage() != null && e.getMessage().contains("Connection refused")) {
            errorMsg = "Connection refused by server " + uri;
            if (retryCount == 0) {
              log.warn(errorMsg);
            } else {
              log.info(errorMsg + " (retry #" + retryCount + ")");
            }
          } else {
            errorMsg = "WebSocket error: " + e.getMessage();
            log.error(errorMsg, e); // Log the full exception
          }
          // Avoid ERROR state if we are already trying to reconnect
          if (currentState != State.RECONNECTING) {
            Connection.notifyStateChange(Connection.this, State.ERROR, errorMsg);
          }
        }
      };
    } catch (final URISyntaxException e) {
      e.printStackTrace();
      return null;
    }
  }

  private void handleReconnect() {
    if (timer != null) {
      return;
    }

    // Only attempt reconnect if not explicitly closed
    if (currentState == State.DISCONNECTED && retryCount > 0 && webSocketClient == null) {
      log.debug("Reconnect cancelled as connection was explicitly closed.");
      return;
    }

    final int k = this.retryWaitSecondIncreasedAfter;
    final int wait = Math.min((int) Math.pow(2, Math.max(retryCount - k + 1, 0)), maxRetryWaitSecond);
    final String reconnectMsg = "Will attempt to reconnect #" + (retryCount + 1) + " after " + wait + " second...";
    log.info(reconnectMsg);
    Connection.notifyStateChange(this, State.RECONNECTING, reconnectMsg);

    timer = new Timer("WebSocketReconnectTimer"); // Give the timer thread a name
    timer.schedule(new TimerTask() {
      @Override
      public void run() {
        if (webSocketClient == null || webSocketClient.isOpen()) { // closed or connected
          return;
        }

        if (retryCount < k) {
          log.info("Attempting to reconnect #" + (retryCount + 1));
        }
        timer = null;
        retryCount++;
        connect(); // This will trigger CONNECTING state notification
      }
    }, 1000L * wait);
  }

  // --- State Management ---

  private final List<ConnectionStateListener> stateListeners = new CopyOnWriteArrayList<>(); // Thread-safe for
                                                                                             // listeners
  private volatile State currentState = State.DISCONNECTED; // Track current state

  public State getState() {
    return currentState;
  }

  public void registerStateListener(final ConnectionStateListener listener) {
    if (listener != null && !stateListeners.contains(listener)) {
      stateListeners.add(listener);
      // Immediately notify the new listener of the current state
      listener.onStateChange(currentState, getCurrentStateMessage());
    }
  }

  public void removeStateListener(final ConnectionStateListener listener) {
    if (listener != null) {
      stateListeners.remove(listener);
    }
  }

  private static void notifyStateChange(final Connection connection, final State newState, final String message) {
    connection.currentState = newState;
    connection.log.debug("Connection State Change: " + newState + " - " + message); // Debug log for state changes
    for (final ConnectionStateListener listener : connection.stateListeners) {
      try {
        listener.onStateChange(newState, message);
      } catch (final Exception e) {
        connection.log.error("Error notifying ConnectionStateListener: " + e.getMessage(), e);
      }
    }
  }

  private String getCurrentStateMessage() {
    // Provide a default message based on the current state for immediate
    // notification
    switch (currentState) {
      case DISCONNECTED:
        return "Currently disconnected.";
      case CONNECTING:
        return "Currently connecting to " + server + "...";
      case CONNECTED:
        return "Currently connected to " + server + ".";
      case RECONNECTING:
        return "Attempting to reconnect..."; // More specific message is sent during the actual event
      case ERROR:
        return "An error occurred."; // More specific message sent during the actual event
      default:
        return "Unknown state.";
    }
  }
}
