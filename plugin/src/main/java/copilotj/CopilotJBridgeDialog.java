/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTextField;
import javax.swing.SwingUtilities;

import org.scijava.command.Command;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.ui.UIService;

@Plugin(type = Command.class, menuPath = "Plugins>CopilotJ")
public class CopilotJBridgeDialog implements Command, Connection.ConnectionStateListener, EventHandler.IdListener {
  private static JFrame opened = null;

  @Parameter
  private org.scijava.Context context;

  @Parameter
  private CopilotJBridgeService service;

  @Parameter
  private UIService uiService;

  @Parameter
  private LogService logService;

  private JButton connectButton;
  private JLabel statusLabel; // Make statusLabel accessible in the class
  private JLabel idLabel; // Label to display the connection ID

  @Override
  public void run() {
    if (opened != null) {
      logService.debug("CopilotJ Bridge Config dialog is already open");
      opened.toFront();
      return;
    }

    final JFrame frame = new JFrame("CopilotJ Bridge Config");
    opened = frame;

    frame.setSize(400, 200);
    frame.setLayout(new BorderLayout());

    final JTextField urlField = new JTextField(service.getServerUrl());
    final Connection existingConn = service.getConnection();
    final boolean isActive = existingConn != null &&
        existingConn.getState() != Connection.State.DISCONNECTED;
    connectButton = new JButton(isActive ? "Disconnect" : "(Re)Connect");
    statusLabel = new JLabel("Initializing..."); // Initial status
    idLabel = new JLabel("ID: N/A"); // Initial ID status

    connectButton.addActionListener(e -> {
      if ("Disconnect".equals(connectButton.getText())) {
        final Connection conn = service.getConnection();
        if (conn != null) conn.close();
        connectButton.setText("(Re)Connect");
        return;
      }

      final String newUrl = urlField.getText();
      logService.info("Attempting to connect to: " + newUrl);
      service.start(newUrl); // This will create a new connection or restart the existing one
      connectButton.setText("Disconnect");

      // Register listeners when new connection is created
      final Connection connection = service.getConnection();
      if (connection != null) {
        connection.registerStateListener(this);
      } else {
        onStateChange(Connection.State.DISCONNECTED, "Currently disconnected.");
      }
    });

    // Register listeners when dialog opens
    final Connection connection = service.getConnection();
    if (connection != null) {
      connection.registerStateListener(this);
    } else {
      onStateChange(Connection.State.DISCONNECTED, "Currently disconnected.");
    }

    final JPanel inputPanel = new JPanel(new BorderLayout(5, 5)); // Panel for URL and button
    inputPanel.add(new JLabel("Server URL:"), BorderLayout.WEST);
    inputPanel.add(urlField, BorderLayout.CENTER);
    inputPanel.add(connectButton, BorderLayout.EAST);

    final JPanel statusPanel = new JPanel(new BorderLayout(5, 5)); // Panel for status and ID
    statusPanel.add(statusLabel, BorderLayout.NORTH);
    statusPanel.add(idLabel, BorderLayout.SOUTH);

    final JPanel mainPanel = new JPanel(new BorderLayout(10, 10)); // Main content panel
    mainPanel.setBorder(javax.swing.BorderFactory.createEmptyBorder(10, 10, 10, 10)); // Add padding
    mainPanel.add(inputPanel, BorderLayout.NORTH);
    mainPanel.add(statusPanel, BorderLayout.CENTER);

    // Register ID listener
    final EventHandler handler = service.getEventHandler();
    if (handler != null) {
      handler.addListener(this);
    } else {
      logService.warn("EventHandler is null, cannot register IdListener.");
    }

    // Unregister listener when dialog closes
    frame.addWindowListener(new WindowAdapter() {
      @Override
      public void windowClosing(final WindowEvent e) {
        final Connection conn = service.getConnection();
        if (conn != null) {
          conn.removeStateListener(CopilotJBridgeDialog.this);
          logService.debug("Removed connection state listener from dialog.");
          if ("Disconnect".equals(connectButton.getText())) {
            conn.close();
          }
        }

        final EventHandler handler = service.getEventHandler();
        if (handler != null) {
          handler.removeListener(CopilotJBridgeDialog.this);
          logService.debug("Removed ID listener from dialog.");
        }

        frame.dispose();

        opened = null;
      }
    });

    frame.add(mainPanel, BorderLayout.CENTER);
    frame.pack(); // Adjust size to fit components
    frame.setLocationRelativeTo(null); // Center on screen
    frame.setVisible(true);
  }

  @Override
  public void onStateChange(final Connection.State state, final String message) {
    // Ensure UI updates are done on the Event Dispatch Thread
    SwingUtilities.invokeLater(() -> {
      if (statusLabel != null) {
        statusLabel.setText("Status: " + state + " - " + message);
      }

      // Optionally change color based on state
      if (statusLabel != null) {
        switch (state) {
          case CONNECTED:
            statusLabel.setForeground(new Color(0, 128, 0)); // Green
            break;
          case CONNECTING:
          case RECONNECTING:
            statusLabel.setForeground(Color.ORANGE);
            break;

          case DISCONNECTED:
          case ERROR:
            statusLabel.setForeground(Color.RED);
            break;

          default:
            statusLabel.setForeground(Color.BLACK);
            break;
        }
      } else {
        logService.warn("statusLabel is null during state change: " + state);
      }
    });
  }

  @Override
  public void onIdChanged(final String newId) {
    // Ensure UI updates are done on the Event Dispatch Thread
    SwingUtilities.invokeLater(() -> {
      if (idLabel == null) {
        logService.warn("idLabel is null during ID change: " + newId);
        return;
      }

      idLabel.setText("ID: " + newId);
    });
  }
}
