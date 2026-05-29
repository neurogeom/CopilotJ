/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.FlowLayout;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.util.List;

import javax.swing.BorderFactory;
import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTabbedPane;
import javax.swing.JTextArea;
import javax.swing.JTextField;
import javax.swing.SwingUtilities;
import javax.swing.SwingWorker;

import org.scijava.command.Command;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.ui.UIService;

@Plugin(type = Command.class, menuPath = "Plugins>CopilotJ")
public class CopilotJBridgeDialog
    implements Command, Connection.ConnectionStateListener, EventHandler.IdListener {
  private static JFrame opened = null;

  @Parameter
  private org.scijava.Context context;

  @Parameter
  private CopilotJBridgeService service;

  @Parameter
  private UIService uiService;

  @Parameter
  private LogService logService;

  // -- Shared components --
  private JLabel statusLabel;
  private JLabel idLabel;
  private JTabbedPane tabbedPane;

  // -- External server tab components --
  private JTextField urlField;
  private JButton connectButton;

  // -- Managed server tab components --
  private JButton installButton;
  private JButton serverToggleButton;
  private JLabel envStatusLabel;
  private JLabel managedStatusLabel;
  private JTextArea progressArea;

  /**
   * True while the managed server is being started — keeps buttons disabled
   * against async syncUIState overwrites.
   */
  private boolean serverStarting;

  /** The connection this dialog is currently listening to. */
  private Connection listenedConnection;

  @Override
  public void run() {
    if (opened != null) {
      logService.debug("CopilotJ Bridge Config dialog is already open");
      opened.toFront();
      return;
    }

    final JFrame frame = new JFrame("CopilotJ Bridge Config");
    opened = frame;
    frame.setMinimumSize(new java.awt.Dimension(520, 320));
    frame.setLayout(new BorderLayout(10, 10));

    // -- Tabbed pane --
    tabbedPane = new JTabbedPane();
    tabbedPane.addTab("Managed Server", buildManagedTab());
    tabbedPane.addTab("External Server", buildExternalTab());

    // Select External tab if currently connected externally.
    if (!service.isManaged()) {
      tabbedPane.setSelectedIndex(1);
    }

    // -- Shared status panel (below tabs) --
    statusLabel = new JLabel("Status: disconnected");
    idLabel = new JLabel("ID: N/A");
    final JPanel statusPanel = new JPanel(new BorderLayout(5, 5));
    statusPanel.add(statusLabel, BorderLayout.NORTH);
    statusPanel.add(idLabel, BorderLayout.SOUTH);

    final JPanel mainPanel = new JPanel(new BorderLayout(10, 10));
    mainPanel.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));
    mainPanel.add(tabbedPane, BorderLayout.CENTER);
    mainPanel.add(statusPanel, BorderLayout.SOUTH);

    // -- Register listeners --
    final Connection connection = service.getConnection();
    if (connection != null) {
      switchConnectionListener(connection);
      syncUIState();
    }

    final EventHandler handler = service.getEventHandler();
    if (handler != null) {
      handler.addListener(this);
    }

    // -- Window cleanup --
    frame.addWindowListener(new WindowAdapter() {
      @Override
      public void windowClosing(final WindowEvent e) {
        if (listenedConnection != null) {
          listenedConnection.removeStateListener(CopilotJBridgeDialog.this);
        }
        final Connection conn = service.getConnection();
        if (conn != null && !service.isManaged()) {
          conn.close();
        }
        final EventHandler h = service.getEventHandler();
        if (h != null) {
          h.removeListener(CopilotJBridgeDialog.this);
        }
        frame.dispose();
        opened = null;
      }
    });

    frame.add(mainPanel, BorderLayout.CENTER);
    frame.pack();
    frame.setLocationRelativeTo(null);
    frame.setVisible(true);
  }

  // -- Tab builders --

  private JPanel buildExternalTab() {
    final JPanel panel = new JPanel(new BorderLayout(10, 10));
    panel.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

    urlField = new JTextField(service.getServerUrl(), 25);
    connectButton = new JButton("(Re)Connect");

    // Sync button state with current connection.
    final Connection conn = service.getConnection();
    final boolean isActive = conn != null &&
        conn.getState() != Connection.State.DISCONNECTED && !service.isManaged();
    connectButton.setText(isActive ? "Disconnect" : "(Re)Connect");

    connectButton.addActionListener(e -> {
      if ("Disconnect".equals(connectButton.getText())) {
        final Connection c = service.getConnection();
        if (c != null)
          c.close();
        connectButton.setText("(Re)Connect");
        return;
      }

      // Warn if managed server is running — connecting externally will stop it.
      if (service.isServerRunning() && service.isManaged()) {
        final int choice = javax.swing.JOptionPane.showConfirmDialog(
            opened,
            "A managed server is currently running.\n"
                + "Connecting to an external server will stop it.\n"
                + "Continue?",
            "Managed Server Running",
            javax.swing.JOptionPane.YES_NO_OPTION,
            javax.swing.JOptionPane.WARNING_MESSAGE);
        if (choice != javax.swing.JOptionPane.YES_OPTION)
          return;
        service.stop();
      }

      final String url = urlField.getText().trim();
      if (url.isEmpty())
        return;
      logService.info("Connecting to: " + url);
      service.start(url);
      connectButton.setText("Disconnect");

      final Connection c = service.getConnection();
      if (c != null) {
        switchConnectionListener(c);
      }
    });

    final JPanel inputRow = new JPanel(new BorderLayout(5, 0));
    inputRow.add(new JLabel("Server URL:"), BorderLayout.WEST);
    inputRow.add(urlField, BorderLayout.CENTER);
    inputRow.add(connectButton, BorderLayout.EAST);

    final JLabel hint = new JLabel("Connect to an externally running CopilotJ server.");
    hint.setForeground(Color.GRAY);

    panel.add(inputRow, BorderLayout.NORTH);
    panel.add(hint, BorderLayout.CENTER);
    return panel;
  }

  private JPanel buildManagedTab() {
    final JPanel panel = new JPanel(new BorderLayout(10, 10));
    panel.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

    // -- Environment row --
    envStatusLabel = new JLabel("Not installed");
    installButton = new JButton("Install");
    installButton.addActionListener(e -> runInstallWorker());

    final JPanel envRow = new JPanel(new FlowLayout(FlowLayout.LEFT, 5, 0));
    envRow.add(new JLabel("Environment:"));
    envRow.add(envStatusLabel);
    envRow.add(Box.createHorizontalStrut(10));
    envRow.add(installButton);

    // -- Server row --
    managedStatusLabel = new JLabel("Stopped");
    serverToggleButton = new JButton("Start Server");

    serverToggleButton.addActionListener(e -> {
      if (serverRunning()) {
        service.stop();
        progressArea.append("Server stopped.\n");
        syncUIState();
      } else {
        // Warn if external connection exists — starting managed will replace it.
        final Connection extConn = service.getConnection();
        if (extConn != null && !service.isManaged()) {
          final int choice = javax.swing.JOptionPane.showConfirmDialog(
              opened,
              "An external server connection exists.\n"
                  + "Starting the managed server will disconnect it.\n"
                  + "Continue?",
              "External Connection Active",
              javax.swing.JOptionPane.YES_NO_OPTION,
              javax.swing.JOptionPane.WARNING_MESSAGE);
          if (choice != javax.swing.JOptionPane.YES_OPTION)
            return;
          // Stop the external connection immediately so it stops reconnecting.
          extConn.close();
        }
        runStartWorker();
      }
    });

    final JPanel serverRow = new JPanel(new FlowLayout(FlowLayout.LEFT, 5, 0));
    serverRow.add(new JLabel("Server:"));
    serverRow.add(managedStatusLabel);
    serverRow.add(Box.createHorizontalStrut(10));
    serverRow.add(serverToggleButton);

    // -- Progress area --
    progressArea = new JTextArea(6, 40);
    progressArea.setEditable(false);
    progressArea.setFont(new java.awt.Font("Monospaced", java.awt.Font.PLAIN, 11));
    final javax.swing.JScrollPane progressScroll = new javax.swing.JScrollPane(progressArea);
    progressScroll.setBorder(BorderFactory.createTitledBorder("Progress Log"));

    // -- Layout --
    final JPanel topRows = new JPanel();
    topRows.setLayout(new BoxLayout(topRows, BoxLayout.Y_AXIS));
    topRows.add(envRow);
    topRows.add(Box.createVerticalStrut(5));
    topRows.add(serverRow);

    panel.add(topRows, BorderLayout.NORTH);
    panel.add(progressScroll, BorderLayout.CENTER);

    // Initial state sync.
    syncUIState();
    return panel;
  }

  // -- SwingWorkers for blocking operations --

  private void runInstallWorker() {
    installButton.setEnabled(false);
    envStatusLabel.setText("Installing...");
    progressArea.setText("");

    new SwingWorker<Void, String>() {
      @Override
      protected Void doInBackground() throws Exception {
        publish("Installing Python environment...\n");
        service.ensureEnvironment();
        return null;
      }

      @Override
      protected void process(final List<String> chunks) {
        for (final String msg : chunks) {
          progressArea.append(msg);
        }
      }

      @Override
      protected void done() {
        try {
          get();
          envStatusLabel.setText("Ready");
          progressArea.append("Environment installed successfully.\n");
        } catch (final Exception e) {
          envStatusLabel.setText("Failed");
          progressArea.append("Installation failed: " + e.getMessage() + "\n");
          installButton.setEnabled(true);
        }
        syncUIState();
      }
    }.execute();
  }

  private void runStartWorker() {
    serverStarting = true;
    serverToggleButton.setEnabled(false);
    managedStatusLabel.setText("Starting...");

    new SwingWorker<String, String>() {
      @Override
      protected String doInBackground() throws Exception {
        service.startManagedServer();
        return service.getServerUrl();
      }

      @Override
      protected void process(final List<String> chunks) {
        for (final String msg : chunks) {
          progressArea.append(msg);
        }
      }

      @Override
      protected void done() {
        serverStarting = false;
        try {
          final String url = get();
          managedStatusLabel.setText("Running at " + url);
          progressArea.append("Server started at " + url + "\n");

          // Switch listener from old (closed) connection to the new managed connection.
          switchConnectionListener(service.getConnection());
        } catch (final Exception e) {
          managedStatusLabel.setText("Failed");
          progressArea.append("Start failed: " + e.getMessage() + "\n");
        }
        syncUIState();
      }
    }.execute();
  }

  // -- State synchronization --

  private boolean serverRunning() {
    return service.isServerRunning() && service.isManaged();
  }

  private void syncUIState() {
    SwingUtilities.invokeLater(() -> {
      // Managed tab button states.
      final boolean envReady = service.isEnvironmentReady() || service.isEnvironmentOnDisk();
      final boolean running = serverRunning();

      installButton.setEnabled(!serverStarting && !envReady);
      serverToggleButton.setEnabled(!serverStarting && envReady);
      serverToggleButton.setText(running ? "Stop Server" : "Start Server");

      if (envReady && !"Ready".equals(envStatusLabel.getText()) &&
          !"Installing...".equals(envStatusLabel.getText())) {
        envStatusLabel.setText("Ready");
      }
      if (!running && !serverStarting && !"Stopped".equals(managedStatusLabel.getText()) &&
          !"Failed".equals(managedStatusLabel.getText())) {
        managedStatusLabel.setText("Stopped");
      }

      // External tab button state.
      if (connectButton != null) {
        if (serverStarting) {
          connectButton.setEnabled(false);
        } else {
          connectButton.setEnabled(true);
          final Connection conn = service.getConnection();
          final boolean isActive = conn != null &&
              conn.getState() != Connection.State.DISCONNECTED && !service.isManaged();
          connectButton.setText(isActive ? "Disconnect" : "(Re)Connect");
        }
      }
    });
  }

  // -- Listener management --

  private void switchConnectionListener(final Connection conn) {
    if (listenedConnection != null && listenedConnection != conn) {
      listenedConnection.removeStateListener(this);
    }
    listenedConnection = conn;
    if (conn != null) {
      conn.registerStateListener(this);
    }
  }

  // -- Connection.ConnectionStateListener --

  @Override
  public void onStateChange(final Connection.State state, final String message) {
    SwingUtilities.invokeLater(() -> {
      if (statusLabel != null) {
        statusLabel.setText("Status: " + state + " - " + message);
        switch (state) {
          case CONNECTED:
            statusLabel.setForeground(new Color(0, 128, 0));
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
      }
      syncUIState();
    });
  }

  // -- EventHandler.IdListener --

  @Override
  public void onIdChanged(final String newId) {
    SwingUtilities.invokeLater(() -> {
      if (idLabel != null) {
        idLabel.setText("ID: " + newId);
      }
    });
  }
}
