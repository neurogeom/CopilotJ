/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Desktop;
import java.awt.FlowLayout;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
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

  // -- Standalone server tab components --
  private JTextField urlField;
  private JButton connectButton;

  // -- Managed server tab components --
  private JButton installButton;
  private JButton uninstallButton;
  private JButton serverToggleButton;
  private JButton openChatButton;
  private JButton openResourcesButton;
  private JLabel envStatusLabel;
  private JLabel managedStatusLabel;
  private JTextArea progressArea;

  // -- MCP panel --
  private JPanel mcpPanel;
  // Typed handle to the bundle's MCP server (null when MCP is unavailable on
  // this JVM, or before the panel is created). Same object as mcpPanel when set.
  private McpControl mcpControl;

  /**
   * True while the managed server is being started — keeps buttons disabled
   * against async syncUIState overwrites.
   */
  private boolean serverStarting;

  /** True while the environment is being installed. */
  private boolean installing;

  /** True while the environment is being uninstalled. */
  private boolean uninstalling;

  /** True while the environment is being synced. */
  private boolean syncing;

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
    tabbedPane.addTab("Standalone Server", buildStandaloneTab());

    // MCP panel tab (loaded dynamically)
    mcpPanel = createMcpPanel();
    tabbedPane.addTab("MCP Server", mcpPanel);

    // Default to the Managed Server tab. In debug mode the plugin auto-connects
    // to a standalone dev server, so open the Standalone Server tab instead.
    if (Boolean.getBoolean("ij.debug")) {
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

        // Stop MCP server if running (typed call via McpControl; no reflection).
        if (mcpControl != null) {
          mcpControl.stopMcp();
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

  private JPanel buildStandaloneTab() {
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

      // Warn if managed server is running — connecting to a standalone server will stop it.
      if (service.isServerRunning() && service.isManaged()) {
        final int choice = javax.swing.JOptionPane.showConfirmDialog(
            opened,
            "A managed server is currently running.\n"
                + "Connecting to a standalone server will stop it.\n"
                + "Continue?",
            "Managed Server Running",
            javax.swing.JOptionPane.YES_NO_OPTION,
            javax.swing.JOptionPane.WARNING_MESSAGE);
        if (choice != javax.swing.JOptionPane.YES_OPTION)
          return;
        service.stop();
      }

      // Warn if MCP server is running — connecting to a standalone server will stop it.
      if (mcpControl != null && mcpControl.isMcpRunning()) {
        final int choice = javax.swing.JOptionPane.showConfirmDialog(
            opened,
            "An MCP server is running.\n"
                + "Connecting to a standalone server will stop it.\n"
                + "Continue?",
            "MCP Server Running",
            javax.swing.JOptionPane.YES_NO_OPTION,
            javax.swing.JOptionPane.WARNING_MESSAGE);
        if (choice != javax.swing.JOptionPane.YES_OPTION)
          return;
        mcpControl.stopMcp();
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

    final JLabel hint = new JLabel("Connect to a standalone CopilotJ server.");
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
    installButton.addActionListener(e -> {
      if (service.isEnvironmentReady() || service.isEnvironmentOnDisk()) {
        runSyncWorker();
      } else {
        runInstallWorker();
      }
    });

    uninstallButton = new JButton("Uninstall");
    uninstallButton.addActionListener(e -> {
      final Object[] options = {"Keep my data (Recommended)", "Delete everything", "Cancel"};
      final int choice = javax.swing.JOptionPane.showOptionDialog(
          opened,
          "Choose an uninstall mode:\n\n"
              + "• Keep my data — removes the Python environment and cached resources, "
              + "but keeps your Knowledge Bank and Config.\n\n"
              + "• Delete everything — removes the entire CopilotJ home directory, "
              + "including Knowledge Bank and Config.\n\n"
              + (serverRunning() ? "The running server will be stopped.\n\n" : "")
              + "Continue?",
          "Uninstall Environment",
          javax.swing.JOptionPane.DEFAULT_OPTION,
          javax.swing.JOptionPane.WARNING_MESSAGE,
          null,
          options,
          options[0]); // default = Keep my data (recommended, non-destructive)
      // 0 = keep data, 1 = delete all, 2 / CLOSED_OPTION = cancel
      if (choice == 2 || choice == javax.swing.JOptionPane.CLOSED_OPTION)
        return;
      runUninstallWorker(choice == 0);
    });

    final JPanel envRow = new JPanel(new FlowLayout(FlowLayout.LEFT, 5, 0));
    envRow.add(new JLabel("Environment:"));
    envRow.add(envStatusLabel);
    envRow.add(Box.createHorizontalStrut(10));
    envRow.add(installButton);
    envRow.add(uninstallButton);

    // -- Server row --
    managedStatusLabel = new JLabel("Stopped");
    serverToggleButton = new JButton("Start Server");

    serverToggleButton.addActionListener(e -> {
      if (serverRunning()) {
        service.stop();
        progressArea.append("Server stopped.\n");
        syncUIState();
      } else {
        // Warn if standalone connection exists — starting managed will replace it.
        final Connection extConn = service.getConnection();
        if (extConn != null && !service.isManaged()) {
          final int choice = javax.swing.JOptionPane.showConfirmDialog(
              opened,
              "A standalone server connection exists.\n"
                  + "Starting the managed server will disconnect it.\n"
                  + "Continue?",
              "Standalone Connection Active",
              javax.swing.JOptionPane.YES_NO_OPTION,
              javax.swing.JOptionPane.WARNING_MESSAGE);
          if (choice != javax.swing.JOptionPane.YES_OPTION)
            return;
          // Stop the standalone connection immediately so it stops reconnecting.
          extConn.close();
        }

        // Warn if MCP server is running — starting managed will stop it.
        if (mcpControl != null && mcpControl.isMcpRunning()) {
          final int choice = javax.swing.JOptionPane.showConfirmDialog(
              opened,
              "An MCP server is running.\n"
                  + "Starting the managed server will stop it.\n"
                  + "Continue?",
              "MCP Server Running",
              javax.swing.JOptionPane.YES_NO_OPTION,
              javax.swing.JOptionPane.WARNING_MESSAGE);
          if (choice != javax.swing.JOptionPane.YES_OPTION)
            return;
          mcpControl.stopMcp();
        }
        runStartWorker();
      }
    });

    final JPanel serverRow = new JPanel(new FlowLayout(FlowLayout.LEFT, 5, 0));
    serverRow.add(new JLabel("Server:"));
    serverRow.add(managedStatusLabel);
    serverRow.add(Box.createHorizontalStrut(10));
    serverRow.add(serverToggleButton);

    // -- Links row --
    openChatButton = new JButton("Open copilotj.chat");
    openChatButton.setToolTipText("Open the hosted CopilotJ chat in your browser");
    openChatButton.addActionListener(e -> openInBrowser("https://copilotj.chat/#/chat"));

    openResourcesButton = new JButton("Open Resources");
    openResourcesButton.setToolTipText("Open the local resource directory ($COPILOTJ_HOME) in your file manager");
    openResourcesButton.addActionListener(e -> openInFileManager(service.getEnvironmentRoot()));

    final JPanel linksRow = new JPanel(new FlowLayout(FlowLayout.LEFT, 5, 0));
    linksRow.add(new JLabel("Links:"));
    linksRow.add(Box.createHorizontalStrut(10));
    linksRow.add(openChatButton);
    linksRow.add(openResourcesButton);

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
    topRows.add(Box.createVerticalStrut(5));
    topRows.add(linksRow);

    panel.add(topRows, BorderLayout.NORTH);
    panel.add(progressScroll, BorderLayout.CENTER);

    // Initial state sync.
    syncUIState();
    return panel;
  }

  // -- SwingWorkers for blocking operations --

  private void runInstallWorker() {
    installing = true;
    syncUIState();
    envStatusLabel.setText("Installing...");
    progressArea.setText("");

    new SwingWorker<Void, String>() {
      @Override
      protected Void doInBackground() throws Exception {
        publish("Installing Python environment...\n");
        service.ensureEnvironment(msg -> publish(msg.endsWith("\n") ? msg : msg + "\n"));
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
        installing = false;
        try {
          get();
          envStatusLabel.setText("Ready");
          progressArea.append("Environment installed successfully.\n");
        } catch (final Exception e) {
          envStatusLabel.setText("Failed");
          progressArea.append("Installation failed: " + getRootCauseMessage(e) + "\n");
        }
        syncUIState();
      }
    }.execute();
  }

  private void runSyncWorker() {
    syncing = true;
    syncUIState();
    envStatusLabel.setText("Syncing...");
    progressArea.setText("");

    new SwingWorker<Void, String>() {
      @Override
      protected Void doInBackground() throws Exception {
        publish("Syncing Python environment dependencies...\n");
        service.syncEnvironment(msg -> publish(msg.endsWith("\n") ? msg : msg + "\n"));
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
        syncing = false;
        try {
          get();
          envStatusLabel.setText("Ready");
          progressArea.append("Environment synced successfully.\n");
        } catch (final Exception e) {
          envStatusLabel.setText("Sync failed");
          progressArea.append("Sync failed: " + getRootCauseMessage(e) + "\n");
        }
        syncUIState();
      }
    }.execute();
  }

  private void runUninstallWorker(final boolean keepUserData) {
    uninstalling = true;
    syncUIState();
    envStatusLabel.setText(keepUserData ? "Removing environment..." : "Uninstalling...");
    progressArea.setText("");

    new SwingWorker<Void, String>() {
      @Override
      protected Void doInBackground() throws Exception {
        publish("Stopping server (if running)...\n");
        publish(keepUserData
            ? "Removing Python environment (keeping user data)...\n"
            : "Deleting Python environment and all data...\n");
        service.uninstallEnvironment(keepUserData);
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
        uninstalling = false;
        try {
          get();
          envStatusLabel.setText("Not installed");
          managedStatusLabel.setText("Stopped");
          progressArea.append(keepUserData
              ? "Environment removed. Your Knowledge Bank and Config were kept.\n"
              : "Environment uninstalled successfully.\n");
        } catch (final Exception e) {
          envStatusLabel.setText("Uninstall failed");
          progressArea.append("Uninstall failed: " + getRootCauseMessage(e) + "\n");
        }
        syncUIState();
      }
    }.execute();
  }

  private void runStartWorker() {
    serverStarting = true;
    syncUIState();
    managedStatusLabel.setText("Starting...");

    new SwingWorker<String, String>() {
      @Override
      protected String doInBackground() throws Exception {
        publish("Starting managed server...\n");
        service.startManagedServer(msg -> publish(msg.endsWith("\n") ? msg : msg + "\n"));
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
          progressArea.append("Start failed: " + getRootCauseMessage(e) + "\n");
        }
        syncUIState();
      }
    }.execute();
  }

  // -- State synchronization --

  private static String getRootCauseMessage(final Throwable t) {
    String message = t.getMessage();
    Throwable cause = t.getCause();
    while (cause != null) {
      if (cause.getMessage() != null && !cause.getMessage().isEmpty()) {
        message = cause.getMessage();
      }
      cause = cause.getCause();
    }
    return message != null ? message : t.getClass().getSimpleName();
  }

  /** Opens {@code url} in the system default browser, if supported. */
  private void openInBrowser(final String url) {
    try {
      if (!Desktop.isDesktopSupported() || !Desktop.getDesktop().isSupported(Desktop.Action.BROWSE)) {
        javax.swing.JOptionPane.showMessageDialog(opened,
            "Opening a browser from Fiji is not supported on this platform.\n"
                + "Open this URL manually in your browser:\n" + url,
            "Cannot Open", javax.swing.JOptionPane.INFORMATION_MESSAGE);
        return;
      }
      Desktop.getDesktop().browse(new URI(url));
    } catch (final IOException | URISyntaxException ex) {
      javax.swing.JOptionPane.showMessageDialog(opened,
          "Could not open the browser automatically (" + getRootCauseMessage(ex) + ").\n"
              + "Open this URL manually in your browser:\n" + url,
          "Error", javax.swing.JOptionPane.ERROR_MESSAGE);
    }
  }

  /** Opens {@code dir} in the system file manager, creating it if necessary. */
  private void openInFileManager(final File dir) {
    try {
      if (!Desktop.isDesktopSupported() || !Desktop.getDesktop().isSupported(Desktop.Action.OPEN)) {
        javax.swing.JOptionPane.showMessageDialog(opened,
            "Opening a folder from Fiji is not supported on this platform.\n"
                + "Open this folder manually in your file manager:\n" + dir.getAbsolutePath(),
            "Cannot Open", javax.swing.JOptionPane.INFORMATION_MESSAGE);
        return;
      }
      if (!dir.exists() && !dir.mkdirs()) {
        javax.swing.JOptionPane.showMessageDialog(opened,
            "Resource directory does not exist and could not be created:\n" + dir.getAbsolutePath(),
            "Not Found", javax.swing.JOptionPane.WARNING_MESSAGE);
        return;
      }
      Desktop.getDesktop().open(dir);
    } catch (final IOException ex) {
      javax.swing.JOptionPane.showMessageDialog(opened,
          "Could not open the folder automatically (" + getRootCauseMessage(ex) + ").\n"
              + "Open this folder manually in your file manager:\n" + dir.getAbsolutePath(),
          "Error", javax.swing.JOptionPane.ERROR_MESSAGE);
    }
  }

  private boolean serverRunning() {
    return service.isServerRunning() && service.isManaged();
  }

  private void syncUIState() {
    SwingUtilities.invokeLater(() -> {
      // Managed tab button states.
      final boolean envReady = service.isEnvironmentReady() || service.isEnvironmentOnDisk();
      final boolean running = serverRunning();
      final boolean busy = serverStarting || installing || uninstalling || syncing;

      installButton.setEnabled(!busy);
      installButton.setText(envReady ? "Sync" : "Install");
      uninstallButton.setEnabled(!busy && envReady);
      serverToggleButton.setEnabled(!busy && envReady);
      serverToggleButton.setText(running ? "Stop" : "Start");

      if (envReady && !"Ready".equals(envStatusLabel.getText()) &&
          !"Installing...".equals(envStatusLabel.getText()) &&
          !"Uninstalling...".equals(envStatusLabel.getText())) {
        envStatusLabel.setText("Ready");
      }
      if (!running && !serverStarting && !"Stopped".equals(managedStatusLabel.getText()) &&
          !"Failed".equals(managedStatusLabel.getText())) {
        managedStatusLabel.setText("Stopped");
      }

      // Standalone tab button state.
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

  // -- MCP panel --

  private JPanel createMcpPanel() {
    // MCP ships as an isolated Java 17 bundle loaded via McpLoader; on Java <17
    // (or if the bundle is missing/corrupt) it degrades to a static label.
    final McpLoader loader = new McpLoader(logService);
    // Forward mutual-exclusion guard, run right before MCP starts: if a managed
    // or standalone server is active, prompt the user and stop it so Fiji keeps a
    // single control endpoint. `opened` is captured into a local so a later
    // static null (window closed) cannot affect an in-flight start. Kept in the
    // dialog (not the bundle) so the bundle never depends on CopilotJBridgeService.
    final JFrame frame = opened;
    final java.util.function.BooleanSupplier requireExclusiveControl = () -> {
      if (!service.isServerRunning()) return true;
      final String what = service.isManaged()
          ? "managed CopilotJ server"
          : "standalone CopilotJ server connection";
      final int choice = javax.swing.JOptionPane.showConfirmDialog(
          frame,
          "A " + what + " is currently active.\n"
              + "Starting the MCP server requires a single control endpoint and will stop it.\n"
              + "Continue?",
          "CopilotJ Server Active",
          javax.swing.JOptionPane.YES_NO_OPTION,
          javax.swing.JOptionPane.WARNING_MESSAGE);
      if (choice != javax.swing.JOptionPane.YES_OPTION) return false;
      service.stop();
      return true;
    };
    // Merge MCP status into the shared status bar (only one endpoint is active
    // at a time). Null-guard + rely on McpPanel.setStatus's invokeLater: the
    // "already running" path fires during the panel's constructor, before this
    // dialog's statusLabel is built, so the actual accept() is deferred past run().
    final java.util.function.BiConsumer<String, java.awt.Color> mcpStatusUpdater = (text, color) -> {
      if (statusLabel != null) {
        statusLabel.setText("MCP: " + text);
        statusLabel.setForeground(color);
      }
      // Intentionally do NOT touch idLabel: MCP owns no ID, and resetting it
      // would race onIdChanged during endpoint transitions.
    };
    final JPanel panel = loader.createPanel(service.getEventHandler(), requireExclusiveControl, mcpStatusUpdater);
    if (panel instanceof McpControl) {
      mcpControl = (McpControl) panel;
      // Register with the service so Fiji Quit (ContextDisposingEvent) stops MCP
      // too, mirroring the managed Python server's shutdown path.
      service.setMcpControl(mcpControl);
    }
    if (panel != null) {
      return panel;
    }
    final String manualUrl = "https://copilotj.chat/#/manual#why-is-my-mcp-server-not-available";
    final JLabel label = new JLabel("<html>MCP can only run with Fiji-Latest "
        + "(requires Java 17+; current Java version: "
        + System.getProperty("java.version") + ").<br>"
        + "See the <a href=\"" + manualUrl + "\">manual FAQ</a> for details.</html>");
    label.setForeground(Color.GRAY);
    label.setCursor(java.awt.Cursor.getPredefinedCursor(java.awt.Cursor.HAND_CURSOR));
    label.addMouseListener(new java.awt.event.MouseAdapter() {
      @Override
      public void mouseClicked(final java.awt.event.MouseEvent e) {
        try {
          java.awt.Desktop.getDesktop().browse(java.net.URI.create(manualUrl));
        } catch (final Exception ex) {
          logService.warn("copilotj: Could not open manual: " + ex.getMessage());
        }
      }
    });
    final JPanel p = new JPanel();
    p.setBorder(BorderFactory.createTitledBorder("MCP Server"));
    p.add(label);
    return p;
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
