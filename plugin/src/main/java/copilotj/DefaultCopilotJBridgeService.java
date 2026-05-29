/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.Collectors;

import org.apposed.appose.Appose;
import org.apposed.appose.BuildException;
import org.apposed.appose.Environment;
import org.apposed.appose.builder.Builders;
import org.scijava.Context;
import org.scijava.Priority;
import org.scijava.event.EventService;
import org.scijava.event.ContextDisposingEvent;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.service.AbstractService;
import org.scijava.service.Service;
import org.scijava.ui.UIService;
import org.scijava.ui.event.UIShownEvent;

import net.imagej.ImageJ;
import net.imagej.legacy.LegacyService;
import net.imagej.patcher.LegacyInjector;

@Plugin(type = Service.class, priority = Priority.LOW)
public class DefaultCopilotJBridgeService extends AbstractService implements CopilotJBridgeService {
  static {
    LegacyInjector.preinit();
  }

  /**
   * This main function serves for development purposes.
   * It allows you to run the plugin immediately out of
   * your integrated development environment (IDE).
   *
   * @param args whatever, it's ignored
   * @throws Exception
   */
  public static void main(final String... args) throws Exception {
    final ImageJ ij = new ImageJ();
    ij.launch(args);
  }

  @Parameter
  private Context context;

  @Parameter
  private UIService ui;

  @Parameter
  private EventService eventService;

  // NOTE: DONT DELETE: This is needed to initialize ImageJ1
  @Parameter
  private LegacyService legacy;

  @Parameter
  private LogService log;

  /** URL used in debug mode when connecting to a manually started server. */
  private final String debugServerUrl = "http://127.0.0.1:8786";

  private EventHandler eventHandler;
  private Connection connection;

  // -- Appose-managed server state --

  private volatile Environment apposeEnv;
  private org.apposed.appose.Service pythonService;
  private volatile String managedServerUrl;
  private volatile boolean managed = false;

  public DefaultCopilotJBridgeService() {
  }

  @Override
  public EventHandler getEventHandler() {
    return eventHandler;
  }

  @Override
  public String getServerUrl() {
    return managed ? managedServerUrl : debugServerUrl;
  }

  @Override
  public Connection getConnection() {
    return connection;
  }

  @Override
  public void initialize() {
    log.info("Initialize CopilotJBridge");

    final boolean debug = Boolean.getBoolean("ij.debug");
    this.eventHandler = new EventHandler(context, log, debug);

    if (debug) {
      // Debug mode: auto-connect to manually started server.
      log.info("Debug mode: connecting to " + debugServerUrl);
      this.start(debugServerUrl);
    }
    // Production mode: user starts managed server via dialog buttons.

    if (ui.isVisible()) {
      installActionTool();
    } else {
      eventService.subscribe(this);
    }
  }

  @Override
  public void start(final String serverURL) {
    if (this.connection != null) {
      this.connection.close();
      this.connection = null;
    }

    log.info("copilotj: Start connection " + serverURL);
    final int maxRetryWaitSecond = Integer.getInteger("copilotj.maxRetryWaitSecond", 64);
    this.connection = new Connection(serverURL, eventHandler, log, maxRetryWaitSecond);
    this.connection.connect();
  }

  @Override
  public void stop() {
    // 1. Close WebSocket connection.
    if (connection != null) {
      connection.close();
      connection = null;
    }

    // 2. Stop managed Python process via Appose.
    if (pythonService != null) {
      try {
        pythonService.close(); // graceful: closes stdin
        pythonService.waitFor();
      } catch (final InterruptedException e) {
        pythonService.kill(); // force kill if interrupted
        Thread.currentThread().interrupt();
      }
      pythonService = null;
    }

    managedServerUrl = null;
    managed = false;
  }

  @Override
  public void dispose() {
    stop();
  }

  // -- Appose lifecycle --

  @Override
  public boolean isEnvironmentReady() {
    return apposeEnv != null;
  }

  @Override
  public boolean isEnvironmentOnDisk() {
    if (apposeEnv != null)
      return true;
    final File envRoot = resolveEnvRoot();
    return Builders.canWrap(new File(envRoot, ".venv"));
  }

  @Override
  public boolean isManaged() {
    return managed;
  }

  @Override
  public boolean isServerRunning() {
    return connection != null && connection.getState() != Connection.State.DISCONNECTED;
  }

  @Override
  public void startManagedServer() throws IOException, InterruptedException {
    // 1. Ensure environment is ready (also sets PYTHONPATH + .env on builder).
    ensureEnvironment();

    // 2. Create Appose Service.
    final Environment env = apposeEnv;
    pythonService = env.python();

    // 3. Send init task: Python starts server on port 0, returns port.
    final String script = "from copilotj.appose_worker import start_server\n"
        + "task.outputs.update(start_server())\n";

    final org.apposed.appose.Service.Task initTask = pythonService.task(script);
    try {
      initTask.waitFor();
    } catch (final org.apposed.appose.TaskException e) {
      throw new IOException("Python server task failed: " + e.getMessage(), e);
    }

    // 4. Extract port from task response.
    final Object portObj = initTask.outputs.get("port");
    if (portObj == null) {
      throw new IOException("Python server did not return a port");
    }
    final int port = ((Number) portObj).intValue();
    managedServerUrl = "http://127.0.0.1:" + port;
    managed = true;

    log.info("copilotj: Python server started on port " + port);

    // 5. Connect WebSocket.
    this.start(managedServerUrl);
  }

  @Override
  public void ensureEnvironment() throws IOException {
    if (apposeEnv != null)
      return;

    final File envRoot = resolveEnvRoot();

    // Read pyproject.toml from JAR resources.
    final String pyprojectContent = readResource("/copilotj-env/pyproject.toml");
    if (pyprojectContent == null) {
      throw new IOException("Bundled pyproject.toml not found in JAR resources");
    }

    // Also write .env template if not present.
    final File dotEnv = new File(envRoot, ".env");
    if (!dotEnv.exists()) {
      final String template = readResource("/copilotj-env/.env.template");
      if (template != null) {
        envRoot.mkdirs();
        Files.write(dotEnv.toPath(), template.getBytes(StandardCharsets.UTF_8));
        log.info("copilotj: Created default .env at " + dotEnv.getAbsolutePath());
      }
    }

    // Resolve copilotj source directory so Python can import it.
    final File sourceDir = resolveCopilotJSource();
    if (sourceDir == null || !new File(sourceDir, "copilotj/__init__.py").exists()) {
      throw new IOException("CopilotJ source not found. Set -Dcopilotj.sourcePath=<dir>");
    }

    // Read user .env config.
    final Map<String, String> dotEnvVars = readDotEnv(dotEnv);

    try {
      final Environment env = Appose.uv()
          .content(pyprojectContent)
          .scheme("pyproject.toml")
          .python("3.12")
          .base(envRoot)
          .env("PYTHONPATH", sourceDir.getAbsolutePath())
          .env(dotEnvVars)
          .subscribeOutput(msg -> log.info("copilotj env: " + msg))
          .subscribeError(msg -> log.warn("copilotj env: " + msg))
          .build();

      apposeEnv = env;
      return;
    } catch (final BuildException e) {
      throw new IOException("Failed to build CopilotJ Python environment", e);
    }
  }

  /**
   * Resolves the copilotj Python source directory.
   *
   * Checks in order:
   * 1. System property {@code copilotj.sourcePath}
   * 2. Relative to the current working directory (development mode)
   */
  private File resolveCopilotJSource() {
    final String explicit = System.getProperty("copilotj.sourcePath");
    if (explicit != null) {
      return new File(explicit);
    }

    final File cwd = new File(System.getProperty("user.dir"));
    if (new File(cwd, "copilotj/__init__.py").exists()) {
      return cwd;
    }
    final File parent = cwd.getParentFile();
    if (parent != null && new File(parent, "copilotj/__init__.py").exists()) {
      return parent;
    }
    return null;
  }

  private File resolveEnvRoot() {
    // Default to ~/.copilotj/env/
    final String home = System.getProperty("user.home");
    return new File(home, ".copilotj" + File.separator + "env");
  }

  // -- Helpers --

  private static Map<String, String> readDotEnv(final File dotEnvFile) throws IOException {
    final Map<String, String> envVars = new HashMap<>();
    if (!dotEnvFile.exists())
      return envVars;

    for (final String line : Files.readAllLines(dotEnvFile.toPath(), StandardCharsets.UTF_8)) {
      final String trimmed = line.trim();
      if (trimmed.isEmpty() || trimmed.startsWith("#"))
        continue;
      final int eq = trimmed.indexOf('=');
      if (eq > 0) {
        final String key = trimmed.substring(0, eq).trim();
        String value = trimmed.substring(eq + 1).trim();
        // Strip surrounding quotes.
        if (value.length() >= 2
            && ((value.startsWith("\"") && value.endsWith("\""))
                || (value.startsWith("'") && value.endsWith("'")))) {
          value = value.substring(1, value.length() - 1);
        }
        envVars.put(key, value);
      }
    }
    return envVars;
  }

  private static String readResource(final String path) throws IOException {
    final InputStream is = DefaultCopilotJBridgeService.class.getResourceAsStream(path);
    if (is == null)
      return null;
    try (final BufferedReader reader = new BufferedReader(
        new InputStreamReader(is, StandardCharsets.UTF_8))) {
      return reader.lines().collect(Collectors.joining("\n"));
    }
  }

  // -- SciJava event handlers --

  @org.scijava.event.EventHandler
  private void onUIShown(final UIShownEvent e) {
    installActionTool();
  }

  @org.scijava.event.EventHandler
  private void onContextDisposing(final ContextDisposingEvent e) {
    stop();
  }

  void installActionTool() {
    CopilotJBridgeActionToolInstaller.install();
  }
}
