/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.UnsupportedEncodingException;
import java.net.URI;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Enumeration;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;

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
    final org.apposed.appose.Service service = pythonService;
    pythonService = null;
    if (service != null) {
      try {
        service.close(); // graceful: closes stdin
      } catch (final Exception ignored) {
        // close() may throw if process already dead
      }

      // Wait up to 5 seconds for the process to exit
      final Thread waiter = new Thread(() -> {
        try {
          service.waitFor();
        } catch (final InterruptedException e) {
          Thread.currentThread().interrupt();
        }
      }, "copilotj-shutdown");
      waiter.start();
      try {
        waiter.join(5_000);
      } catch (final InterruptedException e) {
        Thread.currentThread().interrupt();
      }
      if (waiter.isAlive()) {
        log.warn("copilotj: Python process did not exit in 5s, killing");
        service.kill();
        waiter.interrupt();
      }
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
    startManagedServer(null);
  }

  @Override
  public void startManagedServer(final ProgressListener listener) throws IOException, InterruptedException {
    // 1. Ensure environment exists (does not re-sync dependencies).
    ensureEnvironment(listener);

    log.info("copilotj: Environment ready");
    if (listener != null) listener.onMessage("Environment ready. Starting Python server...");

    // 2. Create Appose Service with init script.
    // Using init() ensures all heavy library imports (numpy, scikit-image,
    // stardist, etc.) happen on the main thread before the Appose worker
    // enters its stdin I/O loop. This avoids the stdin/thread deadlock on
    // Windows (see numpy/numpy#24290, apposed/appose#23).
    final Environment env = apposeEnv;
    final String initScript = "from copilotj.appose_worker import start_server; start_server()";
    pythonService = env.python().init(initScript);
    // Forward Python process debug output (includes stderr/logging) to Java log.
    pythonService.debug(msg -> log.info("copilotj python: " + msg));

    log.info("copilotj: Python service created (server started via init script)");
    if (listener != null) listener.onMessage("Python process started. Server initializing...");

    // 3. Verify startup via a dedicated status task. Appose swallows init-script
    // exceptions into a one-line stderr warning (no traceback), so the init()
    // call above never fails even when start_server() crashed inside the worker.
    // We must check explicitly here, on the reliable task/stdout channel.
    // (Upstream appose issue #25 pitfalls #3 and #8.)
    final String statusScript =
        "from copilotj.appose_worker import query_status; task.outputs.update(query_status())";
    final org.apposed.appose.Service.Task statusTask = pythonService.task(statusScript);
    log.info("copilotj: Verifying server startup...");
    if (listener != null) listener.onMessage("Verifying server startup...");
    try {
      statusTask.waitFor();
    } catch (final org.apposed.appose.TaskException e) {
      logServicePythonOutput(pythonService);
      if (listener != null) listener.onMessage("Server start failed: " + e.getMessage());
      throw new IOException("Python server status query failed: " + e.getMessage(), e);
    }
    final String startupError = (String) statusTask.outputs.get("startup_error");
    if (startupError != null) {
      // start_server() failed; the captured traceback is the real cause.
      logServicePythonOutput(pythonService);
      if (listener != null) listener.onMessage("Server start failed:\n" + startupError);
      throw new IOException("Managed Python server failed to start:\n" + startupError);
    }

    // 4. Query the port via a lightweight task (no heavy imports).
    // NOTE: Use task.outputs.update(), not task.export(). The export() method
    // writes to the worker's global exports dict (cross-task persistent
    // variables), while outputs.update() sets the current task's return value
    // that is sent back in the COMPLETION response.
    final String queryScript = "from copilotj.appose_worker import query_port; task.outputs.update(query_port())";
    final org.apposed.appose.Service.Task portTask = pythonService.task(queryScript);

    log.info("copilotj: Querying server port...");
    if (listener != null) listener.onMessage("Querying server port...");

    try {
      portTask.waitFor();
    } catch (final org.apposed.appose.TaskException e) {
      logServicePythonOutput(pythonService);
      if (listener != null) listener.onMessage("Server start failed: " + e.getMessage());
      throw new IOException("Python server port query failed: " + e.getMessage(), e);
    }

    // 5. Extract port from task response.
    final Object portObj = portTask.outputs.get("port");
    if (portObj == null) {
      logServicePythonOutput(pythonService);
      if (listener != null) listener.onMessage("Server start failed: no port returned");
      throw new IOException("Python server did not return a port");
    }
    final int port = ((Number) portObj).intValue();
    managedServerUrl = "http://127.0.0.1:" + port;
    managed = true;

    log.info("copilotj: Python server started on port " + port);
    if (listener != null) listener.onMessage("Python server started on port " + port);

    if (Boolean.TRUE.equals(portTask.outputs.get("port_changed"))) {
      log.warn("copilotj: Port changed from " + portTask.outputs.get("previous_port")
          + " to " + port + " (saved port was unavailable)");
    }

    // 6. Connect WebSocket.
    if (listener != null) listener.onMessage("Connecting WebSocket...");
    this.start(managedServerUrl);

    log.info("copilotj: Managed server fully connected");
    if (listener != null) listener.onMessage("Server ready.");
  }

  /** Dump captured Python stderr and invalid stdout lines for diagnostics. */
  private void logServicePythonOutput(final org.apposed.appose.Service svc) {
    for (final String line : svc.errorLines()) {
      log.warn("copilotj python stderr: " + line);
    }
    for (final String line : svc.invalidLines()) {
      log.warn("copilotj python stdout (invalid): " + line);
    }
  }

  @Override
  public void ensureEnvironment() throws IOException {
    ensureEnvironment(null);
  }

  @Override
  public void ensureEnvironment(final ProgressListener listener) throws IOException {
    if (apposeEnv != null)
      return;
    try {
      apposeEnv = createEnvBuilder(listener).build();
    } catch (final BuildException e) {
      throw unwrapBuildException("Failed to build CopilotJ Python environment", e);
    }
  }

  @Override
  public void uninstallEnvironment() throws IOException {
    stop();
    if (apposeEnv != null) {
      try {
        apposeEnv.delete();
      } catch (final BuildException e) {
        throw new IOException("Failed to delete CopilotJ Python environment", e);
      }
    } else if (isEnvironmentOnDisk()) {
      createEnvBuilder(null).delete();
    }
    apposeEnv = null;
  }

  private IOException unwrapBuildException(final String context, final BuildException e) {
    Throwable cause = e.getCause();
    if (cause != null && cause.getMessage() != null && !cause.getMessage().isEmpty()) {
      return new IOException(context + ": " + cause.getMessage(), e);
    }
    return new IOException(context, e);
  }

  @Override
  public void syncEnvironment(final ProgressListener listener) throws IOException {
    if (apposeEnv == null) {
      ensureEnvironment(listener);
      return;
    }
    try {
      apposeEnv = createEnvBuilder(listener).build();
    } catch (final BuildException e) {
      throw unwrapBuildException("Failed to sync CopilotJ Python environment", e);
    }
  }

  private org.apposed.appose.Builder createEnvBuilder(final ProgressListener listener) throws IOException {
    final File envRoot = resolveEnvRoot();

    // Resolve copilotj source directory so Python can import it.
    File sourceDir = resolveCopilotJSource();
    if (sourceDir == null || !new File(sourceDir, "copilotj/__init__.py").exists()) {
      // Fallback: extract Python sources from JAR
      sourceDir = extractPythonSources(envRoot);
    }

    // Read pyproject.toml from source directory.
    final File pyprojectFile = new File(sourceDir, "pyproject.toml");
    if (!pyprojectFile.isFile()) {
      throw new IOException("pyproject.toml not found at " + pyprojectFile.getAbsolutePath());
    }
    final String pyprojectContent = new String(java.nio.file.Files.readAllBytes(pyprojectFile.toPath()), "UTF-8");

    // Stage uv.lock into envRoot so `uv sync` honors the locked dependency
    // versions. UvBuilder writes pyproject.toml to envRoot and runs `uv sync`
    // with cwd=envRoot, so uv.lock must live at envRoot/uv.lock to be
    // discovered. Workaround for apposed/appose#33 (Appose has no lock API yet).
    final File lockSource = new File(sourceDir, "uv.lock");
    final String lockContent;
    if (lockSource.isFile()) {
      lockContent = new String(java.nio.file.Files.readAllBytes(lockSource.toPath()), "UTF-8");
      invalidateIfLockChanged(envRoot, lockContent);
      final File stagedLock = new File(envRoot, "uv.lock");
      stagedLock.getParentFile().mkdirs();
      Files.write(stagedLock.toPath(), lockContent.getBytes("UTF-8"));
    } else {
      lockContent = null;
      log.warn("copilotj: uv.lock not found at " + lockSource.getAbsolutePath()
          + " — managed env will NOT be reproducible");
    }

    // In dev mode (copilotj.sourcePath set), point COPILOTJ_HOME to the source
    // tree so Python skips bootstrapping — resources are already there.
    // Otherwise (JAR-extracted sources), use envRoot where Java extracted resources.
    final File homeDir = System.getProperty("copilotj.sourcePath") != null
        ? sourceDir
        : envRoot;

    // NOTE: We intentionally do NOT pass `--frozen` here. Appose's `.flags(...)`
    // are emitted BEFORE the subcommand (uv --frozen sync ...), but uv only
    // accepts `--frozen` AFTER `sync` (uv sync --frozen), so it would error.
    // Instead we rely on staging uv.lock into envRoot above: plain `uv sync`
    // honors a lock that is consistent with pyproject.toml and installs the
    // exact pinned versions. True strict `--frozen` needs upstream
    // apposed/appose#33 (post-subcommand arg injection). See plan caveats.
    final org.apposed.appose.Builder builder = Appose.uv()
        .content(pyprojectContent)
        .scheme("pyproject.toml")
        .python("3.12")
        .base(envRoot)
        .env("PYTHONPATH", sourceDir.getAbsolutePath())
        .env("COPILOTJ_HOME", homeDir.getAbsolutePath())
        .env("COPILOTJ_MANAGED", "1")
        .subscribeOutput(msg -> log.info("copilotj env: " + msg))
        .subscribeError(msg -> log.warn("copilotj env: " + msg));

    if (listener != null) {
      final org.apposed.appose.Builder<?> typed = builder;
      typed.subscribeOutput(msg -> listener.onMessage(msg))
          .subscribeError(msg -> listener.onMessage(msg));
    }

    return builder;
  }

  /**
   * Resolves the copilotj Python source directory.
   *
   * Checks in order:
   * 1. System property {@code copilotj.sourcePath}
   * 2. Relative to the current working directory (development mode)
   * 3. Previously extracted resources (production cache, validated against JAR
   * version)
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

    // Previously extracted resources from JAR (production cache)
    final File envRoot = resolveEnvRoot();
    final File extracted = new File(envRoot, "python_sources");
    if (new File(extracted, "copilotj/__init__.py").exists()) {
      // Validate version — re-extract if plugin was upgraded
      final String fingerprint = jarFingerprint();
      final String cachedSourcesVersion = readVersionMarker(new File(extracted, ".version"));
      final String cachedAssetsVersion = readVersionMarker(new File(envRoot, "assets/.version"));
      final String cachedKbVersion = readVersionMarker(new File(envRoot, "knowledge_bank/.version"));
      if (fingerprint.equals(cachedSourcesVersion)
          && fingerprint.equals(cachedAssetsVersion)
          && fingerprint.equals(cachedKbVersion)) {
        return extracted;
      }
      log.info("copilotj: Cached resources are stale, will re-extract");
    }

    return null;
  }

  /**
   * Extracts Python sources, assets, and knowledge bank bundled in the JAR under
   * {@code META-INF/copilotj/} to {@code $COPILOTJ_HOME/}.
   *
   * The extraction target layout:
   *
   * <pre>
   * $COPILOTJ_HOME/
   * ├── python_sources/
   * │   ├── copilotj/
   * │   └── pyproject.toml
   * ├── assets/
   * │   ├── knowledge_base/
   * │   ├── models/
   * │   └── ...
   * └── knowledge_bank/
   *     ├── task/
   *     ├── macro/
   *     └── research/
   * </pre>
   */
  private File extractPythonSources(final File envRoot) throws IOException {
    final File target = new File(envRoot, "python_sources");
    log.info("copilotj: Extracting Python resources to " + target.getAbsolutePath());

    // Locate the JAR containing this class
    final URI jarUri;
    try {
      jarUri = getClass().getProtectionDomain().getCodeSource().getLocation().toURI();
    } catch (final URISyntaxException e) {
      throw new IOException("Cannot locate plugin JAR", e);
    }

    final File jarFile = new File(jarUri);
    if (!jarFile.isFile()) {
      throw new IOException("CopilotJ source not found and plugin is not running from a JAR. "
          + "Set -Dcopilotj.sourcePath=<dir>");
    }

    final String prefix = "META-INF/copilotj/";
    try (final JarFile jar = new JarFile(jarFile)) {
      final Enumeration<JarEntry> entries = jar.entries();
      while (entries.hasMoreElements()) {
        final JarEntry entry = entries.nextElement();
        if (!entry.getName().startsWith(prefix) || entry.isDirectory()) {
          continue;
        }
        final String relativePath = entry.getName().substring(prefix.length());
        // Route assets/ and knowledge_bank/ directly to $COPILOTJ_HOME/;
        // Python sources go to python_sources/
        final File outFile;
        if (relativePath.startsWith("assets/") || relativePath.startsWith("knowledge_bank/")) {
          outFile = new File(envRoot, relativePath);
        } else {
          outFile = new File(target, relativePath);
        }
        outFile.getParentFile().mkdirs();
        try (final InputStream is = jar.getInputStream(entry)) {
          Files.copy(is, outFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
        }
      }
    }

    if (!new File(target, "copilotj/__init__.py").exists()) {
      throw new IOException("Python sources not found in plugin JAR");
    }

    // Write version markers so we can detect stale caches on upgrade
    final String fingerprint = jarFingerprint();
    writeVersionMarker(new File(target, ".version"), fingerprint);
    writeVersionMarker(new File(envRoot, "assets/.version"), fingerprint);
    writeVersionMarker(new File(envRoot, "knowledge_bank/.version"), fingerprint);

    log.info("copilotj: Python resources extracted successfully");
    return target;
  }

  /** Returns a fingerprint string that changes when the plugin JAR is updated. */
  private String jarFingerprint() {
    try {
      final File jarFile = new File(
          getClass().getProtectionDomain().getCodeSource().getLocation().toURI());
      if (jarFile.isFile()) {
        return jarFile.getAbsolutePath() + ":" + jarFile.lastModified();
      }
    } catch (final URISyntaxException ignored) {
    }
    return "dev";
  }

  private static String readFirstLine(final File file) {
    try {
      final String content = new String(Files.readAllBytes(file.toPath()), "UTF-8").trim();
      final int newline = content.indexOf('\n');
      return newline >= 0 ? content.substring(0, newline) : content;
    } catch (final IOException e) {
      return "";
    }
  }

  /** Reads the version marker file, returning empty string if missing or unreadable. */
  private static String readVersionMarker(final File file) {
    return file.isFile() ? readFirstLine(file) : "";
  }

  /** Writes a version marker file, logging a warning on failure. */
  private void writeVersionMarker(final File file, final String fingerprint) {
    try {
      file.getParentFile().mkdirs();
      Files.write(file.toPath(), fingerprint.getBytes("UTF-8"));
    } catch (final IOException e) {
      log.warn("copilotj: Could not write version marker to " + file.getAbsolutePath(), e);
    }
  }

  /**
   * Forces a managed-env rebuild when the bundled uv.lock content changes, even
   * if pyproject.toml is unchanged. Appose's {@code appose.json} state has no
   * lock-file hash (apposed/appose#33), so without this a uv.lock-only change
   * would be considered up-to-date and never applied to the existing venv.
   */
  private void invalidateIfLockChanged(final File envRoot, final String lockContent) {
    final String fingerprint = sha256(lockContent);
    final File marker = new File(envRoot, ".uv.lock.sha256");
    if (!fingerprint.equals(readVersionMarker(marker))) {
      final File apposeJson = new File(envRoot, "appose.json");
      if (apposeJson.isFile() && !apposeJson.delete()) {
        log.warn("copilotj: could not delete stale " + apposeJson.getAbsolutePath());
      }
      writeVersionMarker(marker, fingerprint);
    }
  }

  /** Hex-encoded SHA-256 of the given UTF-8 string. */
  private static String sha256(final String s) {
    try {
      final MessageDigest md = MessageDigest.getInstance("SHA-256");
      final byte[] digest = md.digest(s.getBytes("UTF-8"));
      final StringBuilder hex = new StringBuilder(digest.length * 2);
      for (final byte b : digest) {
        hex.append(String.format("%02x", b & 0xff));
      }
      return hex.toString();
    } catch (final NoSuchAlgorithmException | UnsupportedEncodingException e) {
      // SHA-256 and UTF-8 are mandated by the Java platform spec — unreachable.
      throw new RuntimeException(e);
    }
  }

  private File resolveEnvRoot() {
    final String os = System.getProperty("os.name").toLowerCase();
    if (os.contains("win")) {
      final String local = System.getenv("LOCALAPPDATA");
      if (local != null && !local.isEmpty())
        return new File(local, "copilotj");
      return new File(System.getProperty("user.home"), "AppData\\Local\\copilotj");
    }
    String xdg = System.getenv("XDG_STATE_HOME");
    if (xdg == null || xdg.isEmpty()) {
      xdg = System.getProperty("user.home") + "/.local/state";
    }
    return new File(xdg, "copilotj");
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
