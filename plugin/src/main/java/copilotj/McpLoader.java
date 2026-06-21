/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URISyntaxException;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.function.BooleanSupplier;

import javax.swing.JPanel;

import org.scijava.log.LogService;

/**
 * Loads the MCP server bundle reflectively from an isolated child-first
 * ClassLoader, so the Java 8 core can run on legacy Fiji while MCP is enabled on
 * Java 17+ Fiji.
 *
 * <p>The bundle ships as a non-classpath resource {@code lib/mcp-bundle.jar}
 * inside the core JAR. At runtime it is released to a cache file under
 * {@code $COPILOTJ_HOME} and loaded via a child-first {@link URLClassLoader}
 * whose parent is the core classloader. MCP classes (and their Jetty 12 /
 * Jackson 2.18 dependencies) stay isolated from Fiji's shared classpath; only
 * shared types ({@link EventHandler}, JDK, SciJava) are resolved by the parent.
 * On a JVM older than 17 the bundle's classes fail bytecode verification and we
 * degrade gracefully.
 *
 * <p>The underlying ClassLoader is cached statically so MCP's own static state
 * (e.g. {@code McpModule}'s server handle) stays consistent across UI reopens.
 */
public final class McpLoader {

  private static final String BUNDLE_RESOURCE = "/lib/mcp-bundle.jar";
  private static final String PANEL_CLASS = "copilotj.mcp.McpPanel";
  private static final String MODULE_CLASS = "copilotj.mcp.McpModule";

  private static volatile ClassLoader bundleLoader;
  private static volatile boolean unavailable;

  private final LogService log;

  public McpLoader(final LogService log) {
    this.log = log;
  }

  /**
   * True if the MCP bundle could be loaded on this JVM (Java 17+ and bundle
   * present).
   */
  public boolean isAvailable() {
    ensureLoaded();
    return bundleLoader != null;
  }

  /**
   * Reflectively creates the MCP control panel, or {@code null} if MCP is
   * unavailable on this JVM.
   *
   * @param handler forwarded into the bundle as the shared event channel
   * @param requireExclusiveControl invoked by the panel right before starting
   *        the MCP server; returns {@code true} to proceed. The core dialog
   *        supplies this so it can detect/stop a running managed or external
   *        server (and prompt the user) without the bundle depending on
   *        {@link CopilotJBridgeService}.
   * @param statusUpdater invoked by the panel on MCP status changes (text +
   *        color); the core dialog routes it to the shared status bar, merging
   *        MCP state into the single endpoint-status display.
   */
  public JPanel createPanel(final EventHandler handler, final BooleanSupplier requireExclusiveControl,
      final java.util.function.BiConsumer<String, java.awt.Color> statusUpdater) {
    if (!isAvailable()) return null;
    try {
      final Class<?> cls = Class.forName(PANEL_CLASS, true, bundleLoader);
      final java.lang.reflect.Constructor<?> ctor =
          cls.getConstructor(EventHandler.class, LogService.class, BooleanSupplier.class,
              java.util.function.BiConsumer.class);
      return (JPanel) ctor.newInstance(handler, log, requireExclusiveControl, statusUpdater);
    } catch (final Throwable t) {
      log.warn("copilotj: Failed to create MCP panel: " + t.getMessage(), t);
      return null;
    }
  }

  private void ensureLoaded() {
    if (bundleLoader != null || unavailable) return;
    synchronized (McpLoader.class) {
      if (bundleLoader != null || unavailable) return;
      try {
        final Path bundle = releaseBundle();
        final ClassLoader loader = new ChildFirstLoader(
            new URL[]{bundle.toUri().toURL()},
            McpLoader.class.getClassLoader());
        // Probe by loading the entry module class — this triggers bytecode
        // verification, which throws UnsupportedClassVersionError on Java <17.
        Class.forName(MODULE_CLASS, true, loader);
        bundleLoader = loader;
      } catch (final Throwable t) {
        unavailable = true;
        log.info("copilotj: MCP unavailable on Java " + System.getProperty("java.version")
            + " (" + t.getClass().getSimpleName() + ": " + t.getMessage() + ")");
      }
    }
  }

  /**
   * Releases the embedded bundle to a cache file under {@code $COPILOTJ_HOME},
   * skipping the copy when the cached file already matches the current plugin
   * JAR fingerprint. Single opaque file copy — never unpacks jar entries, so
   * there is no zip-slip attack surface.
   */
  private Path releaseBundle() throws IOException {
    final File dir = new File(envRoot(), "mcp");
    if (!dir.isDirectory() && !dir.mkdirs()) {
      throw new IOException("Could not create MCP cache dir: " + dir);
    }
    final Path target = new File(dir, "mcp-bundle.jar").toPath();
    final Path marker = new File(dir, ".version").toPath();
    final String fingerprint = jarFingerprint();
    final String cached = readText(marker);
    if (target.toFile().isFile() && fingerprint.equals(cached)) {
      return target; // cache hit
    }
    try (InputStream in = McpLoader.class.getResourceAsStream(BUNDLE_RESOURCE)) {
      if (in == null) {
        throw new IOException("MCP bundle resource not found: " + BUNDLE_RESOURCE);
      }
      Files.copy(in, target, StandardCopyOption.REPLACE_EXISTING);
    }
    Files.write(marker, fingerprint.getBytes("UTF-8"));
    return target;
  }

  /** Resolves {@code $COPILOTJ_HOME} (mirrors DefaultCopilotJBridgeService#resolveEnvRoot). */
  private static File envRoot() {
    final String os = System.getProperty("os.name").toLowerCase();
    if (os.contains("win")) {
      final String local = System.getenv("LOCALAPPDATA");
      if (local != null && !local.isEmpty()) return new File(local, "copilotj");
      return new File(System.getProperty("user.home"), "AppData\\Local\\copilotj");
    }
    String xdg = System.getenv("XDG_STATE_HOME");
    if (xdg == null || xdg.isEmpty()) {
      xdg = System.getProperty("user.home") + "/.local/state";
    }
    return new File(xdg, "copilotj");
  }

  /** A fingerprint that changes when the plugin JAR is updated. */
  private static String jarFingerprint() {
    // Fingerprint the embedded bundle itself, not the loading-class location.
    // In dev the core loads from an exploded directory (no jar mtime), so keying
    // off the bundle resource's mtime+size invalidates the cache on rebuild in
    // BOTH dev (exploded classes) and production (jar).
    try {
      final URL url = McpLoader.class.getResource(BUNDLE_RESOURCE);
      if (url != null) {
        final java.net.URLConnection conn = url.openConnection();
        return conn.getLastModified() + ":" + conn.getContentLengthLong();
      }
    } catch (final IOException ignored) {
    }
    return "dev";
  }

  private static String readText(final Path p) {
    try {
      return new String(Files.readAllBytes(p), "UTF-8").trim();
    } catch (final IOException e) {
      return "";
    }
  }

  /**
   * Child-first (parent-last) URLClassLoader: resolve classes from the bundle
   * first, delegating to the parent only when the bundle does not define them.
   * This isolates MCP's Jetty 12 / Jackson 2.18 from Fiji's versions, while
   * letting shared types (EventHandler, JDK, SciJava) be supplied by the parent
   * so MCP and the core share one {@link EventHandler} class identity.
   */
  private static final class ChildFirstLoader extends URLClassLoader {

    ChildFirstLoader(final URL[] urls, final ClassLoader parent) {
      super(urls, parent);
    }

    @Override
    protected Class<?> loadClass(final String name, final boolean resolve)
        throws ClassNotFoundException {
      synchronized (getClassLoadingLock(name)) {
        Class<?> c = findLoadedClass(name);
        if (c == null) {
          try {
            c = findClass(name);
          } catch (final ClassNotFoundException notInBundle) {
            c = getParent().loadClass(name);
          }
        }
        if (resolve) resolveClass(c);
        return c;
      }
    }
  }
}
