/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj;

import java.util.ArrayDeque;
import java.util.Deque;

/**
 * Bounded, append-only text log. Thread-safe so it can be read from the UI-build
 * thread while the EDT appends.
 *
 * <p>Used as the single source of truth behind a {@code JTextArea} log (e.g. the
 * Managed Server "Progress Log" and the MCP log). Held in a {@code static} field,
 * it survives the dialog's dispose+recreate lifecycle — so closing and reopening
 * the config window replays the retained tail via {@link #snapshot()}. The newest
 * ~{@value #MAX_CHARS} chars are kept; older append-chunks are evicted so the
 * backing string never grows without limit.
 *
 * <p>Lives in the core package on purpose: it is compiled to Java 8 bytecode and
 * excluded from the isolated MCP bundle, so MCP classes share one class identity
 * with the core via the parent classloader (like {@link EventHandler}).
 */
public final class BoundedLog {

  /** Soft cap on retained text. 64 KiB is ample for a progress log and light for a JTextArea. */
  static final int MAX_CHARS = 65_536;

  private final Deque<String> chunks = new ArrayDeque<>();
  private long total = 0L;

  /**
   * Appends {@code s}. When the total length exceeds {@link #MAX_CHARS}, drops the
   * oldest append-chunks until back under the cap (always keeping at least the most
   * recent chunk).
   */
  public synchronized void append(final String s) {
    if (s == null || s.isEmpty()) return;
    chunks.addLast(s);
    total += s.length();
    while (total > MAX_CHARS && chunks.size() > 1) {
      total -= chunks.removeFirst().length();
    }
  }

  /** Empties the log. */
  public synchronized void clear() {
    chunks.clear();
    total = 0L;
  }

  /** Returns the full retained text (joined in append order). */
  public synchronized String snapshot() {
    final StringBuilder sb = new StringBuilder((int) Math.min(total + 16, 1 << 20));
    for (final String c : chunks) {
      sb.append(c);
    }
    return sb.toString();
  }
}
