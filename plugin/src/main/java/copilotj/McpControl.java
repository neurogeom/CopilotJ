/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj;

/**
 * Lets core code (the {@code CopilotJBridgeDialog}) query and stop the MCP
 * server that lives inside the isolated {@code mcp-bundle}, without reflection.
 *
 * <p>This interface MUST stay in package {@code copilotj} (not {@code copilotj.mcp}):
 * <ul>
 *   <li>It is compiled by the default Java&nbsp;8 execution into {@code target/classes}
 *       and is therefore never assembled into {@code lib/mcp-bundle.jar} (whose
 *       {@code <fileSet>} only pulls {@code target/mcp-classes}).</li>
 *   <li>At runtime {@code McpPanel} (loaded by the child-first bundle ClassLoader)
 *       resolves this type via the parent ClassLoader, so {@code McpPanel} and the
 *       dialog share one class identity and {@code instanceof} / casts work.</li>
 * </ul>
 * Moving it into {@code copilotj.mcp} would package a duplicate copy into the
 * bundle and break that shared identity.
 */
public interface McpControl {

  /** True if the MCP HTTP server is currently running. */
  boolean isMcpRunning();

  /** Stop the MCP server if it is running (no-op otherwise). */
  void stopMcp();
}
