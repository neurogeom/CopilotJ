/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj.mcp.tools;

import java.util.List;
import java.util.Map;

import copilotj.EventHandler;
import copilotj.mcp.McpModule;

import io.modelcontextprotocol.server.McpSyncServerExchange;
import io.modelcontextprotocol.spec.McpSchema;

public class TakeSnapshotTool {

	private final EventHandler handler;

	public TakeSnapshotTool(EventHandler handler) {
		this.handler = handler;
	}

	public static McpSchema.Tool definition() {
		return McpSchema.Tool.builder()
			.name("take_snapshot")
			.description("Get a structured snapshot of the current Fiji UI state. "
				+ "Returns open windows and their component trees; each actionable component carries "
				+ "a ref handle and a per-component actions list. A ref is stable for the lifetime of "
				+ "the same live component instance; if Fiji rebuilds a control it gets a new ref. Pass "
				+ "the ref plus an action's short id to call_action; each action's parameters[] documents "
				+ "what to pass. Also returns the current image name and screen dimensions.")
			.inputSchema(new McpSchema.JsonSchema("object", Map.of(), null, true, null, null))
			.build();
	}

	public McpSchema.CallToolResult handle(McpSyncServerExchange exchange, McpSchema.CallToolRequest request) {
		try {
			String result = McpModule.callEvent(handler, "take_snapshot", null);
			return McpSchema.CallToolResult.builder()
				.content(List.of(new McpSchema.TextContent(result)))
				.build();
		} catch (Exception e) {
			return McpSchema.CallToolResult.builder()
				.content(List.of(new McpSchema.TextContent("Failed to take snapshot: " + e.getMessage())))
				.isError(true)
				.build();
		}
	}
}
