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
		return McpSchema.Tool.builder("take_snapshot", Map.of("type", "object"))
			.description("Get a structured snapshot of the current Fiji UI state. "
				+ "Returns open windows, available actions, current image name, and screen dimensions. "
				+ "Use this to understand what's open before running commands.")
			.build();
	}

	public McpSchema.CallToolResult handle(McpSyncServerExchange exchange, McpSchema.CallToolRequest request) {
		try {
			String result = McpModule.callEvent(handler, "take_snapshot", null);
			return McpSchema.CallToolResult.builder()
				.content(List.of(McpSchema.TextContent.builder(result).build()))
				.build();
		} catch (Exception e) {
			return McpSchema.CallToolResult.builder()
				.content(List.of(McpSchema.TextContent.builder("Failed to take snapshot: " + e.getMessage()).build()))
				.isError(true)
				.build();
		}
	}
}
