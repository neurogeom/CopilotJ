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

public class CallActionTool {

	private final EventHandler handler;

	public CallActionTool(EventHandler handler) {
		this.handler = handler;
	}

	public static McpSchema.Tool definition() {
		return McpSchema.Tool.builder("call_action",
				Map.of(
					"type", "object",
					"properties", Map.of(
						"snapshot_id", Map.of("type", "integer", "description", "Snapshot ID from take_snapshot"),
						"action_id", Map.of("type", "integer", "description", "Action ID within the snapshot"),
						"parameters", Map.of("type", "array", "description", "Action parameters", "items", Map.of())),
					"required", List.of("snapshot_id", "action_id")))
			.description("Execute a UI action from a previous snapshot. "
				+ "First call take_snapshot() to get available actions and their IDs, "
				+ "then call this with the snapshot_id and action_id.")
			.build();
	}

	public McpSchema.CallToolResult handle(McpSyncServerExchange exchange, McpSchema.CallToolRequest request) {
		Map<String, Object> args = request.arguments();
		var data = new java.util.LinkedHashMap<String, Object>();
		data.put("snapshot_id", args.get("snapshot_id"));
		data.put("action_id", args.get("action_id"));
		if (args.containsKey("parameters")) {
			data.put("parameters", args.get("parameters"));
		}

		try {
			String result = McpModule.callEvent(handler, "run_action", data);
			return McpSchema.CallToolResult.builder()
				.content(List.of(McpSchema.TextContent.builder(result).build()))
				.build();
		} catch (Exception e) {
			return McpSchema.CallToolResult.builder()
				.content(List.of(McpSchema.TextContent.builder("Failed to execute action: " + e.getMessage()).build()))
				.isError(true)
				.build();
		}
	}
}
