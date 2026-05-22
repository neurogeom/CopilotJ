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

public class ListOperationsTool {

	private final EventHandler handler;
	private volatile String lastCallTimestamp;

	public ListOperationsTool(EventHandler handler) {
		this.handler = handler;
	}

	public static McpSchema.Tool definition() {
		return McpSchema.Tool.builder()
			.name("list_operations")
			.description("Get recent Fiji operation history. "
				+ "Returns list of operations performed since the given datetime (ISO 8601 format). "
				+ "If no datetime is provided, returns operations since the last call.")
			.inputSchema(new McpSchema.JsonSchema("object",
				Map.of("since", Map.of("type", "string", "description", "ISO 8601 datetime (e.g., '2026-04-15T10:00:00')")),
				null, true, null, null))
			.build();
	}

	public McpSchema.CallToolResult handle(McpSyncServerExchange exchange, McpSchema.CallToolRequest request) {
		Map<String, Object> args = request.arguments();
		String since = args.containsKey("since") ? (String) args.get("since") : lastCallTimestamp;

		if (since == null) {
			return McpSchema.CallToolResult.builder()
				.content(List.of(new McpSchema.TextContent(
					"No timestamp provided and no previous call recorded. "
					+ "Please provide a 'since' parameter (ISO 8601 datetime).")))
				.isError(true)
				.build();
		}

		var data = Map.of("since", (Object) since);

		try {
			String result = McpModule.callEvent(handler, "get_operation_history", data);
			// Record timestamp after successful call
			lastCallTimestamp = java.time.LocalDateTime.now().toString();
			return McpSchema.CallToolResult.builder()
				.content(List.of(new McpSchema.TextContent(result)))
				.build();
		} catch (Exception e) {
			return McpSchema.CallToolResult.builder()
				.content(List.of(new McpSchema.TextContent("Failed to get operations: " + e.getMessage())))
				.isError(true)
				.build();
		}
	}
}
