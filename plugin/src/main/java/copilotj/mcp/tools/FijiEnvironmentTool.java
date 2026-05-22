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

public class FijiEnvironmentTool {

	private final EventHandler handler;

	public FijiEnvironmentTool(EventHandler handler) {
		this.handler = handler;
	}

	public static McpSchema.Tool definition() {
		return McpSchema.Tool.builder()
			.name("fiji_environment")
			.description("Get Fiji/ImageJ2 environment information. "
				+ "Returns ImageJ home, Java version, installed plugins, and other system details.")
			.inputSchema(new McpSchema.JsonSchema("object", Map.of(), null, true, null, null))
			.build();
	}

	public McpSchema.CallToolResult handle(McpSyncServerExchange exchange, McpSchema.CallToolRequest request) {
		try {
			String result = McpModule.callEvent(handler, "summarise_environment", null);
			return McpSchema.CallToolResult.builder()
				.content(List.of(new McpSchema.TextContent(result)))
				.build();
		} catch (Exception e) {
			return McpSchema.CallToolResult.builder()
				.content(List.of(new McpSchema.TextContent("Failed to get environment: " + e.getMessage())))
				.isError(true)
				.build();
		}
	}
}
