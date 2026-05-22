/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj.mcp.resources;

import java.util.List;

import copilotj.EventHandler;
import copilotj.mcp.McpModule;

import io.modelcontextprotocol.server.McpSyncServerExchange;
import io.modelcontextprotocol.spec.McpSchema;

public class EnvironmentResource {

	private final EventHandler handler;

	public EnvironmentResource(EventHandler handler) {
		this.handler = handler;
	}

	public static McpSchema.Resource definition() {
		return McpSchema.Resource.builder()
			.uri("fiji://environment")
			.name("fiji-environment")
			.description("Fiji/ImageJ2 environment information")
			.mimeType("application/json")
			.build();
	}

	public McpSchema.ReadResourceResult handle(McpSyncServerExchange exchange, McpSchema.ReadResourceRequest request) {
		try {
			String result = McpModule.callEvent(handler, "summarise_environment", null);
			return new McpSchema.ReadResourceResult(
				List.of(new McpSchema.TextResourceContents(request.uri(), "application/json", result)));
		} catch (Exception e) {
			return new McpSchema.ReadResourceResult(
				List.of(new McpSchema.TextResourceContents(request.uri(), "application/json",
					"{\"error\": \"Fiji not connected\"}")));
		}
	}
}
