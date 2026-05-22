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
import copilotj.util.IjImageHelper;

import com.fasterxml.jackson.databind.JsonNode;

import io.modelcontextprotocol.server.McpSyncServerExchange;
import io.modelcontextprotocol.spec.McpSchema;

public class CaptureScreenTool {

	private final EventHandler handler;

	public CaptureScreenTool(EventHandler handler) {
		this.handler = handler;
	}

	public static McpSchema.Tool definition() {
		return McpSchema.Tool.builder()
			.name("capture_fiji_screen")
			.description("Capture the current Fiji screen as a PNG image. "
				+ "Returns a screenshot showing all open Fiji windows and their state.")
			.inputSchema(new McpSchema.JsonSchema("object", Map.of(), null, true, null, null))
			.build();
	}

	public McpSchema.CallToolResult handle(McpSyncServerExchange exchange, McpSchema.CallToolRequest request) {
		try {
			String result = McpModule.callEvent(handler, "capture_screen", null);
			JsonNode data = McpModule.objectMapper.readTree(result);

			JsonNode screenshots = data.get("screenshots");
			if (screenshots == null || screenshots.isEmpty()) {
				return error("No screenshots captured. Fiji may not have any windows open.");
			}

			// The screenshot is stored as a data URI ("data:image/png;base64,…");
			// ImageContent already wraps a base64 payload, so keep just that part
			// rather than decode→re-encode round-tripping the whole data URI.
			String base64Image = IjImageHelper.extractBase64(screenshots.get(0).get("image").asText());

			return McpSchema.CallToolResult.builder()
				.content(List.of(new McpSchema.ImageContent(null, base64Image, "image/png")))
				.build();
		} catch (Exception e) {
			return error("Failed to capture screen: " + e.getMessage());
		}
	}

	private static McpSchema.CallToolResult error(String msg) {
		return McpSchema.CallToolResult.builder()
			.content(List.of(new McpSchema.TextContent(msg)))
			.isError(true)
			.build();
	}
}
