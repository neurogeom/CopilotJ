/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj.mcp.tools;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import copilotj.EventHandler;
import copilotj.mcp.McpModule;
import copilotj.util.IjImageHelper;

import com.fasterxml.jackson.databind.JsonNode;

import io.modelcontextprotocol.server.McpSyncServerExchange;
import io.modelcontextprotocol.spec.McpSchema;

public class CaptureImageTool {

	private final EventHandler handler;

	public CaptureImageTool(EventHandler handler) {
		this.handler = handler;
	}

	public static McpSchema.Tool definition() {
		return McpSchema.Tool.builder("capture_image",
				Map.of(
					"type", "object",
					"properties", Map.of(
						"title", Map.of("type", "string", "description", "Window title of the image to capture (optional)"))))
			.description("Capture the current active Fiji image with metadata. "
				+ "Returns image content along with dimensions, bit depth, and histogram. "
				+ "Optionally specify a window title to capture a specific image.")
			.build();
	}

	public McpSchema.CallToolResult handle(McpSyncServerExchange exchange, McpSchema.CallToolRequest request) {
		Map<String, Object> args = request.arguments();
		var data = new java.util.LinkedHashMap<String, Object>();
		data.put("title", args.get("title"));

		try {
			String result = McpModule.callEvent(handler, "capture_image", data);
			JsonNode root = McpModule.objectMapper.readTree(result);

			List<McpSchema.Content> content = new ArrayList<>();

			Map<String, Object> metadata = new java.util.LinkedHashMap<>();
			if (root.has("info") && !root.get("info").isNull()) {
				metadata.put("info", McpModule.objectMapper.convertValue(root.get("info"), Map.class));
			}
			if (root.has("histogram") && !root.get("histogram").isNull()) {
				metadata.put("histogram", McpModule.objectMapper.convertValue(root.get("histogram"), Map.class));
			}
			if (!metadata.isEmpty()) {
				content.add(McpSchema.TextContent.builder(
						McpModule.objectMapper.writerWithDefaultPrettyPrinter().writeValueAsString(metadata)).build());
			}

			if (root.has("image") && !root.get("image").isNull()) {
				// The image is stored as a data URI ("data:image/png;base64,…");
				// keep only the base64 payload for ImageContent.
				String base64Image = IjImageHelper.extractBase64(root.get("image").asText());
				content.add(McpSchema.ImageContent.builder(base64Image, "image/png").build());
			}

			if (content.isEmpty()) {
				return error("No image data or metadata returned.");
			}

			return McpSchema.CallToolResult.builder().content(content).build();
		} catch (Exception e) {
			return error("Failed to capture image: " + e.getMessage());
		}
	}

	private static McpSchema.CallToolResult error(String msg) {
		return McpSchema.CallToolResult.builder()
			.content(List.of(McpSchema.TextContent.builder(msg).build()))
			.isError(true)
			.build();
	}
}
