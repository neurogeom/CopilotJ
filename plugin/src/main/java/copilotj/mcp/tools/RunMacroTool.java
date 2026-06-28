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

public class RunMacroTool {

	private final EventHandler handler;

	public RunMacroTool(EventHandler handler) {
		this.handler = handler;
	}

	public static McpSchema.Tool definition() {
		return McpSchema.Tool.builder("run_macro",
				Map.of(
					"type", "object",
					"properties", Map.of(
						"script", Map.of("type", "string", "description", "ImageJ macro script to execute"),
						"timeout", Map.of("type", "integer", "description", "Timeout in seconds (default: auto-detected)")),
					"required", List.of("script")))
			.description("Execute an ImageJ macro script in the running Fiji instance. "
				+ "Timeout is auto-detected: 15s for normal scripts, 180s for batch/loop scripts. "
				+ "Set timeout explicitly to override auto-detection.")
			.build();
	}

	public McpSchema.CallToolResult handle(McpSyncServerExchange exchange, McpSchema.CallToolRequest request) {
		Map<String, Object> args = request.arguments();
		String script = (String) args.get("script");
		int timeout = args.containsKey("timeout")
			? ((Number) args.get("timeout")).intValue()
			: detectTimeout(script);

		String scriptWithMarker = script + "\nprint(\"Macro executed.\");";

		var data = Map.of(
			"language", "macro",
			"script", scriptWithMarker,
			"timeout", timeout
		);

		try {
			String result = McpModule.callEvent(handler, "run_script", data);
			return McpSchema.CallToolResult.builder()
				.content(List.of(McpSchema.TextContent.builder("Macro executed successfully.\n" + result).build()))
				.build();
		} catch (Exception e) {
			return McpSchema.CallToolResult.builder()
				.content(List.of(McpSchema.TextContent.builder("Error: " + e.getMessage()).build()))
				.isError(true)
				.build();
		}
	}

	static int detectTimeout(String script) {
		String lower = script.toLowerCase();
		if (lower.contains("batch") || lower.contains("for(") || lower.contains("for (")
			|| lower.contains("while") || lower.contains("getfilelist")) {
			return 180;
		}
		return 15;
	}
}
