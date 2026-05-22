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

public class RunScriptTool {

	private final EventHandler handler;

	public RunScriptTool(EventHandler handler) {
		this.handler = handler;
	}

	public static McpSchema.Tool definition() {
		return McpSchema.Tool.builder()
			.name("run_script")
			.description("Execute a script in Fiji. Supported languages: macro, JavaScript, Python, etc. "
				+ "Returns script output or error message.")
			.inputSchema(new McpSchema.JsonSchema("object",
				Map.of(
					"language", Map.of("type", "string", "description", "Script language (macro, JavaScript, Python)"),
					"script", Map.of("type", "string", "description", "Script code to execute"),
					"timeout", Map.of("type", "integer", "description", "Timeout in seconds (default: auto-detected)")
				),
				List.of("language", "script"), true, null, null))
			.build();
	}

	public McpSchema.CallToolResult handle(McpSyncServerExchange exchange, McpSchema.CallToolRequest request) {
		Map<String, Object> args = request.arguments();
		String language = (String) args.get("language");
		String script = (String) args.get("script");
		int timeout = args.containsKey("timeout")
			? ((Number) args.get("timeout")).intValue()
			: detectTimeout(language, script);

		var data = Map.of(
			"language", language,
			"script", script,
			"timeout", timeout
		);

		try {
			String result = McpModule.callEvent(handler, "run_script", data);
			return McpSchema.CallToolResult.builder()
				.content(List.of(new McpSchema.TextContent(result)))
				.build();
		} catch (Exception e) {
			return McpSchema.CallToolResult.builder()
				.content(List.of(new McpSchema.TextContent("Script error: " + e.getMessage())))
				.isError(true)
				.build();
		}
	}

	/**
	 * Detect timeout based on script language and content.
	 * Macros use the RunMacroTool heuristic; other languages default to 60s.
	 */
	static int detectTimeout(String language, String script) {
		if ("macro".equalsIgnoreCase(language) || "ijm".equalsIgnoreCase(language)) {
			return RunMacroTool.detectTimeout(script);
		}
		// Non-macro scripts (JavaScript, Python, etc.) get a more generous default
		return 60;
	}
}
