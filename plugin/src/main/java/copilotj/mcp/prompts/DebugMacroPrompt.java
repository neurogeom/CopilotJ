/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj.mcp.prompts;

import java.util.List;
import java.util.Map;

import io.modelcontextprotocol.server.McpSyncServerExchange;
import io.modelcontextprotocol.spec.McpSchema;

public class DebugMacroPrompt {

	public static McpSchema.Prompt definition() {
		return new McpSchema.Prompt("debug_macro",
			"Template for debugging ImageJ macro errors",
			List.of(
				new McpSchema.PromptArgument("error_message", "The error message received", true),
				new McpSchema.PromptArgument("original_script", "The macro script that caused the error", true)
			));
	}

	public McpSchema.GetPromptResult handle(McpSyncServerExchange exchange, McpSchema.GetPromptRequest request) {
		Map<String, Object> args = request.arguments();
		String errorMessage = args.getOrDefault("error_message", "(unknown error)").toString();
		String originalScript = args.getOrDefault("original_script", "(no script provided)").toString();

		String promptText = """
			I got an error running an ImageJ macro. Please help me debug it.

			Error message:
			%s

			Original script:
			```
			%s
			```

			Please:
			1. Analyze the error and identify the likely cause
			2. Check the current Fiji state with take_snapshot
			3. Propose a corrected macro
			4. Test the corrected macro with run_macro
			""".formatted(errorMessage, originalScript);

		return new McpSchema.GetPromptResult(null, List.of(
			new McpSchema.PromptMessage(McpSchema.Role.USER, new McpSchema.TextContent(promptText))));
	}
}
