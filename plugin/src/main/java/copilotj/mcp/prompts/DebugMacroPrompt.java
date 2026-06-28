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
		return McpSchema.Prompt.builder("debug_macro")
			.description("Template for debugging ImageJ macro errors")
			.arguments(List.of(
				McpSchema.PromptArgument.builder("error_message")
					.description("The error message received")
					.required(true)
					.build(),
				McpSchema.PromptArgument.builder("original_script")
					.description("The macro script that caused the error")
					.required(true)
					.build()))
			.build();
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

		return McpSchema.GetPromptResult.builder(List.of(
				new McpSchema.PromptMessage(McpSchema.Role.USER, McpSchema.TextContent.builder(promptText).build())))
			.build();
	}
}
