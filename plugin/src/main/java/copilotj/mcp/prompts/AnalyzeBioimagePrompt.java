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

public class AnalyzeBioimagePrompt {

	public static McpSchema.Prompt definition() {
		return McpSchema.Prompt.builder("analyze_bioimage")
			.description("Template for bioimage analysis workflows in Fiji")
			.arguments(List.of(
				McpSchema.PromptArgument.builder("task")
					.description("Analysis goal (e.g., 'segment objects', 'measure fluorescence')")
					.required(false)
					.build()))
			.build();
	}

	public McpSchema.GetPromptResult handle(McpSyncServerExchange exchange, McpSchema.GetPromptRequest request) {
		Map<String, Object> args = request.arguments();
		String task = args.getOrDefault("task", "segment objects").toString();

		String promptText = """
			I want to analyze bioimages in Fiji/ImageJ2.
			My goal is: %s

			Please help me by:
			1. First, check the Fiji environment with fiji_environment
			2. Capture the current screen with capture_fiji_screen to see what's open
			3. If no image is open, guide me to open one or use run_macro to open a sample
			4. Develop a step-by-step ImageJ macro workflow for my task
			5. Execute the macro with run_macro and verify results with capture_fiji_screen
			""".formatted(task);

		return McpSchema.GetPromptResult.builder(List.of(
				new McpSchema.PromptMessage(McpSchema.Role.USER, McpSchema.TextContent.builder(promptText).build())))
			.build();
	}
}
