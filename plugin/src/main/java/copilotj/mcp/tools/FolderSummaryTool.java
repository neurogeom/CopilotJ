/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj.mcp.tools;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

import io.modelcontextprotocol.server.McpSyncServerExchange;
import io.modelcontextprotocol.spec.McpSchema;

public class FolderSummaryTool {

	private static final int MAX_FILES = 300;

	public static McpSchema.Tool definition() {
		return McpSchema.Tool.builder()
			.name("folder_summary")
			.description("List files in a directory on the local filesystem. "
				+ "Useful for discovering image files to open in Fiji. "
				+ "Returns up to 300 items with relative paths.")
			.inputSchema(new McpSchema.JsonSchema("object",
				Map.of("folder_path", Map.of("type", "string", "description", "Absolute or relative directory path")),
				List.of("folder_path"), true, null, null))
			.build();
	}

	public McpSchema.CallToolResult handle(McpSyncServerExchange exchange, McpSchema.CallToolRequest request) {
		Map<String, Object> args = request.arguments();
		String folderPath = (String) args.get("folder_path");

		if (folderPath == null || folderPath.trim().isEmpty() || folderPath.trim().equals(".")) {
			return error("Please provide a specific folder path.");
		}

		Path folder = Path.of(folderPath);
		if (!Files.exists(folder) || !Files.isDirectory(folder)) {
			return error("The path '" + folderPath + "' is not a valid directory.");
		}

		List<String> items = new ArrayList<>();
		try (Stream<Path> walk = Files.walk(folder)) {
			walk.limit(MAX_FILES + 1).forEach(p -> {
				if (items.size() >= MAX_FILES) return;
				Path relative = folder.relativize(p);
				if (p.equals(folder)) return;
				if (Files.isDirectory(p)) {
					items.add("Directory: " + relative);
				} else {
					items.add(String.valueOf(relative));
				}
			});
		} catch (IOException e) {
			return error("Failed to read directory: " + e.getMessage());
		}

		String totalMsg = items.size() >= MAX_FILES
			? " (Showing first " + MAX_FILES + " items, more files may exist)"
			: "";

		StringBuilder sb = new StringBuilder();
		sb.append("Folder: ").append(folderPath).append(totalMsg).append("\n");
		for (int i = 0; i < items.size(); i++) {
			sb.append("  ").append(i + 1).append(". ").append(items.get(i)).append("\n");
		}

		return McpSchema.CallToolResult.builder()
			.content(List.of(new McpSchema.TextContent(sb.toString())))
			.build();
	}

	private static McpSchema.CallToolResult error(String msg) {
		return McpSchema.CallToolResult.builder()
			.content(List.of(new McpSchema.TextContent(msg)))
			.isError(true)
			.build();
	}
}
