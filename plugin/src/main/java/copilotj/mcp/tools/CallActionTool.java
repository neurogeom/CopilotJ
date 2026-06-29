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

public class CallActionTool {

	private final EventHandler handler;

	public CallActionTool(EventHandler handler) {
		this.handler = handler;
	}

	public static McpSchema.Tool definition() {
		return McpSchema.Tool.builder()
			.name("call_action")
			.description("Execute a UI action on a component identified by its ref handle. "
				+ "First call take_snapshot() to inspect the component tree: each actionable "
				+ "component carries a ref handle and a per-component actions list. Then call "
				+ "this with the component's ref, the action's short id (e.g. click, setState, "
				+ "selectItem, setValue), and the positional parameters. A ref stays valid while the "
				+ "underlying component is alive; if it is not found the widget closed or the UI "
				+ "changed, so capture a new snapshot. After the action, take a fresh snapshot to "
				+ "observe the result.")
			.inputSchema(new McpSchema.JsonSchema("object",
				Map.of(
					"ref", Map.of("type", "string", "description", "Component ref handle from take_snapshot (e.g. e5)"),
					"action", Map.of("type", "string",
						"description", "Action short id — the part after the last dot of the action type (e.g. click, setState)"),
					"parameters", Map.of("type", "array",
						"description", "Positional arguments in the order given by the action's parameters[] in the snapshot, e.g. [false] for setState or [42] for setValue",
						"items", Map.of())
				),
				List.of("ref", "action"), true, null, null))
			.build();
	}

	public McpSchema.CallToolResult handle(McpSyncServerExchange exchange, McpSchema.CallToolRequest request) {
		Map<String, Object> args = request.arguments();
		var data = new java.util.LinkedHashMap<String, Object>();
		data.put("ref", args.get("ref"));
		data.put("action", args.get("action"));
		if (args.containsKey("parameters")) {
			data.put("parameters", args.get("parameters"));
		}

		try {
			String result = McpModule.callEvent(handler, "run_action", data);
			return McpSchema.CallToolResult.builder()
				.content(List.of(new McpSchema.TextContent(result)))
				.build();
		} catch (Exception e) {
			return McpSchema.CallToolResult.builder()
				.content(List.of(new McpSchema.TextContent("Failed to execute action: " + e.getMessage())))
				.isError(true)
				.build();
		}
	}
}
