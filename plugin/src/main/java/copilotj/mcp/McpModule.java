/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj.mcp;

import java.util.LinkedHashMap;
import java.util.Map;

import org.eclipse.jetty.ee10.servlet.ServletContextHandler;
import org.eclipse.jetty.ee10.servlet.ServletHolder;
import org.eclipse.jetty.server.Server;
import org.eclipse.jetty.server.ServerConnector;
import org.eclipse.jetty.util.thread.QueuedThreadPool;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fasterxml.jackson.databind.MapperFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.node.NullNode;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;

import copilotj.EventHandler;
import copilotj.mcp.resources.EnvironmentResource;
import copilotj.mcp.resources.WindowsResource;
import copilotj.mcp.prompts.AnalyzeBioimagePrompt;
import copilotj.mcp.prompts.DebugMacroPrompt;
import copilotj.mcp.tools.CallActionTool;
import copilotj.mcp.tools.CaptureImageTool;
import copilotj.mcp.tools.CaptureScreenTool;
import copilotj.mcp.tools.FijiEnvironmentTool;
import copilotj.mcp.tools.FolderSummaryTool;
import copilotj.mcp.tools.ListOperationsTool;
import copilotj.mcp.tools.RunMacroTool;
import copilotj.mcp.tools.RunScriptTool;
import copilotj.mcp.tools.TakeSnapshotTool;

import io.modelcontextprotocol.json.jackson2.JacksonMcpJsonMapper;
import io.modelcontextprotocol.server.McpServer;
import io.modelcontextprotocol.server.McpServerFeatures;
import io.modelcontextprotocol.server.McpSyncServer;
import io.modelcontextprotocol.server.transport.HttpServletStreamableServerTransportProvider;
import io.modelcontextprotocol.spec.McpSchema;

public class McpModule {

	private static final Logger log = LoggerFactory.getLogger(McpModule.class);

	private static Server jettyServer;
	private static McpSyncServer mcpServer;
	private static EventHandler eventHandler;

	public static final ObjectMapper objectMapper = new ObjectMapper()
		.registerModule(new JavaTimeModule())
		.configure(MapperFeature.SORT_PROPERTIES_ALPHABETICALLY, true)
		.configure(SerializationFeature.WRITE_DATES_AS_TIMESTAMPS, false);

	public static void start(EventHandler handler, String host, int port) throws Exception {
		// Resolve MCP's ServiceLoader (JsonSchemaValidatorSupplier, JSON mapper, ...)
		// from the isolated bundle rather than the thread's default classloader.
		// Without this, ServiceLoader.load(Class) uses the TCCL, which holds either a
		// duplicate of the MCP classes (dev classpath) or none at all (real Fiji),
		// causing "not a subtype" or "no provider found" failures.
		//
		// start() runs on the EDT (McpPanel start button), so the TCCL swap is scoped
		// to the build and restored afterwards. Otherwise the EDT's TCCL is left
		// pointing at the isolated MCP bundle loader, which corrupts later EDT
		// classloading — including Context.dispose() during Quit. Jetty worker threads
		// created inside startInternal capture the MCP loader at construction, so
		// restoring the EDT's TCCL afterwards does not affect request handling.
		final Thread currentThread = Thread.currentThread();
		final ClassLoader savedTccl = currentThread.getContextClassLoader();
		currentThread.setContextClassLoader(McpModule.class.getClassLoader());
		try {
			startInternal(handler, host, port);
		} finally {
			currentThread.setContextClassLoader(savedTccl);
		}
	}

	private static void startInternal(EventHandler handler, String host, int port) throws Exception {
		McpModule.eventHandler = handler;

		var transport = HttpServletStreamableServerTransportProvider.builder()
			.jsonMapper(new JacksonMcpJsonMapper(objectMapper))
			.mcpEndpoint("/mcp")
			.build();

		// Pre-create handler instances
		var runMacroTool = new RunMacroTool(handler);
		var runScriptTool = new RunScriptTool(handler);
		var captureScreenTool = new CaptureScreenTool(handler);
		var captureImageTool = new CaptureImageTool(handler);
		var takeSnapshotTool = new TakeSnapshotTool(handler);
		var callActionTool = new CallActionTool(handler);
		var fijiEnvironmentTool = new FijiEnvironmentTool(handler);
		var listOperationsTool = new ListOperationsTool(handler);
		var folderSummaryTool = new FolderSummaryTool();

		var envResource = new EnvironmentResource(handler);
		var windowsResource = new WindowsResource(handler);

		var analyzePrompt = new AnalyzeBioimagePrompt();
		var debugPrompt = new DebugMacroPrompt();

		mcpServer = McpServer.sync(transport)
			.serverInfo("CopilotJ", "0.1.0")
			.instructions(
				"CopilotJ provides Fiji/ImageJ2 bioimage analysis tools. "
				+ "Use capture_fiji_screen to see the current Fiji state, "
				+ "run_macro to execute ImageJ macros, and capture_image to "
				+ "inspect specific images. Start by checking the environment "
				+ "with fiji_environment.")
			// Tools — use method references for BiFunction compatibility
			.toolCall(RunMacroTool.definition(), runMacroTool::handle)
			.toolCall(RunScriptTool.definition(), runScriptTool::handle)
			.toolCall(CaptureScreenTool.definition(), captureScreenTool::handle)
			.toolCall(CaptureImageTool.definition(), captureImageTool::handle)
			.toolCall(TakeSnapshotTool.definition(), takeSnapshotTool::handle)
			.toolCall(CallActionTool.definition(), callActionTool::handle)
			.toolCall(FijiEnvironmentTool.definition(), fijiEnvironmentTool::handle)
			.toolCall(ListOperationsTool.definition(), listOperationsTool::handle)
			.toolCall(FolderSummaryTool.definition(), folderSummaryTool::handle)
			// Resources
			.resources(
				new McpServerFeatures.SyncResourceSpecification(
					EnvironmentResource.definition(), envResource::handle),
				new McpServerFeatures.SyncResourceSpecification(
					WindowsResource.definition(), windowsResource::handle))
			// Prompts
			.prompts(
				new McpServerFeatures.SyncPromptSpecification(
					AnalyzeBioimagePrompt.definition(), analyzePrompt::handle),
				new McpServerFeatures.SyncPromptSpecification(
					DebugMacroPrompt.definition(), debugPrompt::handle))
			.build();

		// Daemon, bounded thread pool so Jetty workers never pin the JVM on Quit.
		// (The default QueuedThreadPool from new Server(InetSocketAddress) uses
		// non-daemon threads, which keep the process alive after Fiji quits.)
		final QueuedThreadPool pool = new QueuedThreadPool();
		pool.setDaemon(true);
		pool.setName("copilotj-mcp");
		pool.setStopTimeout(5_000L);

		jettyServer = new Server(pool);
		final ServerConnector connector = new ServerConnector(jettyServer);
		connector.setHost(host);
		connector.setPort(port);
		connector.setIdleTimeout(60_000L);
		jettyServer.addConnector(connector);

		var ctx = new ServletContextHandler();
		ctx.addServlet(new ServletHolder(transport), "/mcp");
		jettyServer.setHandler(ctx);
		jettyServer.setStopTimeout(5_000L);
		jettyServer.start();

		log.info("MCP server started on {}:{}", host, port);
	}

	public static void stop() {
		try {
			if (jettyServer != null) {
				jettyServer.stop();
				jettyServer = null;
			}
		} catch (Exception e) {
			log.warn("Error stopping Jetty: {}", e.getMessage());
		}
		if (mcpServer != null) {
			mcpServer.close();
			mcpServer = null;
		}
		log.info("MCP server stopped");
	}

	public static boolean isRunning() {
		return jettyServer != null && jettyServer.isRunning();
	}

	public static String callEvent(EventHandler handler, String event, Object data) {
		try {
			var payload = createPayload(event, data);
			String requestJson = objectMapper.writeValueAsString(payload);
			String responseJson = handler.handle(requestJson);
			if (responseJson == null) {
				throw new RuntimeException("No response from Fiji for event: " + event);
			}
			var response = objectMapper.readTree(responseJson);
			if (response.has("err") && !response.get("err").isNull()) {
				throw new RuntimeException(response.get("err").asText());
			}
			return objectMapper.writerWithDefaultPrettyPrinter()
				.writeValueAsString(response.get("data"));
		} catch (RuntimeException e) {
			throw e;
		} catch (Exception e) {
			throw new RuntimeException("Event handler error: " + e.getMessage(), e);
		}
	}

	private static Map<String, Object> createPayload(String event, Object data) {
		var payload = new LinkedHashMap<String, Object>();
		payload.put("id", java.util.UUID.randomUUID().toString());
		payload.put("event_id", java.util.UUID.randomUUID().toString());
		payload.put("event", event);
		payload.put("data", data != null ? data : NullNode.instance);
		payload.put("err", NullNode.instance);
		return payload;
	}
}
