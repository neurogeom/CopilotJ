/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj.mcp;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.FlowLayout;
import java.util.function.BiConsumer;
import java.util.function.BooleanSupplier;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;
import javax.swing.JTextField;
import javax.swing.SwingUtilities;

import org.scijava.log.LogService;

import copilotj.EventHandler;
import copilotj.McpControl;

public class McpPanel extends JPanel implements McpControl {

	private final EventHandler handler;
	private final LogService log;
	// Core-supplied guard run right before starting MCP; returns false to abort
	// (e.g. a managed/standalone server is running and the user declined to stop it).
	private final BooleanSupplier requireExclusiveControl;

	private JTextField hostField;
	private JTextField portField;
	private JButton startButton;
	private JButton stopButton;
	// Routes MCP status (text + color) to the dialog's shared status bar. Merged
	// because only one control endpoint is active at a time, so a single status
	// bar can reflect it. Supplied by the core dialog.
	private final BiConsumer<String, Color> statusUpdater;
	private JTextArea logArea;

	public McpPanel(EventHandler handler, LogService log, BooleanSupplier requireExclusiveControl,
			BiConsumer<String, Color> statusUpdater) {
		this.handler = handler;
		this.log = log;
		this.requireExclusiveControl = requireExclusiveControl;
		this.statusUpdater = statusUpdater;
		buildUI();
	}

	private void buildUI() {
		hostField = new JTextField("127.0.0.1", 12);
		portField = new JTextField("3001", 6);
		startButton = new JButton("Start MCP");
		stopButton = new JButton("Stop MCP");
		stopButton.setEnabled(false);

		startButton.addActionListener(e -> {
			String host = hostField.getText().trim();
			int port;
			try {
				port = Integer.parseInt(portField.getText().trim());
			} catch (NumberFormatException ex) {
				setStatus("Invalid port number", Color.RED);
				return;
			}
			try {
				// Fiji exposes a single control endpoint at a time. If a managed
				// or standalone server is running, the core-supplied guard prompts
				// the user and stops it; returning false aborts the start.
				if (requireExclusiveControl != null && !requireExclusiveControl.getAsBoolean()) {
					return;
				}
				McpModule.start(handler, host, port);
				setStatus("Running on " + host + ":" + port, new Color(0, 128, 0));
				startButton.setEnabled(false);
				stopButton.setEnabled(true);
				log.info("MCP server started on " + host + ":" + port);
				appendLog("MCP server started on " + host + ":" + port + "\n");
			} catch (Exception ex) {
				setStatus("Failed: " + ex.getMessage(), Color.RED);
				log.warn("Failed to start MCP: " + ex.getMessage());
				appendLog("Failed to start MCP: " + ex.getMessage() + "\n");
			}
		});

		stopButton.addActionListener(e -> {
			McpModule.stop();
			setStatus("Stopped", Color.BLACK);
			startButton.setEnabled(true);
			stopButton.setEnabled(false);
			appendLog("MCP server stopped.\n");
		});

		// -- MCP log (mirrors the Managed Server tab's Progress Log) --
		// Created BEFORE the isRunning() check below: when the window reopens with
		// MCP already running in the background, that branch calls appendLog(...),
		// which dereferences logArea. If logArea is still null the NPE aborts panel
		// construction (swallowed by McpLoader) and the dialog falls back to the
		// "Fiji-Latest / Java 17+" unavailable label.
		logArea = new JTextArea(8, 40);
		logArea.setEditable(false);
		logArea.setFont(new java.awt.Font("Monospaced", java.awt.Font.PLAIN, 11));
		final JScrollPane logScroll = new JScrollPane(logArea);
		logScroll.setBorder(BorderFactory.createTitledBorder("MCP Log"));

		// Check if already running
		if (McpModule.isRunning()) {
			setStatus("Running", new Color(0, 128, 0));
			startButton.setEnabled(false);
			stopButton.setEnabled(true);
			appendLog("MCP server is already running.\n");
		}

		JPanel hostPortPanel = new JPanel(new FlowLayout(FlowLayout.LEFT, 5, 0));
		hostPortPanel.add(new JLabel("Host:"));
		hostPortPanel.add(hostField);
		hostPortPanel.add(new JLabel("Port:"));
		hostPortPanel.add(portField);

		JPanel buttonPanel = new JPanel(new FlowLayout(FlowLayout.LEFT, 5, 0));
		buttonPanel.add(startButton);
		buttonPanel.add(stopButton);

		JPanel configPanel = new JPanel(new BorderLayout(5, 5));
		configPanel.add(hostPortPanel, BorderLayout.NORTH);
		configPanel.add(buttonPanel, BorderLayout.CENTER);

		setLayout(new BorderLayout(5, 5));
		setBorder(BorderFactory.createTitledBorder("MCP Server"));
		add(configPanel, BorderLayout.NORTH);
		add(logScroll, BorderLayout.CENTER);
	}

	private void appendLog(String msg) {
		logArea.append(msg);
		logArea.setCaretPosition(logArea.getDocument().getLength());
	}

	private void setStatus(String text, Color color) {
		if (statusUpdater == null) return;
		// Keep invokeLater: the "already running" path calls setStatus from this
		// panel's constructor, before the dialog's shared statusLabel exists;
		// deferring lets the dialog's lambda run after run() has built it.
		SwingUtilities.invokeLater(() -> statusUpdater.accept(text, color));
	}

	public void dispose() {
		stopMcp();
	}

	@Override
	public boolean isMcpRunning() {
		return McpModule.isRunning();
	}

	@Override
	public void stopMcp() {
		if (McpModule.isRunning()) {
			McpModule.stop();
		}
	}
}
