/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj.mcp;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.FlowLayout;

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

public class McpPanel extends JPanel {

	private final EventHandler handler;
	private final LogService log;

	private JTextField hostField;
	private JTextField portField;
	private JButton startButton;
	private JButton stopButton;
	private JLabel statusLabel;
	private JTextArea logArea;

	public McpPanel(EventHandler handler, LogService log) {
		this.handler = handler;
		this.log = log;
		buildUI();
	}

	private void buildUI() {
		hostField = new JTextField("127.0.0.1", 12);
		portField = new JTextField("3001", 6);
		startButton = new JButton("Start MCP");
		stopButton = new JButton("Stop MCP");
		stopButton.setEnabled(false);
		statusLabel = new JLabel("Stopped");

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
		configPanel.add(statusLabel, BorderLayout.SOUTH);

		// -- MCP log (mirrors the Managed Server tab's Progress Log) --
		logArea = new JTextArea(8, 40);
		logArea.setEditable(false);
		logArea.setFont(new java.awt.Font("Monospaced", java.awt.Font.PLAIN, 11));
		final JScrollPane logScroll = new JScrollPane(logArea);
		logScroll.setBorder(BorderFactory.createTitledBorder("MCP Log"));

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
		SwingUtilities.invokeLater(() -> {
			statusLabel.setText(text);
			statusLabel.setForeground(color);
		});
	}

	public void dispose() {
		if (McpModule.isRunning()) {
			McpModule.stop();
		}
	}
}
