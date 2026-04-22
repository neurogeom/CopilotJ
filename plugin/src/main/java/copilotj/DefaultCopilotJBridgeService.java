/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj;

import org.scijava.Context;
import org.scijava.Priority;
import org.scijava.event.EventService;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.service.AbstractService;
import org.scijava.service.Service;
import org.scijava.ui.UIService;
import org.scijava.ui.event.UIShownEvent;

import net.imagej.ImageJ;
import net.imagej.legacy.LegacyService;
import net.imagej.patcher.LegacyInjector;

@Plugin(type = Service.class, priority = Priority.LOW)
public class DefaultCopilotJBridgeService extends AbstractService implements CopilotJBridgeService {
  static {
    LegacyInjector.preinit();
  }

  /**
   * This main function serves for development purposes.
   * It allows you to run the plugin immediately out of
   * your integrated development environment (IDE).
   *
   * @param args whatever, it's ignored
   * @throws Exception
   */
  public static void main(final String... args) throws Exception {
    final ImageJ ij = new ImageJ();
    ij.launch(args);
  }

  @Parameter
  private Context context;

  @Parameter
  private UIService ui;

  @Parameter
  private EventService eventService;

  // NOTE: DONT DELETE: This is needed to initialize ImageJ1
  @Parameter
  private LegacyService legacy;

  @Parameter
  private LogService log;

  private final String serverUrl = "http://127.0.0.1:8786";
  private String accessToken = null;
  private EventHandler eventHandler;
  private Connection connection;

  public DefaultCopilotJBridgeService() {
  }

  @Override
  public EventHandler getEventHandler() {
    return eventHandler;
  }

  @Override
  public String getServerUrl() {
    return serverUrl;
  }

  @Override
  public String getAccessToken() {
    return accessToken;
  }

  @Override
  public Connection getConnection() {
    return connection;
  }

  @Override
  public void initialize() {
    log.info("Initialize CopilotJBridge");

    final boolean debug = Boolean.getBoolean("ij.debug");
    this.eventHandler = new EventHandler(context, log, debug);

    final String token = System.getProperty("copilotj.accessToken");
    if (token != null && !token.isEmpty()) {
      this.accessToken = token;
    }

    if (debug) {
      log.info("Automatically starting connection to " + serverUrl);
      this.start(serverUrl, accessToken);
    }

    if (ui.isVisible()) {
      installActionTool();
    } else {
      eventService.subscribe(this);
    }
  }

  @Override
  public void start(final String serverURL, final String accessToken) {
    if (this.connection != null) {
      this.connection.close();
      this.connection = null;
    }

    log.info("copilotj: Start connection " + serverURL);
    final int maxRetryWaitSecond = Integer.getInteger("copilotj.maxRetryWaitSecond", 64);
    this.accessToken = accessToken;
    this.connection = new Connection(serverURL, eventHandler, log, maxRetryWaitSecond, accessToken);
    this.connection.connect();
  }

  @org.scijava.event.EventHandler
  private void onUIShown(final UIShownEvent e) {
    installActionTool();
  }

  void installActionTool() {
    CopilotJBridgeActionToolInstaller.install();
  }
}
