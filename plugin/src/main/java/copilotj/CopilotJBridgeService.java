/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj;

import org.scijava.service.SciJavaService;

public interface CopilotJBridgeService extends SciJavaService {

  public EventHandler getEventHandler();

  public String getServerUrl();

  public Connection getConnection();

  public void start(final String serverURL);

  public void stop();

  public void ensureEnvironment() throws java.io.IOException;

  public void ensureEnvironment(ProgressListener listener) throws java.io.IOException;

  public void syncEnvironment(ProgressListener listener) throws java.io.IOException;

  public void uninstallEnvironment() throws java.io.IOException;

  public boolean isEnvironmentReady();

  public boolean isEnvironmentOnDisk();

  public void startManagedServer() throws java.io.IOException, InterruptedException;

  public void startManagedServer(ProgressListener listener) throws java.io.IOException, InterruptedException;

  public boolean isManaged();

  public boolean isServerRunning();

  /**
   * Resolved on-disk root directory holding extracted Python sources, assets,
   * knowledge bank, and the virtual environment (a.k.a. {@code $COPILOTJ_HOME}).
   * Returns a deterministic path even if the directory has not been created yet.
   */
  public java.io.File getEnvironmentRoot();
}
