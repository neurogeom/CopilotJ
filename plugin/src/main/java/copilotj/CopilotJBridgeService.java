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

  public String getAccessToken();

  public Connection getConnection();

  public void start(final String serverURL, final String accessToken);
}
