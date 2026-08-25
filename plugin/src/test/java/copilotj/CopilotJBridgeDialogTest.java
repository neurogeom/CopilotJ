/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

/**
 * Unit tests for the Progress Log retry hint shown after a failed Install/Sync.
 * Lives in package {@code copilotj} so it can call the package-private
 * {@link CopilotJBridgeDialog#retryHint} and
 * {@link CopilotJBridgeDialog#isLikelyNetworkFailure} directly.
 */
public class CopilotJBridgeDialogTest {

  @Test
  public void truncatedDownloadIsRecognizedAsNetworkFailure() {
    // The message users actually reported after a dropped connection.
    assertTrue(CopilotJBridgeDialog.isLikelyNetworkFailure("Unexpected end of file from server"));
  }

  @Test
  public void commonTransportErrorsAreRecognized() {
    final String[] messages = {
        "Connection reset",
        "Read timed out",
        "java.net.UnknownHostException: pypi.org",
        "Network is unreachable",
        "SSL handshake failed",
        "Failed to download python-3.12 (proxy error)"};
    for (final String message : messages) {
      assertTrue(CopilotJBridgeDialog.isLikelyNetworkFailure(message), message);
    }
  }

  @Test
  public void unrelatedFailuresAreNotTreatedAsNetworkFailures() {
    assertFalse(CopilotJBridgeDialog.isLikelyNetworkFailure(null));
    assertFalse(CopilotJBridgeDialog.isLikelyNetworkFailure("Permission denied"));
    assertFalse(CopilotJBridgeDialog.isLikelyNetworkFailure("No space left on device"));
  }

  @Test
  public void networkHintTellsTheUserToRetryTheNamedButton() {
    final String hint = CopilotJBridgeDialog.retryHint("Unexpected end of file from server", "Install");
    assertTrue(hint.contains("interrupted download"), hint);
    assertTrue(hint.contains("Click Install again"), hint);
    assertTrue(hint.endsWith("\n"), hint);
  }

  @Test
  public void genericHintStillOffersARetryAndAWayToReport() {
    final String hint = CopilotJBridgeDialog.retryHint("Permission denied", "Sync");
    assertTrue(hint.contains("Click Sync again"), hint);
    assertTrue(hint.contains("github.com/neurogeom/CopilotJ/issues"), hint);
    assertTrue(hint.endsWith("\n"), hint);
  }
}
