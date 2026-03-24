/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj.awt.window;

import java.awt.Window;

import copilotj.awt.WindowIdentifier;

public class UnknownWindow extends AbstractAwtWindow<Window> {
  public UnknownWindow(final WindowIdentifier identifier, final Window window) {
    super(identifier, window, window.getClass().getName());
  }

  @Override
  public String describe() {
    return "Unknown window: id=" + this.id;
  }
}
