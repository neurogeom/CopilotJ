/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj.awt.window;

import java.awt.Window;

import copilotj.ImagejListener;
import copilotj.awt.WindowIdentifier;

import ij.ImageJ;

public class IjImageJ extends AbstractAwtWindow<ImageJ> {
  public static class Provider implements AwtWindowProvider {
    @Override
    public AwtWindow tryCreate(final WindowIdentifier identifier, final Window window) {
      if (window instanceof ImageJ) {
        return new IjImageJ(identifier, (ImageJ) window);
      }
      return null;
    }
  }

  private static final String TYPE = "ij.ImageJ";

  public IjImageJ(final WindowIdentifier identifier, final ImageJ window) {
    super(identifier, window, TYPE);
  }

  @Override
  public String describe() {
    return "ImageJ: id=" + this.id;
  }

  @Override
  public AwtWindow.Difference compare(final AwtWindow from, final ImagejListener.HistoryResponse history) {
    if (!(from instanceof IjImageJ)) {
      return super.compare(from, history);
    }

    // always return null, because ImageJ windows are not comparable
    return null;
  }
}
