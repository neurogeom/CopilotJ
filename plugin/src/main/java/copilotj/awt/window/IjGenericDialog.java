/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj.awt.window;

import java.awt.Window;

import ij.gui.GenericDialog;

import copilotj.ImagejListener;
import copilotj.awt.WindowIdentifier;

public class IjGenericDialog extends AbstractAwtWindow<GenericDialog> {
  public static class Provider implements AwtWindowProvider {
    @Override
    public AwtWindow tryCreate(final WindowIdentifier identifier, final Window window) {
      if (window instanceof GenericDialog) {
        return new IjGenericDialog(identifier, (GenericDialog) window);
      }
      return null;
    }
  }

  private static final String TYPE = "ij.gui.GenericDialog";

  public IjGenericDialog(final WindowIdentifier identifier, final ij.gui.GenericDialog window) {
    super(identifier, window, TYPE);
  }

  @Override
  public String describe() {
    return "GenericDialog: title=" + this.component.getTitle() + ", id=" + this.id;
  }

  @Override
  public AwtWindow.Difference compare(final AwtWindow from, final ImagejListener.HistoryResponse history) {
    if (!(from instanceof GenericDialog)) {
      return super.compare(from, history);
    }

    // TODO: Implement meaningful comparison for GenericDialog states if required.
    // For now, mirroring IjImageJ's approach of not providing detailed comparisons
    // for this specific window type, as GenericDialogs can be highly variable.
    return null;
  }
}
