/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj.awt.window;

import java.awt.Component;
import java.awt.Container;
import java.awt.Window;
import java.util.Objects;

import javax.swing.JFrame;
import javax.swing.JLayeredPane;
import javax.swing.JPanel;
import javax.swing.JTabbedPane;

import org.scijava.ui.swing.console.ConsolePanel;

import ij.gui.GenericDialog;

import copilotj.ImagejListener;
import copilotj.awt.WindowIdentifier;

public class Ij2ConsoleWindow extends AbstractAwtWindow<JFrame> {
  public static class Provider implements AwtWindowProvider {
    @Override
    public AwtWindow tryCreate(final WindowIdentifier identifier, final Window window) {
      if (window instanceof JFrame) {
        final JFrame frame = (JFrame) window;
        if (!Objects.equals(frame.getTitle(), "Console")) {
          return null;
        }

        final int[] path = { 1, 0, 0, 0 };
        final Class<?>[] components = {
            JLayeredPane.class, JPanel.class, JTabbedPane.class, ConsolePanel.class
        };

        Container node = ((JFrame) window).getRootPane();
        for (int i = 0; i < path.length; i++) {
          if (path[i] >= node.getComponentCount()) {
            return null; // out of bounds
          }

          final Component child = node.getComponent(path[i]);
          if (!components[i].isInstance(child)) {
            return null;
          }

          node = (Container) child;
          if (i < path.length - 1 && node == null) { // should not happen
            return null;
          }
        }

        return new Ij2ConsoleWindow(identifier, (JFrame) window);
      }

      return null;
    }
  }

  private static final String TYPE = "ij.Console";

  public Ij2ConsoleWindow(final WindowIdentifier identifier, final javax.swing.JFrame frame) {
    super(identifier, frame, TYPE);
  }

  @Override
  public String describe() {
    return "Console: id=" + this.id;
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
