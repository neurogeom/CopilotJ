/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj.awt.window;

import java.awt.Window;
import java.util.ServiceLoader;

import org.scijava.log.LogService;

import copilotj.awt.WindowIdentifier;

/**
 * Service Provider Interface for creating AwtWindow instances.
 * Implementations of this interface are discovered using
 * java.util.ServiceLoader.
 */
public interface AwtWindowProvider {
  /**
   * Special exception to indicate that the provider identify the window
   * but cannot create a node for it.
   */
  public static class InvalidWindowException extends Exception {
    public InvalidWindowException(String message) {
      super(message);
    }
  }

  public static AwtWindow create(WindowIdentifier identifier, Window w, LogService log) throws InvalidWindowException {
    if (!w.isVisible()) {
      throw new InvalidWindowException("Window is not visible");
    }

    // Use ServiceLoader to find and try providers
    final ServiceLoader<AwtWindowProvider> loader = ServiceLoader.load(AwtWindowProvider.class);
    for (final AwtWindowProvider provider : loader) {
      AwtWindow window = provider.tryCreate(identifier, w);
      if (window != null) {
        return window; // Provider successfully created a node
      }
    }

    // } else if (w instanceof ij.gui.PlotWindow) {
    // // PlotWindow pw = (PlotWindow) w;
    // window = new UnknownWindow(title, "PlotWindow");

    // Fallback for any other type
    return new UnknownWindow(identifier, w);
  }

  /**
   * Attempts to create a ComponentNode for the given AWT component.
   *
   * @param window The AWT window
   * @return A specific AwtWindow instance if this provider handles the
   *         component type, otherwise null.
   */
  AwtWindow tryCreate(WindowIdentifier identifier, Window window) throws InvalidWindowException;
}
