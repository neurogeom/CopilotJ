/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj.awt.component;

import java.awt.Component;
import java.awt.Canvas;

/**
 * @see https://docs.oracle.com/javase/8/docs/api/java/awt/Canvas.html
 */
public class CanvasNode extends AbstractComponentNode<Canvas> {
  public static class Provider implements ComponentNodeProvider {
    @Override
    public ComponentNode tryCreate(final Component component) {
      if (component instanceof Canvas) {
        return new CanvasNode((Canvas) component);
      }
      return null;
    }
  }

  public CanvasNode(final Canvas component) {
    super("java.awt.Canvas", component);
  }

  @Override
  public String describe() {
    return "Canvas";
  }
}
