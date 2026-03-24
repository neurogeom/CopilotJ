/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj.awt.component;

import java.awt.Component;
import java.awt.Scrollbar;

public class ScrollbarNode extends AbstractComponentNode<Scrollbar> {
  public static class Provider implements ComponentNodeProvider {
    @Override
    public ComponentNode tryCreate(final Component component) {
      if (component instanceof Scrollbar) {
        return new ScrollbarNode((Scrollbar) component);
      }
      return null;
    }
  }

  public final int value;
  private final int orientation; // e.g., Scrollbar.VERTICAL or Scrollbar.HORIZONTAL

  public ScrollbarNode(final Scrollbar component) {
    super("java.awt.Scrollbar", component);
    this.value = component.getValue();
    this.orientation = component.getOrientation();
  }

  public String getOrientation() {
    switch (this.orientation) {
      case Scrollbar.VERTICAL:
        return "vertical";

      case Scrollbar.HORIZONTAL:
        return "horizontal";

      default:
        throw new IllegalArgumentException("Unknown orientation: " + this.orientation);
    }
  }

  @Override
  public String describe() {
    return "Scrollbar: value=" + value + ", orientation=" + this.getOrientation();
  }
}
