/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj.awt.component;

import java.awt.Component;
import java.awt.Scrollbar;
import java.awt.event.AdjustmentEvent;
import java.awt.event.AdjustmentListener;
import java.util.Collections;
import java.util.List;

import copilotj.awt.Action;

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

  private static final String TYPE = "java.awt.Scrollbar";

  public final int value;
  public final int minimum;
  public final int maximum;
  public final int visibleAmount;
  private final int orientation; // e.g., Scrollbar.VERTICAL or Scrollbar.HORIZONTAL

  public ScrollbarNode(final Scrollbar component) {
    super(TYPE, component);
    this.value = component.getValue();
    this.minimum = component.getMinimum();
    this.maximum = component.getMaximum();
    this.visibleAmount = component.getVisibleAmount();
    this.orientation = component.getOrientation();
    if (component.isEnabled()) {
      this.actions = Collections.singletonList(Action
          .builder(TYPE + ".setValue", "Set Value", "Sets the scrollbar value.")
          .addIntegerParameter("value", "The new value", this.minimum, this.maximum)
          .build());
    }
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

  @Override
  public Object runAction(final String action, final List<Object> parameters) {
    if (!this.isActivate()) {
      throw new IllegalStateException("Scrollbar is not activated");
    }

    switch (action) {
      case "setValue":
        if (parameters == null || parameters.size() != 1) {
          throw new IllegalArgumentException(
              "Action 'setValue' requires exactly one integer 'value' parameter. Found: " +
                  (parameters == null ? 0 : parameters.size()) + " parameters.");
        }

        final Object param = parameters.get(0);
        if (!(param instanceof Integer)) {
          throw new IllegalArgumentException(
              "Action 'setValue' requires an integer 'value' parameter, but got " +
                  (param != null ? param.getClass().getSimpleName() : "null"));
        }

        return setValue((Integer) param);

      default:
        throw new IllegalArgumentException("Unknown action: " + action + " for ScrollbarNode");
    }
  }

  /**
   * Sets the value of the scrollbar component and notifies adjustment listeners.
   *
   * @param value The new value (clamped to the scrollbar's range by AWT).
   * @return null
   */
  public Object setValue(final int value) {
    if (!component.isEnabled()) {
      throw new IllegalStateException("Scrollbar is not enabled");
    }

    this.component.setValue(value);

    // Notify listeners. AWT Scrollbar fires an AdjustmentEvent when its value
    // changes; we mimic a user-driven (TRACK) adjustment so ImageJ listeners
    // (e.g. the Brightness/Contrast or Threshold adjusters) react.
    final AdjustmentEvent event = new AdjustmentEvent(
        this.component,
        AdjustmentEvent.ADJUSTMENT_VALUE_CHANGED,
        AdjustmentEvent.TRACK,
        this.component.getValue());

    for (final AdjustmentListener listener : this.component.getAdjustmentListeners()) {
      listener.adjustmentValueChanged(event);
    }

    return null;
  }
}
