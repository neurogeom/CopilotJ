/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj.awt.component;

import java.awt.Component;
import java.awt.Checkbox;
import java.awt.event.ItemEvent;
import java.awt.event.ItemListener;
import java.util.Collections;
import java.util.List;

import copilotj.awt.Action;

/**
 * @see https://docs.oracle.com/javase/8/docs/api/java/awt/Checkbox.html
 */
public class CheckboxNode extends AbstractComponentNode<Checkbox> {
  public static class Provider implements ComponentNodeProvider {
    @Override
    public ComponentNode tryCreate(final Component component) {
      if (component instanceof Checkbox) {
        return new CheckboxNode((Checkbox) component);
      }
      return null;
    }
  }

  private static final String TYPE = "java.awt.Checkbox";

  public final String label;
  public final boolean state;

  public CheckboxNode(final Checkbox component) {
    super(TYPE, component);
    this.label = component.getLabel() != null ? component.getLabel().trim() : null;
    this.state = component.getState();
  }

  @Override
  public String describe() {
    return "Checkbox: label=" + label + ", state=" + state;
  }

  public List<Action> getActions() {
    final Action setStateAction = Action
        .builder(TYPE + ".setState", "Set State", "Sets the checkbox to a specific state.")
        .addBooleanParameter("state", "The desired state (true for checked, false for unchecked)")
        .build();
    return Collections.singletonList(setStateAction);
  }

  @Override
  public Object runAction(final List<String> path, final String type, final List<Object> parameters) {
    if (!this.isActivate()) {
      throw new IllegalStateException("Checkbox is not activated");
    } else if (path.size() != 0) {
      throw new IllegalArgumentException("Path must be empty for CheckboxNode actions like 'setState'");
    }

    switch (type) {
      case TYPE + ".setState":
        if (parameters == null || parameters.size() != 1) {
          throw new IllegalArgumentException(
              "Action 'setState' requires exactly one boolean 'state' parameter. Found: " +
                  (parameters == null ? 0 : parameters.size()) + " parameters.");
        }

        final Object param = parameters.get(0);
        if (!(param instanceof Boolean)) {
          throw new IllegalArgumentException(
              "Action 'setState' requires a boolean 'state' parameter, but got " +
                  param.getClass().getSimpleName());
        }

        return setState((Boolean) param);

      default:
        throw new IllegalArgumentException("Unknown action type: " + type + " for CheckboxNode");
    }
  }

  /**
   * Sets the state of the checkbox component and notifies listeners.
   * 
   * @param desiredState The desired state (true for checked, false for
   *                     unchecked).
   * @return null
   */
  public Object setState(final boolean desiredState) {
    if (!component.isEnabled()) {
      throw new IllegalStateException("Checkbox is not enabled");
    }

    this.component.setState(desiredState);

    // Notify listeners
    // Note: Checkbox uses ItemListener, not ActionListener like Button.
    // The ItemEvent needs to indicate what changed (ITEM_STATE_CHANGED)
    // and the new state (SELECTED or DESELECTED).
    final ItemEvent event = new ItemEvent(this.component, ItemEvent.ITEM_STATE_CHANGED, this.component,
        desiredState ? ItemEvent.SELECTED : ItemEvent.DESELECTED);

    for (final ItemListener listener : this.component.getItemListeners()) {
      listener.itemStateChanged(event);
    }

    return null;
  }
}
