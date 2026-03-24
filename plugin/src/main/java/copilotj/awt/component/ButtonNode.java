/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj.awt.component;

import java.awt.Button;
import java.awt.Component;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.Collections;
import java.util.List;

import copilotj.awt.Action;

/**
 * @see https://docs.oracle.com/javase/8/docs/api/java/awt/Button.html
 */
public class ButtonNode extends AbstractComponentNode<Button> {
  public static class Provider implements ComponentNodeProvider {
    @Override
    public ComponentNode tryCreate(final Component component) {
      if (component instanceof Button) {
        return new ButtonNode((Button) component);
      }
      return null;
    }
  }

  private static final String TYPE = "java.awt.Button";

  public final String label;

  public ButtonNode(final Button component) {
    super(TYPE, component);
    this.label = component.getLabel() != null ? component.getLabel().trim() : null;
  }

  @Override
  public String describe() {
    return "Button: label=" + label;
  }

  @Override
  public List<Action> getActions() {
    final Action click = Action
        .builder(TYPE + ".click", "Click", "Click on the button \"" + this.label + "\"")
        .build();
    return Collections.singletonList(click);
  }

  @Override
  public Object runAction(final List<String> path, final String type, final List<Object> parameters) {
    if (!this.isActivate()) {
      throw new IllegalStateException("Button is not activated");
    } else if (path.size() != 0) {
      throw new IllegalArgumentException("Path must be empty for ButtonNode");
    }

    switch (type) {
      case TYPE + ".click":
        if (parameters != null && parameters.size() != 0) {
          throw new IllegalArgumentException("Action 'Click' does not accept any parameters. Found: " + parameters);
        }

        return click();

      default:
        throw new IllegalArgumentException("Unknown action type: " + type);
    }
  }

  public Object click() {
    if (!component.isEnabled()) {
      throw new IllegalStateException("Button is not enabled");
    }

    final ActionEvent event = new ActionEvent(this.component, ActionEvent.ACTION_PERFORMED,
        this.component.getActionCommand());
    // component.dispatchEvent(event);
    for (final ActionListener listener : this.component.getActionListeners()) {
      listener.actionPerformed(event);
    }

    return null;
  }
}
