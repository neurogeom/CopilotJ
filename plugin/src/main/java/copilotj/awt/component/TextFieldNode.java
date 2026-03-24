/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj.awt.component;

import java.awt.Component;
import java.awt.TextField;
import java.awt.event.TextEvent;
import java.awt.event.TextListener;
import java.util.Collections;
import java.util.List;

import copilotj.awt.Action;

public class TextFieldNode extends AbstractComponentNode<TextField> {
  public static class Provider implements ComponentNodeProvider {
    @Override
    public ComponentNode tryCreate(final Component component) {
      if (component instanceof TextField) {
        return new TextFieldNode((TextField) component);
      }
      return null;
    }
  }

  private static final String TYPE = "java.awt.TextField";

  public final String text;

  public TextFieldNode(final TextField component) {
    super(TYPE, component);
    this.text = component.getText() != null ? component.getText().trim() : null;
  }

  @Override
  public String describe() {
    return "TextField: text=" + text;
  }

  @Override
  public List<Action> getActions() {
    if (this.component.isEditable()) {
      final Action setTextAction = Action
          .builder(TYPE + ".setText", "Set Text", "Sets the text of the text field.")
          .addStringParameter("text", "The text to set")
          .build();
      return Collections.singletonList(setTextAction);
    }
    return Collections.emptyList();
  }

  @Override
  public Object runAction(final List<String> path, final String type, final List<Object> parameters) {
    if (!this.isActivate()) {
      throw new IllegalStateException("TextField is not activated");
    } else if (path.size() != 0) {
      throw new IllegalArgumentException("Path must be empty for TextFieldNode actions like 'setText'");
    }

    switch (type) {
      case TYPE + ".setText":
        if (parameters == null || parameters.size() != 1) {
          throw new IllegalArgumentException(
              "Action 'setText' requires exactly one string 'text' parameter. Found: " +
                  (parameters == null ? 0 : parameters.size()) + " parameters.");
        }

        Object param = parameters.get(0);
        // Allow null for text parameter, as TextField.setText(null) is valid (sets to
        // empty string)
        if (param != null && !(param instanceof String)) {
          throw new IllegalArgumentException(
              "Action 'setText' requires a string 'text' parameter, but got " +
                  param.getClass().getSimpleName());
        }

        return setText((String) param);

      default:
        throw new IllegalArgumentException("Unknown action type: " + type + " for TextFieldNode");
    }
  }

  /**
   * Sets the text of the text field component and notifies listeners.
   * 
   * @param text The text to set.
   * @return null
   */
  public Object setText(String text) {
    if (!component.isEditable()) {
      throw new IllegalStateException("TextField is not editable");
    }
    if (!component.isEnabled()) {
      throw new IllegalStateException("TextField is not enabled");
    }

    this.component.setText(text);

    // Notify listeners
    // The TextEvent needs to indicate what changed (TEXT_VALUE_CHANGED).
    final TextEvent event = new TextEvent(this.component, TextEvent.TEXT_VALUE_CHANGED);

    // TextField uses ActionListener for Enter key press, but TextListener for text
    // changes.
    // We are simulating direct text manipulation, so TextListener is appropriate.
    for (TextListener listener : this.component.getTextListeners()) {
      listener.textValueChanged(event);
    }
    // Note: AWT TextField also fires ActionEvent when enter is pressed.
    // Simulating that is more complex and might not be desired for a simple
    // setText.
    // If enter key press simulation is needed, it would be a separate action.

    return null;
  }
}
