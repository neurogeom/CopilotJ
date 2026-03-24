/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj.awt.component;

import java.awt.Component;
import java.awt.TextArea;
import java.awt.event.TextEvent;
import java.awt.event.TextListener;
import java.util.Collections;
import java.util.List;

import copilotj.awt.Action;

public class TextAreaNode extends AbstractComponentNode<TextArea> {
  public static class Provider implements ComponentNodeProvider {
    @Override
    public ComponentNode tryCreate(final Component component) {
      if (component instanceof TextArea) {
        return new TextAreaNode((TextArea) component);
      }
      return null;
    }
  }

  private static final String TYPE = "java.awt.TextArea";

  public final String text;

  public TextAreaNode(final TextArea component) {
    super(TYPE, component);
    this.text = component.getText() != null ? component.getText().trim() : null;
  }

  @Override
  public String describe() {
    // Limit potentially long text in debug output
    final String shortText = (text != null && text.length() > 50) ? text.substring(0, 47) + "..." : text;
    return "TextArea: text=" + shortText;
  }

  @Override
  public List<Action> getActions() {
    if (!this.component.isEditable()) {
      return null;
    }

    final Action setTextAction = Action
        .builder(TYPE + ".setText", "Set Text", "Sets the text of the text area.")
        .addStringParameter("text", "The text to set")
        .build();
    return Collections.singletonList(setTextAction);
  }

  @Override
  public Object runAction(final List<String> path, final String type, final List<Object> parameters) {
    if (!this.isActivate()) {
      throw new IllegalStateException("TextArea is not activated");
    } else if (path.size() != 0) {
      throw new IllegalArgumentException("Path must be empty for TextAreaNode actions like 'setText'");
    }

    switch (type) {
      case TYPE + ".setText":
        if (parameters == null || parameters.size() != 1) {
          throw new IllegalArgumentException(
              "Action 'setText' requires exactly one string 'text' parameter. Found: " +
                  (parameters == null ? 0 : parameters.size()) + " parameters.");
        }

        final Object param = parameters.get(0);
        // Allow null for text parameter, as TextArea.setText(null) is valid (sets to
        // empty string)
        if (param != null && !(param instanceof String)) {
          throw new IllegalArgumentException(
              "Action 'setText' requires a string 'text' parameter, but got " +
                  param.getClass().getSimpleName());
        }

        return setText((String) param);

      default:
        throw new IllegalArgumentException("Unknown action type: " + type + " for TextAreaNode");
    }
  }

  /**
   * Sets the text of the text area component and notifies listeners.
   * 
   * @param text The text to set.
   * @return null
   */
  public Object setText(final String text) {
    if (!component.isEditable()) { // AWT TextArea uses isEditable()
      throw new IllegalStateException("TextArea is not editable");
    }
    if (!component.isEnabled()) {
      throw new IllegalStateException("TextArea is not enabled");
    }

    this.component.setText(text);

    // Notify listeners
    // The TextEvent needs to indicate what changed (TEXT_VALUE_CHANGED).
    final TextEvent event = new TextEvent(this.component, TextEvent.TEXT_VALUE_CHANGED);

    for (final TextListener listener : this.component.getTextListeners()) {
      listener.textValueChanged(event);
    }

    return null;
  }
}
