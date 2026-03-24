/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj.awt.component;

import java.awt.Component;
import java.awt.Label;

public class LabelNode extends AbstractComponentNode<Label> {
  public static class Provider implements ComponentNodeProvider {
    @Override
    public ComponentNode tryCreate(final Component component) {
      if (component instanceof Label) {
        return new LabelNode((Label) component);
      }
      return null;
    }
  }

  public final String text;

  public LabelNode(final Label component) {
    super("java.awt.Label", component);
    this.text = component.getText() != null ? component.getText().trim() : null;
  }

  @Override
  public String describe() {
    return "Label: text=" + text;
  }
}
