/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj.awt.component;

import java.awt.Component;
import java.util.List;

import copilotj.awt.Action;
import copilotj.awt.ComponentIdentifier;

public abstract class AbstractComponentNode<T extends Component> implements ComponentNode {
  public final String type;
  public final String name; // AWT component's name property

  /**
   * Playwright-mcp-style ref handle ({@code "e" + int}) for this component, or
   * {@code null} when the node is not ref-eligible. Assigned during snapshot
   * build via {@link #assignRef(ComponentIdentifier)} and serialized to clients.
   */
  public String ref;

  /**
   * The actions available on this component, captured at snapshot time (while
   * the underlying AWT component is still live). Serialized per-component so the
   * client can show each widget's short action ids inline. May be {@code null}
   * or empty for descriptive-only nodes.
   */
  public List<Action> actions;

  protected T component;

  protected AbstractComponentNode(final String type, final T component) {
    this.type = type;
    this.name = component.getName();
    this.component = component;
  }

  @Override
  public String getName() {
    return name;
  }

  @Override
  public String getType() {
    return type;
  }

  @Override
  public String getRef() {
    return ref;
  }

  @Override
  public List<Action> getActions() {
    return actions;
  }

  @Override
  public boolean isRefEligible() {
    return true;
  }

  @Override
  public void assignRef(final ComponentIdentifier identifier) {
    if (isRefEligible() && component != null) {
      this.ref = "e" + identifier.getRef(component);
    }
  }

  @Override
  public boolean isActivate() {
    return component != null && component.isShowing();
  }

  @Override
  public void deactivate() {
    this.component = null;
  }

  @Override
  public void print() {
    // Print the root node itself using its description
    System.out.println(describe() + " (name=" + name + ", class=" + type + ")");
  }
}
