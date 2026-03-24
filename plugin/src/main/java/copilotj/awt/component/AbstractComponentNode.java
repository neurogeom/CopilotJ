/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj.awt.component;

import java.awt.Component;
import java.util.List;

import copilotj.awt.Action;

public abstract class AbstractComponentNode<T extends Component> implements ComponentNode {
  public final String type;
  public final String name; // AWT component's name property
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
  public boolean isActivate() {
    return component != null && component.isShowing();
  }

  @Override
  public void deactivate() {
    this.component = null;
  }

  @Override
  public List<Action> getActions() {
    return null;
  }

  @Override
  public Object runAction(final List<String> path, final String type, final List<Object> parameters) {
    // TODO: change to register manager
    throw new UnsupportedOperationException("Action not supported: " + type);
  }

  @Override
  public void print() {
    // Print the root node itself using its description
    System.out.println(describe() + " (name=" + name + ", class=" + type + ")");
  }
}
