/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj.awt.container;

import java.awt.Component;
import java.awt.Container;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import copilotj.awt.Action;
import copilotj.awt.component.AbstractComponentNode;
import copilotj.awt.component.ComponentNode;
import copilotj.awt.component.ComponentNodeProvider;

/**
 * Generic node for container types like Panel, Window, Frame, Dialog etc.
 *
 * @see https://docs.oracle.com/javase/8/docs/api/java/awt/Container.html
 */
public class AbstractContainerNode<T extends Container> extends AbstractComponentNode<T> implements ContainerNode {
  public final boolean isContainer = true;
  public final List<ComponentNode> children;

  public AbstractContainerNode(final String type, final T component) {
    super(type, component);

    final List<ComponentNode> children = new ArrayList<>();

    // Recursively build the tree for its children.
    for (final Component childComponent : component.getComponents()) {
      final ComponentNode childNode = ComponentNodeProvider.create(childComponent);
      if (childNode != null) {
        children.add(childNode);
      }
    }
    this.children = children.size() > 0 ? children : null;
  }

  @Override
  public boolean isActivate() {
    return super.isActivate() && (children == null || children.stream().allMatch(ComponentNode::isActivate));
  }

  @Override
  public void deactivate() {
    super.deactivate();
    if (children != null) {
      children.forEach(ComponentNode::deactivate);
    }
  }

  public List<Action> getActions() {
    if (children == null || children.size() == 0) {
      return null;
    }

    final List<Action> actions = new ArrayList<>();
    for (int i = 0; i < children.size(); i++) {
      final ComponentNode child = children.get(i);
      if (child == null) {
        throw new IllegalStateException("Child node is null at index " + i);
      }

      final List<Action> childActions = child.getActions();
      if (childActions != null && childActions.size() > 0) {
        for (final Action action : childActions) {
          action.path.add("children[" + i + "]");
          actions.add(action);
        }
      }
    }

    if (actions.size() == 0) {
      return null;
    }
    return actions;
  }

  @Override
  public Object runAction(final List<String> path, final String type, final List<Object> parameters) {
    if (!this.isActivate()) {
      throw new IllegalStateException("Button is not activated");
    } else if (children == null || children.size() == 0) {
      throw new IllegalStateException("No children to run action on");
    } else if (path.size() < 1) {
      throw new IllegalArgumentException("Path must contain at least 1 element");
    }

    // Extract the index from the path
    final String indexString = path.remove(path.size() - 1);
    final Pattern pattern = Pattern.compile("children\\[(\\d+)\\]", Pattern.CASE_INSENSITIVE);
    final Matcher matcher = pattern.matcher(indexString);
    if (!matcher.find()) {
      throw new IllegalArgumentException("Invalid path: " + indexString);
    }

    final int index;
    try {
      index = Integer.parseInt(matcher.group(1));
    } catch (final NumberFormatException e) {
      throw new IllegalArgumentException("Invalid index in path: " + indexString, e);
    }

    if (index < 0 || index >= children.size()) {
      throw new IndexOutOfBoundsException("Index out of bounds: " + index);
    }

    return children.get(index).runAction(path, type, parameters);
  }

  @Override
  public String describe() {
    return "Container"; // Generic container description, could be improved if needed
  }

  /**
   * Prints the tree structure starting from this node to System.out for
   * debugging.
   */
  @Override
  public void print() {
    super.print();

    // Print children with tree prefixes
    if (children != null) {
      final String initialPrefix = ""; // Start with no prefix for direct children
      for (int i = 0; i < children.size(); i++) {
        print(children.get(i), initialPrefix, i == children.size() - 1);
      }
    }
  }

  @Override
  public List<ComponentNode> getChildren() {
    return children;
  }

  /**
   * Internal recursive helper for printing the tree structure.
   *
   * @param prefix The string prefix for the current level (e.g., "│ ").
   * @param isTail True if this is the last node in the parent's list.
   */
  protected static void print(final ComponentNode node, final String prefix, final boolean isTail) {
    // Use the node's own describe() method here
    final String suffix = " (name=" + node.getName() + ", class=" + node.getType() + ")";
    System.out.println(prefix + (isTail ? "└── " : "├── ") + node.describe() + suffix);

    if (!(node instanceof ContainerNode)) {
      return;
    }

    // Recursive call to the internal method
    final ContainerNode containerNode = (ContainerNode) node;
    final List<ComponentNode> children = containerNode.getChildren();
    if (children != null) {
      final String nextPrefix = prefix + (isTail ? "    " : "│   ");
      for (int i = 0; i < children.size(); i++) {
        print(children.get(i), nextPrefix, i == children.size() - 1);
      }
    }
  }
}
