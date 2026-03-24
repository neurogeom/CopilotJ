/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj.awt.container;

import java.awt.Component;
import java.awt.Container;

import copilotj.awt.component.ComponentNode;
import copilotj.awt.component.ComponentNodeProvider;

/**
 * Generic node for container types like Panel, Window, Frame, Dialog etc.
 *
 * @see https://docs.oracle.com/javase/8/docs/api/java/awt/Container.html
 */
public class UnknownContainerNode extends AbstractContainerNode<Container> {
  public static class Provider implements ComponentNodeProvider {
    @Override
    public ComponentNode tryCreate(final Component component) {
      if (component instanceof Container) {
        return new UnknownContainerNode((Container) component);
      }
      return null;
    }
  }

  public UnknownContainerNode(final Container component) {
    super(component.getClass().getName(), component);
  }
}
