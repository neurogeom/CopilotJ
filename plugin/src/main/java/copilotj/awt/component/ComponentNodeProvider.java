/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj.awt.component;

import java.awt.Component;
import java.util.ServiceLoader;

/**
 * Service Provider Interface for creating ComponentNode instances.
 * Implementations of this interface are discovered using
 * java.util.ServiceLoader.
 */
public interface ComponentNodeProvider {
  /**
   * Builds a tree structure representing the component hierarchy under a given
   * AWT component.
   *
   * @param component The root AWT component to start building the tree from.
   * @return The root ComponentNode of the generated tree, or null if the input
   *         component is null.
   */
  public static ComponentNode create(final Component component) {
    if (component == null) {
      return null;
    }

    // Use ServiceLoader to find and try providers
    final ServiceLoader<ComponentNodeProvider> loader = ServiceLoader.load(ComponentNodeProvider.class);
    for (final ComponentNodeProvider provider : loader) {
      final ComponentNode node = provider.tryCreate(component);
      if (node != null) {
        // Provider successfully created a node
        return node;
      }
    }

    // Fallback for any other non-container component type
    return new UnknownNode(component);
  }

  /**
   * Attempts to create a ComponentNode for the given AWT component.
   * 
   * @param component The AWT component.
   * @return A specific ComponentNode instance if this provider handles the
   *         component type, otherwise null.
   */
  ComponentNode tryCreate(Component component);
}
