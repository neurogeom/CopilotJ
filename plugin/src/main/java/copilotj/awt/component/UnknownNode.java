/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj.awt.component;

import java.awt.Component;

// Node for components not specifically handled above
// Note: This does not have a tryCreate, it's used as a fallback.
public class UnknownNode extends AbstractComponentNode<Component> {
  public UnknownNode(final Component component) { // Keep constructor public/package-private
    super(component.getClass().getName(), component);
  }

  @Override
  public String describe() {
    return "Unknown";
  }
}
