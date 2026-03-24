/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj.awt.container;

import java.util.List;

import copilotj.awt.component.ComponentNode;

public interface ContainerNode extends ComponentNode {
  public List<ComponentNode> getChildren();
}
