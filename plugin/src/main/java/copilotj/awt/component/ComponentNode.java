/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj.awt.component;

import java.util.List;

import com.fasterxml.jackson.annotation.JsonIgnore;

import copilotj.awt.Action;

/**
 * @see https://docs.oracle.com/javase/8/docs/api/java/awt/Component.html
 */
public interface ComponentNode {

  public interface WithLabel extends ComponentNode {
    public String getLabel();
  }

  public String getName();

  public String getType();

  @JsonIgnore
  public boolean isActivate();

  public void deactivate();

  @JsonIgnore
  public List<Action> getActions();

  public Object runAction(final List<String> path, final String type, final List<Object> parameters);

  /**
   * Returns a concise, single-line description of the component node.
   * 
   * @return A string description.
   */
  public abstract String describe();

  /**
   * Prints the tree structure starting from this node to System.out for
   * debugging.
   */
  public void print();
}
