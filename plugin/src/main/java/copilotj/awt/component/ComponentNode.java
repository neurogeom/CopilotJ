/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj.awt.component;

import java.util.List;

import com.fasterxml.jackson.annotation.JsonIgnore;

import copilotj.awt.ComponentIdentifier;

/**
 * @see https://docs.oracle.com/javase/8/docs/api/java/awt/Component.html
 */
public interface ComponentNode {

  public interface WithLabel extends ComponentNode {
    public String getLabel();
  }

  public String getName();

  public String getType();

  /**
   * Returns the playwright-mcp-style ref handle ({@code "e" + int}) assigned to
   * this component, or {@code null} if this node is not ref-eligible (e.g.
   * labels and intermediate containers).
   *
   * @return the ref handle, or {@code null}.
   */
  public String getRef();

  /**
   * Returns whether this node should receive a ref handle. Ref-eligible nodes
   * are windows and actionable leaf components; descriptive-only nodes (labels)
   * and intermediate non-window containers opt out.
   *
   * @return {@code true} if this node should be assigned a ref.
   */
  @JsonIgnore
  public boolean isRefEligible();

  /**
   * Assigns (or re-affirms) the stable ref for this component using the
   * session-scoped {@link ComponentIdentifier}. No-op for non-ref-eligible
   * nodes.
   *
   * @param identifier the session-scoped component identifier.
   */
  public void assignRef(final ComponentIdentifier identifier);

  @JsonIgnore
  public boolean isActivate();

  public void deactivate();

  /**
   * Runs an action on this node, addressed by its short id (e.g. {@code click},
   * {@code selectItem}). The node owns the dispatch for its own action set; the
   * caller resolves a ref to this node first, so no path navigation is needed.
   *
   * @param action the short action id (see {@code Action.shortId()}).
   * @param parameters the action parameters, or {@code null}.
   * @return the action result (may be {@code null}).
   */
  public default Object runAction(final String action, final List<Object> parameters) {
    throw new UnsupportedOperationException("Action not supported: " + action);
  }

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
