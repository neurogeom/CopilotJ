/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj.awt.container;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.awt.Component;
import java.awt.Container;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.junit.jupiter.api.Test;

import copilotj.awt.Action;
import copilotj.awt.component.ComponentNode;

/**
 * Tests {@link AbstractContainerNode}'s action aggregation and path routing.
 *
 * <p>The node is built from a Mockito {@link Container} (NOT a real
 * {@code java.awt.Panel}/{@code Button}) and its children are injected mock
 * {@link ComponentNode}s. This is required because real AWT component
 * construction throws {@link java.awt.HeadlessException} on headless Linux CI
 * ("no headful library support was found"), even though it works on a mac with a
 * full AWT toolkit. {@code getActions()} has no activation gate and is exercised
 * directly; {@code runAction} is reached via a subclass overriding
 * {@code isActivate()} to true.
 */
public class AbstractContainerNodeTest {

  /** A mock Container whose getComponents() won't NPE the constructor's child loop. */
  private static Container mockContainer() {
    final Container c = mock(Container.class);
    when(c.getComponents()).thenReturn(new Component[0]);
    return c;
  }

  private static void setChildren(final AbstractContainerNode<?> node, final List<ComponentNode> children) {
    try {
      final Field f = AbstractContainerNode.class.getField("children"); // public final
      f.setAccessible(true);
      f.set(node, children);
    } catch (final ReflectiveOperationException e) {
      throw new AssertionError("cannot set children field", e);
    }
  }

  private static AbstractContainerNode<Container> node(final List<ComponentNode> children) {
    final AbstractContainerNode<Container> node = new UnknownContainerNode(mockContainer());
    setChildren(node, children);
    return node;
  }

  /** A node that reports itself activated, bypassing the headless isShowing() gate. */
  private static AbstractContainerNode<Container> activated(final List<ComponentNode> children) {
    final AbstractContainerNode<Container> node = new UnknownContainerNode(mockContainer()) {
      @Override
      public boolean isActivate() {
        return true;
      }
    };
    setChildren(node, children);
    return node;
  }

  private static ComponentNode childWithClickAction() {
    final ComponentNode child = mock(ComponentNode.class);
    when(child.getActions()).thenReturn(Collections.singletonList(Action.builder("t.click", "Click", "d").build()));
    return child;
  }

  private static String last(final List<String> path) {
    return path.get(path.size() - 1);
  }

  // ===== getActions() =====

  @Test
  public void getActionsNullForChildlessContainer() {
    assertNull(new UnknownContainerNode(mockContainer()).getActions(), "no children -> null");
  }

  @Test
  public void getActionsPrefixesEachChildPath() {
    final AbstractContainerNode<Container> n = node(Arrays.asList(childWithClickAction(), childWithClickAction()));
    final List<Action> actions = n.getActions();
    assertEquals(2, actions.size(), "one aggregated action per child");
    assertEquals("children[0]", last(actions.get(0).path));
    assertEquals("children[1]", last(actions.get(1).path));
  }

  // ===== runAction() =====

  @Test
  public void runActionThrowsWhenNotActivated() {
    // node() does not override isActivate() -> permanently deactivated (mock isShowing()=false).
    final AbstractContainerNode<Container> n = node(Arrays.asList(childWithClickAction()));
    final IllegalStateException ex = assertThrows(IllegalStateException.class,
        () -> n.runAction(new ArrayList<>(Arrays.asList("children[0]")), "x", null));
    assertTrue(ex.getMessage().contains("not activated"));
  }

  @Test
  public void runActionThrowsOnEmptyPath() {
    final AbstractContainerNode<Container> n = activated(Arrays.asList(childWithClickAction()));
    final IllegalArgumentException ex = assertThrows(IllegalArgumentException.class,
        () -> n.runAction(new ArrayList<>(), "x", null));
    assertTrue(ex.getMessage().contains("at least 1 element"));
  }

  @Test
  public void runActionThrowsOnMalformedPath() {
    final AbstractContainerNode<Container> n = activated(Arrays.asList(childWithClickAction()));
    final IllegalArgumentException ex = assertThrows(IllegalArgumentException.class,
        () -> n.runAction(new ArrayList<>(Arrays.asList("foo")), "x", null));
    assertTrue(ex.getMessage().contains("Invalid path"));
  }

  @Test
  public void runActionThrowsOnOutOfRangeIndex() {
    final AbstractContainerNode<Container> n = activated(Arrays.asList(childWithClickAction(), childWithClickAction()));
    assertThrows(IndexOutOfBoundsException.class,
        () -> n.runAction(new ArrayList<>(Arrays.asList("children[99]")), "x", null));
  }

  @Test
  public void runActionDelegatesToIndexedChild() {
    // Distinguishable children: children[0] vs children[1] return different sentinels,
    // so this proves the index is parsed and routed correctly (not just "some child answered").
    final ComponentNode child0 = mock(ComponentNode.class);
    when(child0.runAction(any(), any(), any())).thenReturn("from-0");
    final ComponentNode child1 = mock(ComponentNode.class);
    when(child1.runAction(any(), any(), any())).thenReturn("from-1");
    final AbstractContainerNode<Container> n = activated(Arrays.asList(child0, child1));

    assertEquals("from-0", n.runAction(new ArrayList<>(Arrays.asList("children[0]")), "t", null));
    assertEquals("from-1", n.runAction(new ArrayList<>(Arrays.asList("children[1]")), "t", null));
  }
}
