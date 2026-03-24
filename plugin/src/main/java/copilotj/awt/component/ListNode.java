/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj.awt.component;

import java.awt.Component;
import java.awt.List;
import java.awt.event.ItemEvent;
import java.awt.event.ItemListener;
import java.util.Arrays;
import java.util.Collections;

import copilotj.awt.Action;

/**
 * AWT List
 *
 * @see https://docs.oracle.com/javase/8/docs/api/java/awt/List.html
 */
public class ListNode extends AbstractComponentNode<List> {
  public static class Provider implements ComponentNodeProvider {
    @Override
    public ComponentNode tryCreate(final Component component) {
      if (component instanceof List) {
        return new ListNode((List) component);
      }
      return null;
    }
  }

  public static final String TYPE = "java.awt.List";

  public final String[] items;
  public final String selectedItem;

  public ListNode(final List component) {
    super(TYPE, component);
    this.items = component.getItems();
    this.selectedItem = component.getSelectedItem();
  }

  @Override
  public String describe() {
    // Keep items description short for single line
    final String itemsStr = (items != null && items.length > 3)
        ? "[" + items[0] + ", " + items[1] + ", ..., " + items[items.length - 1] + "]"
        : Arrays.toString(items);
    return "List: selected=" + selectedItem + ", items=" + itemsStr;
  }

  @Override
  public java.util.List<Action> getActions() {
    if (!this.component.isEnabled()) { // AWT List doesn't have isEditable
      return null;
    }

    final Action selectAction = Action
        .builder(this.type + ".select", "Select Item", "Selects an item in the list.")
        .addStringParameter("item", "The item to select", Arrays.asList(this.items))
        .build();
    return Collections.singletonList(selectAction);
  }

  @Override
  public Object runAction(final java.util.List<String> path, final String type,
      final java.util.List<Object> parameters) {
    if (!this.isActivate()) {
      throw new IllegalStateException("List is not activated");
    } else if (path.size() != 0) {
      throw new IllegalArgumentException("Path must be empty for ListNode actions like 'select'");
    }

    switch (type) {
      case TYPE + ".select":
        if (parameters == null || parameters.size() != 1) {
          throw new IllegalArgumentException(
              "Action 'select' requires exactly one string 'item' parameter. Found: " +
                  (parameters == null ? 0 : parameters.size()) + " parameters.");
        }

        final Object param = parameters.get(0);
        if (!(param instanceof String)) {
          throw new IllegalArgumentException(
              "Action 'select' requires a string 'item' parameter, but got " +
                  param.getClass().getSimpleName());
        }
        return selectItem((String) param);

      default:
        throw new IllegalArgumentException("Unknown action type: " + type + " for ListNode");
    }
  }

  /**
   * Selects an item in the list component and notifies listeners.
   *
   * @param item The item to select.
   * @return null
   */
  public Object selectItem(final String item) {
    if (!component.isEnabled()) {
      throw new IllegalStateException("List is not enabled");
    }

    int itemIndex = -1;
    final String[] currentItems = component.getItems();
    for (int i = 0; i < currentItems.length; i++) {
      if (currentItems[i].equals(item)) {
        itemIndex = i;
        break;
      }
    }

    if (itemIndex == -1) {
      throw new IllegalArgumentException("Item not found in list: " + item);
    }

    // Store old selected index and item for the event
    final int oldSelectedIndex = component.getSelectedIndex();
    // String oldSelectedItem = component.getSelectedItem(); // Not directly used in
    // ItemEvent for deselection

    component.select(itemIndex);

    // Notify listeners
    // AWT List typically fires two ItemEvents for a selection change if an item was
    // previously selected:
    // 1. One for the deselected item (if any)
    // 2. One for the selected item
    // We will mimic this behavior.

    // Event for deselected item (if one was selected and it's different from the
    // new one)
    if (oldSelectedIndex != -1 && oldSelectedIndex != itemIndex) {
      // The ItemEvent constructor for deselection needs the item object that was
      // deselected.
      // We need to be careful if the list was modified between getting
      // oldSelectedIndex and now.
      // For simplicity, we assume the item at oldSelectedIndex is still valid for
      // deselection event.
      // A more robust way might involve getting the item string before selection.
      // However, ItemEvent takes an Object, and for AWT List, this is the String
      // item.
      // We'll use the item string from the component's current state at that index if
      // possible.
      String deselectedItemString = null;
      if (oldSelectedIndex < currentItems.length) { // Check bounds
        deselectedItemString = currentItems[oldSelectedIndex];
      }

      if (deselectedItemString != null) {
        final ItemEvent deselectEvent = new ItemEvent(
            this.component,
            ItemEvent.ITEM_STATE_CHANGED,
            deselectedItemString, // The item that was deselected
            ItemEvent.DESELECTED);
        for (final ItemListener listener : this.component.getItemListeners()) {
          listener.itemStateChanged(deselectEvent);
        }
      }
    }

    // Event for selected item
    final ItemEvent selectEvent = new ItemEvent(
        this.component,
        ItemEvent.ITEM_STATE_CHANGED,
        item, // The item that was selected
        ItemEvent.SELECTED);

    for (final ItemListener listener : this.component.getItemListeners()) {
      listener.itemStateChanged(selectEvent);
    }

    return null;
  }
}
