/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj.awt.component;

import java.awt.Component;
import java.awt.Choice;
import java.awt.event.ItemEvent;
import java.awt.event.ItemListener;
import java.util.Collections;
import java.util.List;
import java.util.Arrays;

import copilotj.awt.Action;

/**
 * @see https://docs.oracle.com/javase/8/docs/api/java/awt/Choice.html
 */
public class ChoiceNode extends AbstractComponentNode<Choice> {
  public static class Provider implements ComponentNodeProvider {
    @Override
    public ComponentNode tryCreate(final Component component) {
      if (component instanceof Choice) {
        return new ChoiceNode((Choice) component);
      }
      return null;
    }
  }

  private static final String TYPE = "java.awt.Choice";

  public final String[] items;
  public final String selectedItem;

  public ChoiceNode(final Choice component) {
    super(TYPE, component);
    this.selectedItem = component.getSelectedItem();
    // Get all items for context
    this.items = new String[component.getItemCount()];
    for (int i = 0; i < component.getItemCount(); i++) {
      this.items[i] = component.getItem(i);
    }
  }

  @Override
  public String describe() {
    // Keep items description short for single line
    final String itemsStr = (items != null && items.length > 3)
        ? "[" + items[0] + ", " + items[1] + ", ..., " + items[items.length - 1] + "]"
        : Arrays.toString(items);
    return "Choice: selected=" + selectedItem + ", items=" + itemsStr;
  }

  @Override
  public List<Action> getActions() {
    final Action selectItemAction = Action
        .builder(TYPE + ".selectItem", "Select Item", "Selects an item in the choice menu.")
        .addStringParameter("item", "The item to select", Arrays.asList(this.items))
        .build();
    return Collections.singletonList(selectItemAction);
  }

  @Override
  public Object runAction(final List<String> path, final String type, final List<Object> parameters) {
    if (!this.isActivate()) {
      throw new IllegalStateException("Choice is not activated");
    } else if (path.size() != 0) {
      throw new IllegalArgumentException("Path must be empty for ChoiceNode actions like 'selectItem'");
    }

    switch (type) {
      case TYPE + ".selectItem":
        if (parameters == null || parameters.size() != 1) {
          throw new IllegalArgumentException(
              "Action 'selectItem' requires exactly one string 'item' parameter. Found: " +
                  (parameters == null ? 0 : parameters.size()) + " parameters.");
        }

        final Object param = parameters.get(0);
        if (!(param instanceof String)) {
          throw new IllegalArgumentException(
              "Action 'selectItem' requires a string 'item' parameter, but got " +
                  param.getClass().getSimpleName());
        }

        return selectItem((String) param);

      default:
        throw new IllegalArgumentException("Unknown action type: " + type + " for ChoiceNode");
    }
  }

  /**
   * Selects an item in the choice component and notifies listeners.
   * 
   * @param item The item to select.
   * @return null
   */
  public Object selectItem(final String item) {
    if (!component.isEnabled()) {
      throw new IllegalStateException("Choice is not enabled");
    }

    // Check if the item exists in the choice list
    boolean itemExists = false;
    for (int i = 0; i < this.component.getItemCount(); i++) {
      if (this.component.getItem(i).equals(item)) {
        itemExists = true;
        break;
      }
    }
    if (!itemExists) {
      throw new IllegalArgumentException(
          "Item '" + item + "' not found in Choice. Available items: " + Arrays.toString(this.items));
    }

    this.component.select(item);

    // Notify listeners
    // The ItemEvent needs to indicate what changed (ITEM_STATE_CHANGED)
    // and the new state (the selected item itself).
    final ItemEvent event = new ItemEvent(this.component, ItemEvent.ITEM_STATE_CHANGED, item, ItemEvent.SELECTED);

    for (final ItemListener listener : this.component.getItemListeners()) {
      listener.itemStateChanged(event);
    }

    return null;
  }
}
