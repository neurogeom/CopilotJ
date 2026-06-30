/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj.awt;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.Arrays;
import java.util.List;

import org.junit.jupiter.api.Test;

/**
 * Constructor-invariant tests for the {@link Action.ParameterSchema} hierarchy.
 * Schemas are constructed directly (not via the Builder) to exercise the
 * constructors' own validation.
 */
public class ParameterSchemaTest {

  // --- base ParameterSchema name validation ---

  @Test
  public void baseRejectsNullName() {
    assertThrows(IllegalArgumentException.class, () -> new Action.BooleanParameterSchema(null, "d"));
  }

  @Test
  public void baseRejectsEmptyName() {
    assertThrows(IllegalArgumentException.class, () -> new Action.BooleanParameterSchema("", "d"));
  }

  // --- StringParameterSchema bounds ---

  @Test
  public void stringRejectsNegativeMinLength() {
    assertThrows(IllegalArgumentException.class,
        () -> new Action.StringParameterSchema("p", "d", -1, null, null, null));
  }

  @Test
  public void stringRejectsNegativeMaxLength() {
    assertThrows(IllegalArgumentException.class,
        () -> new Action.StringParameterSchema("p", "d", null, -5, null, null));
  }

  @Test
  public void stringRejectsInvertedMinMax() {
    assertThrows(IllegalArgumentException.class,
        () -> new Action.StringParameterSchema("p", "d", 5, 2, null, null));
  }

  @Test
  public void stringAcceptsNullOptionals() {
    assertNotNull(new Action.StringParameterSchema("p", "d", null, null, null, null));
  }

  // Regression guard: enum-list null entries are rejected at construction (the guard was
  // previously inverted dead code). Blank strings are ACCEPTED because they are legitimate
  // widget labels (AWT Choice/List placeholder items) — see ChoiceNode/ListNode.getActions().

  @Test
  public void stringRejectsEnumListWithNullEntry() {
    final List<String> bad = Arrays.asList("ok", null);
    assertThrows(IllegalArgumentException.class,
        () -> new Action.StringParameterSchema("p", "d", null, null, null, bad));
  }

  @Test
  public void stringAcceptsEnumListWithBlankEntry() {
    // Blank entries must be accepted: ChoiceNode/ListNode pass AWT item lists here and a
    // blank placeholder item is valid. Rejecting it would crash getActions()/snapshot generation.
    final List<String> vals = Arrays.asList("ok", "");
    assertNotNull(new Action.StringParameterSchema("p", "d", null, null, null, vals));
  }

  // --- IntegerParameterSchema ---

  @Test
  public void integerRejectsInvertedMinMax() {
    assertThrows(IllegalArgumentException.class, () -> new Action.IntegerParameterSchema("p", "d", 10, 1));
  }

  @Test
  public void integerAcceptsNullBounds() {
    assertNotNull(new Action.IntegerParameterSchema("p", "d", null, null));
  }

  // --- NumberParameterSchema ---

  @Test
  public void numberRejectsInvertedMinMax() {
    assertThrows(IllegalArgumentException.class, () -> new Action.NumberParameterSchema("p", "d", 10.0, 1.0));
  }

  @Test
  public void numberAcceptsNullBounds() {
    assertNotNull(new Action.NumberParameterSchema("p", "d", null, null));
  }
}
