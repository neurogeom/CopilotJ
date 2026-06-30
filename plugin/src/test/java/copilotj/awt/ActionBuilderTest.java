/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj.awt;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.junit.jupiter.api.Test;

import copilotj.awt.Action.Builder;

/**
 * Unit tests for {@link Action.Builder} construction and {@link Action} assembly.
 */
public class ActionBuilderTest {

  @Test
  public void builderRejectsNullType() {
    assertThrows(IllegalStateException.class, () -> Action.builder(null, "n", "d"));
  }

  @Test
  public void builderRejectsNullName() {
    assertThrows(IllegalStateException.class, () -> Action.builder("t", null, "d"));
  }

  @Test
  public void builderRejectsNullDescription() {
    assertThrows(IllegalStateException.class, () -> Action.builder("t", "n", null));
  }

  @Test
  public void builderAcceptsAllNonNull() {
    assertNotNull(Action.builder("t", "n", "d"));
  }

  @Test
  public void buildCopiesFieldsAndStartsEmpty() {
    final Action a = Action.builder("t", "n", "d").build();
    assertEquals("t", a.type);
    assertEquals("n", a.name);
    assertEquals("d", a.description);
    assertTrue(a.parameters.isEmpty());
    assertTrue(a.path.isEmpty());
  }

  @Test
  public void addStringParameterEnumOverloadRejectsNull() {
    final Builder b = Action.builder("t", "n", "d");
    assertThrows(IllegalArgumentException.class, () -> b.addStringParameter("p", "d", (List<String>) null));
  }

  @Test
  public void addStringParameterEnumOverloadRejectsEmpty() {
    final Builder b = Action.builder("t", "n", "d");
    assertThrows(IllegalArgumentException.class, () -> b.addStringParameter("p", "d", Collections.emptyList()));
  }

  @Test
  public void addPathAppendsInPlace() {
    final Action a = Action.builder("t", "n", "d").build();
    a.addPath("x");
    a.addPath("y");
    assertEquals(Arrays.asList("x", "y"), a.path);
  }

  @Test
  public void descriptionIsMutableOnBuiltAction() {
    // description is the only non-final Action field.
    final Action a = Action.builder("t", "n", "d").build();
    a.description = "new";
    assertEquals("new", a.description);
  }

  @Test
  public void chainingReturnsSelfAndBuilds() {
    final Action a = Action.builder("t", "n", "d")
        .addStringParameter("s", "desc")
        .addBooleanParameter("b", "desc")
        .addIntegerParameter("i", "desc")
        .addNumberParameter("n2", "desc")
        .build();
    assertEquals(4, a.parameters.size());
  }
}
