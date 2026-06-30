/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj.util;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertNull;

import org.junit.jupiter.api.Test;

/**
 * Unit tests for {@link FromTo}, a minimal generic value pair with no validation
 * and no defined {@code equals}.
 */
public class FromToTest {

  @Test
  public void holdsFromAndToValues() {
    final FromTo<Integer> ft = new FromTo<>(1, 2);
    assertEquals(1, ft.from);
    assertEquals(2, ft.to);
  }

  @Test
  public void acceptsNullOnBothEnds() {
    final FromTo<String> ft = new FromTo<>(null, null);
    assertNull(ft.from);
    assertNull(ft.to);
  }

  @Test
  public void acceptsNullOnOneEnd() {
    final FromTo<String> ft = new FromTo<>(null, "x");
    assertNull(ft.from);
    assertEquals("x", ft.to);
  }

  @Test
  public void worksAsStringPair() {
    final FromTo<String> ft = new FromTo<>("a", "b");
    assertEquals("a", ft.from);
    assertEquals("b", ft.to);
  }

  @Test
  public void distinctInstancesAreNotEqualByReference() {
    // FromTo defines no equals; two equal-content instances remain distinct references.
    final FromTo<Integer> a = new FromTo<>(1, 2);
    final FromTo<Integer> b = new FromTo<>(1, 2);
    assertNotSame(a, b);
  }
}
