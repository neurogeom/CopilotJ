/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj.awt;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import java.util.List;

import org.junit.jupiter.api.Test;

/**
 * Tests for {@link Action.ParameterSchema#validate(Object)} across all four
 * subclasses. Contract: a valid value returns {@code null}; an invalid value
 * returns a non-empty error list (never an empty list).
 */
public class ParameterSchemaValidateTest {

  private static void assertInvalid(final List<String> errors, final String containsFragment) {
    assertNotNull(errors, "invalid value must return a non-null error list");
    assertFalse(errors.isEmpty(), "invalid value must return a non-empty error list");
    if (containsFragment != null) {
      assertTrue(
          errors.stream().anyMatch(e -> e.contains(containsFragment)),
          "expected an error containing \"" + containsFragment + "\", got " + errors);
    }
  }

  // ===== String =====

  @Test
  public void stringValidReturnsNull() {
    assertNull(new Action.StringParameterSchema("p", "d", null, null, null, null).validate("ok"));
  }

  @Test
  public void stringNullReturnsTypeError() {
    assertInvalid(new Action.StringParameterSchema("p", "d", null, null, null, null).validate(null), "cannot be null");
  }

  @Test
  public void stringWrongTypeReportsSimpleClassName() {
    final List<String> errors =
        new Action.StringParameterSchema("p", "d", null, null, null, null).validate(Integer.valueOf(5));
    assertInvalid(errors, "got Integer");
    assertEquals(1, errors.size(), "type mismatch returns early with a single error");
  }

  @Test
  public void stringTooShort() {
    assertInvalid(new Action.StringParameterSchema("p", "d", 3, null, null, null).validate("ab"), "less than minimum");
  }

  @Test
  public void stringTooLong() {
    assertInvalid(new Action.StringParameterSchema("p", "d", null, 3, null, null).validate("abcd"), "greater than maximum");
  }

  @Test
  public void stringPatternMismatch() {
    assertInvalid(new Action.StringParameterSchema("p", "d", null, null, "[0-9]+", null).validate("abc"),
        "does not match pattern");
  }

  @Test
  public void stringNotInEnum() {
    assertInvalid(new Action.StringParameterSchema("p", "d", null, null, null, Arrays.asList("a", "b")).validate("c"),
        "not in the allowed enum");
  }

  @Test
  public void stringEnumBoundaryValid() {
    assertNull(new Action.StringParameterSchema("p", "d", null, null, null, Arrays.asList("a", "b")).validate("a"));
  }

  // ===== Integer =====

  @Test
  public void integerValidReturnsNull() {
    assertNull(new Action.IntegerParameterSchema("p", "d", null, null).validate(Integer.valueOf(3)));
  }

  @Test
  public void integerNullReturnsTypeError() {
    assertInvalid(new Action.IntegerParameterSchema("p", "d", null, null).validate(null), "cannot be null");
  }

  @Test
  public void integerRejectsLong() {
    // Only java.lang.Integer is accepted; a Long reports a type error.
    final List<String> errors =
        new Action.IntegerParameterSchema("p", "d", null, null).validate(Long.valueOf(3L));
    assertInvalid(errors, "got Long");
  }

  @Test
  public void integerBelowMinimum() {
    assertInvalid(new Action.IntegerParameterSchema("p", "d", 0, null).validate(Integer.valueOf(-1)), "less than minimum");
  }

  @Test
  public void integerAboveMaximum() {
    assertInvalid(new Action.IntegerParameterSchema("p", "d", null, 10).validate(Integer.valueOf(11)), "greater than maximum");
  }

  // ===== Number =====

  @Test
  public void numberAcceptsInteger() {
    assertNull(new Action.NumberParameterSchema("p", "d", null, null).validate(Integer.valueOf(3)));
  }

  @Test
  public void numberAcceptsDouble() {
    assertNull(new Action.NumberParameterSchema("p", "d", null, null).validate(Double.valueOf(3.5)));
  }

  @Test
  public void numberNullReturnsTypeError() {
    assertInvalid(new Action.NumberParameterSchema("p", "d", null, null).validate(null), "cannot be null");
  }

  @Test
  public void numberBelowMinimum() {
    assertInvalid(new Action.NumberParameterSchema("p", "d", 0.0, null).validate(Double.valueOf(-1.0)), "less than minimum");
  }

  @Test
  public void numberAboveMaximum() {
    assertInvalid(new Action.NumberParameterSchema("p", "d", null, 1.0).validate(Double.valueOf(2.5)), "greater than maximum");
  }

  // ===== Boolean (regression: valid -> null, NOT an empty list) =====

  @Test
  public void booleanValidTrueReturnsNull() {
    assertNull(new Action.BooleanParameterSchema("p", "d").validate(Boolean.TRUE));
  }

  @Test
  public void booleanValidFalseReturnsNull() {
    assertNull(new Action.BooleanParameterSchema("p", "d").validate(Boolean.FALSE));
  }

  @Test
  public void booleanNullReturnsTypeError() {
    assertInvalid(new Action.BooleanParameterSchema("p", "d").validate(null), "cannot be null");
  }

  @Test
  public void booleanWrongTypeReportsSimpleClassName() {
    assertInvalid(new Action.BooleanParameterSchema("p", "d").validate("yes"), "got String");
  }
}
