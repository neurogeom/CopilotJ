/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj.awt;

import java.util.ArrayList;
import java.util.List;

import com.fasterxml.jackson.annotation.JsonIgnore;

public class Action {
  public static class Builder {
    private final String type;
    private final String name;
    private final String description;
    private final List<String> path;
    private final List<ParameterSchema> parameters;

    public Builder(final String type, final String name, final String description) {
      // Basic validation
      if (type == null || name == null || description == null) {
        throw new IllegalStateException("Type, name, and description cannot be null");
      }

      this.type = type;
      this.name = name;
      this.description = description;
      this.path = new ArrayList<>();
      this.parameters = new ArrayList<>();
    }

    public Builder addStringParameter(final String name, final String description) {
      return addParameter(new StringParameterSchema(name, description, null, null, null, null));
    }

    // Add string parameter with min/max length
    public Builder addStringParameter(final String name, final String description, final Integer minLength,
        final Integer maxLength) {
      return addParameter(new StringParameterSchema(name, description, minLength, maxLength, null, null));
    }

    // Add a string parameter with a regex pattern
    public Builder addStringParameter(final String name, final String description, final String pattern) {
      return addParameter(new StringParameterSchema(name, description, null, null, pattern, null));
    }

    // Add a string parameter with enum values
    public Builder addStringParameter(final String name, final String description, final List<String> enumValues) {
      if (enumValues == null || enumValues.isEmpty()) {
        throw new IllegalArgumentException("Enum values cannot be null or empty for parameter: " + name);
      }
      return addParameter(new StringParameterSchema(name, description, null, null, null, enumValues));
    }

    // Add a boolean parameter
    public Builder addBooleanParameter(final String name, final String description) {
      return addParameter(new BooleanParameterSchema(name, description));
    }

    // Add an integer parameter
    public Builder addIntegerParameter(final String name, final String description) {
      return addParameter(new IntegerParameterSchema(name, description, null, null));
    }

    // Add an integer parameter with min/max
    public Builder addIntegerParameter(final String name, final String description, final int minimum,
        final int maximum) {
      return addParameter(new IntegerParameterSchema(name, description, minimum, maximum));
    }

    // Add a number parameter (for float/double types)
    public Builder addNumberParameter(final String name, final String description) {
      return addParameter(new NumberParameterSchema(name, description, null, null));
    }

    // Add a number parameter (for float/double types)
    public Builder addNumberParameter(final String name, final String description, final Number minimum,
        final Number maximum) {
      return addParameter(new NumberParameterSchema(name, description, minimum, maximum));
    }

    public Action build() {
      return new Action(this);
    }

    private Builder addParameter(final ParameterSchema parameter) {
      this.parameters.add(parameter);
      return this;
    }
  }

  public abstract static class ParameterSchema {
    public final String type; // Type of the parameter (e.g., "string", "integer", etc.)
    public final String name;
    public final String description; // Optional

    protected ParameterSchema(final String type, final String name, final String description) {
      if (name == null || name.isEmpty()) {
        throw new IllegalArgumentException("Parameter name cannot be null or empty.");
      }

      this.type = type;
      this.name = name;
      this.description = description; // Description can be null
    }

    /**
     * Validates the given value against this parameter's schema.
     *
     * @param value The value to validate.
     * @return A list of error messages, or null if the value is valid.
     */
    public abstract List<String> validate(Object value);
  }

  public static class StringParameterSchema extends ParameterSchema {
    public final Integer minLength; // Optional
    public final Integer maxLength; // Optional
    public final String pattern; // Optional, regex
    public final List<String> enumValues; // Optional, for enum constraints

    public StringParameterSchema(final String name, final String description, final Integer minLength,
        final Integer maxLength,
        final String pattern, final List<String> enumValues) {

      super("string", name, description);

      if (minLength != null && minLength < 0) {
        throw new IllegalArgumentException("minLength cannot be negative for parameter: " + name);
      }
      if (maxLength != null && maxLength < 0) {
        throw new IllegalArgumentException("maxLength cannot be negative for parameter: " + name);
      }
      if (minLength != null && maxLength != null && minLength > maxLength) {
        throw new IllegalArgumentException(
            "minLength cannot be greater than maxLength for parameter: " + name);
      }

      this.minLength = minLength;
      this.maxLength = maxLength;

      // Basic regex validation could be done here using Pattern.compile(pattern)
      // but for now, we'll assume it's a valid regex string if provided.
      this.pattern = pattern;

      if (enumValues != null) {
        // Reject null entries. Blank strings are legitimate widget labels (e.g. an
        // AWT Choice/List placeholder item passed via ChoiceNode/ListNode.getActions()),
        // so they must NOT be rejected here.
        for (final String value : enumValues) {
          if (value == null) {
            throw new IllegalArgumentException("Enum values cannot contain null for parameter: " + name);
          }
        }
      }
      this.enumValues = enumValues;
    }

    @Override
    public List<String> validate(final Object value) {
      final List<String> errors = new ArrayList<>();
      if (!(value instanceof String)) {
        if (value == null) {
          errors.add("Value cannot be null. Expected string.");
        } else {
          errors.add("Invalid type. Expected string, got " + value.getClass().getSimpleName());
        }
        return errors; // Type check failed, further checks are likely irrelevant
      }

      final String strValue = (String) value;

      if (minLength != null && strValue.length() < minLength) {
        errors.add("String length " + strValue.length() + " is less than minimum " + minLength);
      }
      if (maxLength != null && strValue.length() > maxLength) {
        errors.add("String length " + strValue.length() + " is greater than maximum " + maxLength);
      }
      if (pattern != null && !strValue.matches(pattern)) {
        errors.add("String does not match pattern: " + pattern);
      }

      if (enumValues != null && !enumValues.contains(strValue)) {
        errors.add("Value '" + strValue + "' is not in the allowed enum values: " + String.join(", ", enumValues));
      }

      return errors.size() > 0 ? errors : null; // Return null if no errors
    }
  }

  public static class IntegerParameterSchema extends ParameterSchema {
    public final Integer minimum; // Optional
    public final Integer maximum; // Optional

    public IntegerParameterSchema(final String name, final String description, final Integer minimum,
        final Integer maximum) {
      super("integer", name, description);
      if (minimum != null && maximum != null && minimum > maximum) {
        throw new IllegalArgumentException("Minimum cannot be greater than maximum for parameter: " + name);
      }
      this.minimum = minimum;
      this.maximum = maximum;
    }

    @Override
    public List<String> validate(final Object value) {
      final List<String> errors = new ArrayList<>();
      if (!(value instanceof Integer)) {
        if (value == null) {
          errors.add("Value cannot be null. Expected integer.");
        } else {
          errors.add("Invalid type. Expected integer, got " + value.getClass().getSimpleName());
        }
        return errors;
      }

      final Integer intValue = (Integer) value;

      if (minimum != null && intValue < minimum) {
        errors.add("Value " + intValue + " is less than minimum " + minimum);
      }
      if (maximum != null && intValue > maximum) {
        errors.add("Value " + intValue + " is greater than maximum " + maximum);
      }
      return errors.size() > 0 ? errors : null; // Return null if no errors
    }
  }

  public static class NumberParameterSchema extends ParameterSchema {
    public final Number minimum; // Optional
    public final Number maximum; // Optional

    public NumberParameterSchema(final String name, final String description, final Number minimum,
        final Number maximum) {
      super("number", name, description);
      if (minimum != null && maximum != null && minimum.doubleValue() > maximum.doubleValue()) {
        throw new IllegalArgumentException("Minimum cannot be greater than maximum for parameter: " + name);
      }
      this.minimum = minimum;
      this.maximum = maximum;
    }

    @Override
    public List<String> validate(final Object value) {
      final List<String> errors = new ArrayList<>();
      if (!(value instanceof Number)) {
        if (value == null) {
          errors.add("Value cannot be null. Expected number.");
        } else {
          errors.add("Invalid type. Expected number, got " + value.getClass().getSimpleName());
        }
        return errors;
      }

      final Number numValue = (Number) value;

      if (minimum != null && numValue.doubleValue() < minimum.doubleValue()) {
        errors.add("Value " + numValue + " is less than minimum " + minimum);
      }
      if (maximum != null && numValue.doubleValue() > maximum.doubleValue()) {
        errors.add("Value " + numValue + " is greater than maximum " + maximum);
      }
      return errors.size() > 0 ? errors : null; // Return null if no errors
    }
  }

  public static class BooleanParameterSchema extends ParameterSchema {
    public BooleanParameterSchema(final String name, final String description) {
      super("boolean", name, description);
    }

    @Override
    public List<String> validate(final Object value) {
      final List<String> errors = new ArrayList<>();
      if (!(value instanceof Boolean)) {
        if (value == null) {
          errors.add("Value cannot be null. Expected boolean.");
        } else {
          errors.add("Invalid type. Expected boolean, got " + value.getClass().getSimpleName());
        }
      }
      return errors.isEmpty() ? null : errors;
    }
  }

  public static class Response {
    public final String type;
    public final Object result;

    public Response(final String type, final Object result) {
      this.type = type;
      this.result = result;
    }
  }

  public static Builder builder(final String type, final String name, final String description) {
    return new Builder(type, name, description);
  }

  public final String type; // identifies the action
  public final String name;
  public String description;
  public final List<ParameterSchema> parameters;

  @JsonIgnore
  public final List<String> path;

  private Action(final Builder builder) {
    this.type = builder.type;
    this.path = builder.path;
    this.name = builder.name;
    this.description = builder.description;
    this.parameters = builder.parameters;
  }

  public void addPath(final String path) {
    this.path.add(path);
  }
}
