# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from copilotj.util.prompt_template import render_prompt


# --- Basic conditionals ---


def test_renders_if_branch_when_condition_true():
    template = '{% if mode == "react" %}\nAction line\n{% endif %}'
    assert "Action line" in render_prompt(template, mode="react")


def test_skips_if_branch_when_condition_false():
    template = '{% if mode == "react" %}\nAction line\n{% endif %}'
    assert "Action line" not in render_prompt(template, mode="native")


def test_renders_else_branch_when_condition_false():
    template = '{% if mode == "react" %}\nuse Action\n{% else %}\ncall directly\n{% endif %}'
    result = render_prompt(template, mode="native")
    assert "call directly" in result
    assert "use Action" not in result


def test_skips_else_branch_when_condition_true():
    template = '{% if mode == "react" %}\nuse Action\n{% else %}\ncall directly\n{% endif %}'
    result = render_prompt(template, mode="react")
    assert "use Action" in result
    assert "call directly" not in result


# --- elif ---


def test_elif_branch_taken():
    template = '{% if x == "a" %}\nA\n{% elif x == "b" %}\nB\n{% endif %}'
    result = render_prompt(template, x="b")
    assert "B" in result
    assert "A" not in result


def test_elif_skipped_when_if_taken():
    template = '{% if x == "a" %}\nA\n{% elif x == "b" %}\nB\n{% endif %}'
    result = render_prompt(template, x="a")
    assert "A" in result
    assert "B" not in result


def test_elif_else_fallback():
    template = '{% if x == "a" %}\nA\n{% elif x == "b" %}\nB\n{% else %}\nC\n{% endif %}'
    result = render_prompt(template, x="z")
    assert "C" in result
    assert "A" not in result
    assert "B" not in result


# --- Variable substitution ---


def test_substitutes_template_variables():
    template = "Hello {name}, your mode is {mode}."
    assert render_prompt(template, name="Alice", mode="react") == "Hello Alice, your mode is react."


def test_leaves_unknown_variables_as_is():
    template = "Hello {name}, {unknown_var} stays."
    assert render_prompt(template, name="Alice") == "Hello Alice, {unknown_var} stays."


# --- Edge cases ---


def test_no_conditionals_returns_template_as_is():
    template = "Just a plain template with no conditionals."
    assert render_prompt(template) == "Just a plain template with no conditionals."


def test_empty_template():
    assert render_prompt("") == ""


def test_nested_conditionals():
    template = '{% if a == "1" %}\nouter-true\n{% if b == "2" %}\ninner-true\n{% endif %}\n{% endif %}'
    result = render_prompt(template, a="1", b="2")
    assert "outer-true" in result
    assert "inner-true" in result


def test_nested_conditionals_inner_false():
    template = '{% if a == "1" %}\nouter-true\n{% if b == "2" %}\ninner-true\n{% endif %}\n{% endif %}'
    result = render_prompt(template, a="1", b="9")
    assert "outer-true" in result
    assert "inner-true" not in result


def test_consecutive_if_blocks():
    template = '{% if x == "a" %}\nAAA\n{% endif %}\n{% if x == "b" %}\nBBB\n{% endif %}'
    assert "AAA" in render_prompt(template, x="a")
    assert "BBB" not in render_prompt(template, x="a")
    assert "BBB" in render_prompt(template, x="b")
    assert "AAA" not in render_prompt(template, x="b")


def test_multiline_content_in_branches():
    template = '{% if mode == "react" %}\nLine 1\nLine 2\n{% else %}\nAlt 1\nAlt 2\n{% endif %}'
    result = render_prompt(template, mode="react")
    assert "Line 1" in result
    assert "Line 2" in result
    assert "Alt" not in result


# --- Error cases ---


def test_unclosed_if_raises_syntax_error():
    with pytest.raises(SyntaxError, match="Unclosed"):
        render_prompt('{% if x == "1" %}\ncontent')


def test_elif_without_if_raises_syntax_error():
    with pytest.raises(SyntaxError, match="without matching"):
        render_prompt('{% elif x == "1" %}\ncontent\n{% endif %}')


def test_else_without_if_raises_syntax_error():
    with pytest.raises(SyntaxError, match="without matching"):
        render_prompt("{% else %}\ncontent\n{% endif %}")


def test_endif_without_if_raises_syntax_error():
    with pytest.raises(SyntaxError, match="without matching"):
        render_prompt("{% endif %}")


def test_elif_after_else_raises_syntax_error():
    with pytest.raises(SyntaxError, match="after"):
        render_prompt('{% if x == "1" %}\nA\n{% else %}\nB\n{% elif x == "2" %}\nC\n{% endif %}')


def test_if_without_condition_raises_syntax_error():
    with pytest.raises(SyntaxError, match="Missing condition"):
        render_prompt("{% if %}\ncontent\n{% endif %}")
