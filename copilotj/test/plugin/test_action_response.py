# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the call_action response contract.

Locks in the D1 fix from the /review pass: every action response must carry its
fully-qualified action type (e.g. ``java.awt.Button.click``), and the
``TypedActionResponse`` union must accept each one — including the scrollbar and
list branches that were missing before. Short ids (``click``) must be rejected.
"""

import pytest
from pydantic import TypeAdapter, ValidationError

from copilotj.plugin.awt.action import TypedActionResponse

# TypedActionResponse is a `type` union alias, not a BaseModel — validate via TypeAdapter.
_RESPONSE = TypeAdapter(TypedActionResponse)

# Action types whose response result is None (the simple mutations). The
# image/results-table responses carry non-None results and are exercised elsewhere.
_NONE_RESULT_ACTION_TYPES = [
    "java.awt.Button.click",
    "java.awt.Checkbox.setState",
    "java.awt.Choice.selectItem",
    "java.awt.List.select",
    "java.awt.Scrollbar.setValue",
    "java.awt.TextArea.setText",
    "java.awt.TextField.setText",
]


@pytest.mark.parametrize("full_type", _NONE_RESULT_ACTION_TYPES)
def test_typed_action_response_accepts_each_action_type(full_type):
    """Each fully-qualified action type must validate against the union."""
    resp = _RESPONSE.validate_python({"type": full_type, "result": None})
    assert resp.type == full_type


def test_short_action_id_is_rejected():
    """The contract is fully-qualified types; a bare short id must not validate."""
    with pytest.raises(ValidationError):
        _RESPONSE.validate_python({"type": "click", "result": None})
