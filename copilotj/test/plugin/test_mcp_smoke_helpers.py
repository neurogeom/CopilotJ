# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the snapshot-parsing helpers in ``scripts/mcp_smoke_test.py``.

These run under ``just test`` (no live Fiji needed) so the JSON-walking logic —
the part most likely to silently break when the snapshot shape changes — stays
CI-covered. The helpers are loaded straight from the script via ``importlib``.
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


def _load_smoke_module() -> ModuleType:
    """Import scripts/mcp_smoke_test.py as a module (its ``__main__`` guard is no-op)."""
    root = Path(__file__).resolve()
    while root != root.parent:
        if (root / "scripts" / "mcp_smoke_test.py").exists():
            break
        root = root.parent
    spec = importlib.util.spec_from_file_location("mcp_smoke_test", root / "scripts" / "mcp_smoke_test.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so the @dataclass decorator can resolve the module.
    sys.modules["mcp_smoke_test"] = mod
    spec.loader.exec_module(mod)
    return mod


_smoke = _load_smoke_module()
_walk_components = _smoke._walk_components
find_action_ref = _smoke.find_action_ref
checkbox_state_by_ref = _smoke.checkbox_state_by_ref


def _snapshot_text() -> str:
    """A minimal take_snapshot result mirroring the ref-model JSON shape."""
    snap = {
        "id": 1,
        "timestamp": "2026-06-23T00:00:00",
        "current_image": None,
        "windows": [
            {
                "type": "ij.GenericDialog",
                "id": 1,
                "ref": "e1",
                "name": "ROI Manager",
                "children": [
                    {
                        "type": "java.awt.Checkbox",
                        "name": None,
                        "ref": "e2",
                        "label": "Show All",
                        "state": False,
                        "actions": [
                            {
                                "type": "java.awt.Checkbox.setState",
                                "name": "Set State",
                                "description": "d",
                                "parameters": [],
                            }
                        ],
                    },
                    {"type": "java.awt.Label", "name": None, "ref": None, "text": "Count: 0"},
                    {
                        "type": "java.awt.Button",
                        "name": "add",
                        "ref": "e3",
                        "label": "Add",
                        "actions": [
                            {"type": "java.awt.Button.click", "name": "Click", "description": "d", "parameters": []}
                        ],
                    },
                ],
            }
        ],
        "screen_width": 1920,
        "screen_height": 1080,
        "gui_scale": "1.00",
    }
    return json.dumps(snap)


def test_walk_components_yields_every_node_with_window_context():
    nodes = list(_walk_components(_snapshot_text()))
    # window, checkbox, label, button — label included even though it has no ref.
    assert [n.get("ref") for n, _ in nodes] == ["e1", "e2", None, "e3"]
    window_node = nodes[0][0]
    # Every node shares the same containing window.
    assert all(w is nodes[0][1] for _, w in nodes)
    assert (window_node.get("name")) == "ROI Manager"


def test_find_action_ref_by_label_is_deterministic():
    ref, action = find_action_ref(_snapshot_text(), ".setState", label="Show All")
    assert (ref, action) == ("e2", "setState")


def test_find_action_ref_unscoped_returns_first_match():
    assert find_action_ref(_snapshot_text(), ".setState") == ("e2", "setState")
    assert find_action_ref(_snapshot_text(), ".click") == ("e3", "click")


def test_find_action_ref_label_miss_returns_none():
    assert find_action_ref(_snapshot_text(), ".setState", label="Nonexistent") == (None, None)


def test_find_action_ref_window_title_scopes_via_name_field():
    # The window carries its title in `name`; window_title matches it as a substring.
    assert find_action_ref(_snapshot_text(), ".setState", window_title="ROI Manager") == ("e2", "setState")
    assert find_action_ref(_snapshot_text(), ".setState", window_title="Other Window") == (None, None)


def test_checkbox_state_by_ref():
    snap = _snapshot_text()
    assert checkbox_state_by_ref(snap, "e2") is False
    assert checkbox_state_by_ref(snap, "e999") is None  # unknown ref
    assert checkbox_state_by_ref(snap, "e3") is None  # button has no `state` field


def test_helpers_are_robust_to_bad_input():
    assert find_action_ref("", ".setState") == (None, None)
    assert find_action_ref("not valid json", ".setState") == (None, None)
    assert list(_walk_components("")) == []
    assert list(_walk_components("not valid json")) == []
    assert checkbox_state_by_ref("", "e2") is None
