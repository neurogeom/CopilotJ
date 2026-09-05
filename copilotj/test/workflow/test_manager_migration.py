# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import importlib
from pathlib import Path

from copilotj.workflow import manager as workflow_manager
from copilotj.workflow.manager import migrate_workflow_layout


def _set_home(monkeypatch, tmp_path: Path) -> Path:
    """Point COPILOTJ_HOME at tmp_path so migrate_workflow_layout operates there."""
    monkeypatch.setenv("COPILOTJ_HOME", str(tmp_path))
    return tmp_path


def _wf(root: Path, wf_id: str, body: str = "{}") -> Path:
    f = root / "temp" / "workflows" / wf_id / "workflow.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(body, encoding="utf-8")
    return f


def test_migrate_noop_when_old_missing(tmp_path, monkeypatch):
    home = _set_home(monkeypatch, tmp_path)
    migrate_workflow_layout()  # no temp/workflows -> nothing to do
    assert not (home / "workflows").exists() or not any((home / "workflows").iterdir())


def test_migrate_moves_when_new_missing(tmp_path, monkeypatch):
    home = _set_home(monkeypatch, tmp_path)
    _wf(home, "wf1", '{"id": "wf1"}')
    migrate_workflow_layout()
    assert (home / "workflows" / "wf1" / "workflow.json").read_text() == '{"id": "wf1"}'
    assert not (home / "temp" / "workflows" / "wf1").exists()  # moved, not copied


def test_migrate_moves_children_when_new_empty(tmp_path, monkeypatch):
    home = _set_home(monkeypatch, tmp_path)
    _wf(home, "wf1")
    (home / "workflows").mkdir(parents=True)  # new exists but empty
    migrate_workflow_layout()
    assert (home / "workflows" / "wf1" / "workflow.json").exists()


def test_migrate_merges_into_populated_new(tmp_path, monkeypatch):
    """A populated new dir is preserved (no clobber) but old children are still merged in
    (no early-out that would strand data after a partial prior run)."""
    home = _set_home(monkeypatch, tmp_path)
    _wf(home, "old_wf")
    existing = home / "workflows" / "existing_wf" / "workflow.json"
    existing.parent.mkdir(parents=True)
    existing.write_text("KEEP", encoding="utf-8")
    migrate_workflow_layout()
    assert existing.read_text() == "KEEP"  # existing workflow preserved (no clobber)
    assert (home / "workflows" / "old_wf" / "workflow.json").exists()  # old_wf merged in
    assert not (home / "temp" / "workflows" / "old_wf").exists()  # ...and moved out of temp


def test_migrate_completes_after_partial_run(tmp_path, monkeypatch):
    """A previously-interrupted migration (some children moved, some stranded) completes
    on the next run instead of abandoning the stranded children."""
    home = _set_home(monkeypatch, tmp_path)
    _wf(home, "wf_a")
    _wf(home, "wf_b")
    # Simulate a partial prior migration: wf_a already in new, wf_b still stranded in temp.
    (home / "workflows" / "wf_a").mkdir(parents=True)
    (home / "workflows" / "wf_a" / "workflow.json").write_text("ALREADY", encoding="utf-8")
    migrate_workflow_layout()
    assert (home / "workflows" / "wf_b" / "workflow.json").exists()  # stranded child recovered
    assert (home / "workflows" / "wf_a" / "workflow.json").read_text() == "ALREADY"


def test_migrate_idempotent(tmp_path, monkeypatch):
    home = _set_home(monkeypatch, tmp_path)
    _wf(home, "wf1")
    migrate_workflow_layout()
    migrate_workflow_layout()  # second run: new is populated -> no-op
    assert (home / "workflows" / "wf1" / "workflow.json").exists()


def test_import_runs_migration_before_mkdir(tmp_path, monkeypatch):
    """Regression (codex #1/#2): the module-level mkdir must not pre-create the new
    dir and defeat migration. Importing the module with old data present must move it.
    """
    home = _set_home(monkeypatch, tmp_path)
    _wf(home, "wf_import", '{"id": "wf_import"}')
    try:
        importlib.reload(workflow_manager)
        # After reload, migration ran (before mkdir) and BASE_DIR points at the new home.
        assert (workflow_manager.BASE_DIR / "wf_import" / "workflow.json").exists()
        assert not (home / "temp" / "workflows" / "wf_import").exists()
    finally:
        monkeypatch.delenv("COPILOTJ_HOME", raising=False)
        importlib.reload(workflow_manager)  # restore module state to default home
