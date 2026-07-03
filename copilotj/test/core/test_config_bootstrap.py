# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from copilotj.core.config import bootstrap_dir_if_empty


def _seed(path: Path, name: str, body: str = "seed") -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / name).write_text(body, encoding="utf-8")


def test_copies_when_dst_missing(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    _seed(src, "a.toml")
    assert bootstrap_dir_if_empty(src, dst) is True
    assert (dst / "a.toml").read_text() == "seed"


def test_copies_when_dst_empty(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    _seed(src, "a.toml")
    dst.mkdir()
    assert bootstrap_dir_if_empty(src, dst) is True
    assert (dst / "a.toml").exists()


def test_noop_when_dst_populated(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    _seed(src, "a.toml", body="NEW")
    _seed(dst, "existing.toml", body="USER")
    assert bootstrap_dir_if_empty(src, dst) is False
    assert not (dst / "a.toml").exists()  # nothing copied
    assert (dst / "existing.toml").read_text() == "USER"  # user data intact


def test_noop_when_src_missing(tmp_path):
    dst = tmp_path / "dst"
    assert bootstrap_dir_if_empty(tmp_path / "nope", dst) is False
    assert not dst.exists()


def test_noop_when_src_is_dst(tmp_path):
    # Dev mode: home dir IS the source tree -> must not copy a dir onto itself.
    src = tmp_path / "shared"
    _seed(src, "a.toml")
    assert bootstrap_dir_if_empty(src, src) is False
    assert not (src / ".anything").exists()  # no marker/side effects written
