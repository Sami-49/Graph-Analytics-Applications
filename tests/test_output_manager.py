import json
import re
from pathlib import Path

import pytest

from src.gfn.output_manager import OutputManager, make_run_id


def test_make_run_id_format():
    rid = make_run_id({"a": 1, "b": 2})
    assert re.match(r"^\d{8}_\d{6}_[0-9a-f]{8}$", rid)


def test_output_manager_creates_layout(tmp_path: Path):
    om = OutputManager(tmp_path, {"seed": 42}, run_id="20260101_010203_deadbeef")
    paths = om.create_layout()

    assert paths.run_root.exists()
    assert paths.logs_dir.exists()
    assert paths.tables_dir.exists()
    assert paths.figures_dir.exists()
    assert paths.reports_dir.exists()


def test_manifest_written_and_latest_index(tmp_path: Path):
    om = OutputManager(tmp_path, {"seed": 42}, run_id="20260101_010203_deadbeef")
    om.create_layout()

    om.write_latest_pointer()
    assert (tmp_path / "LATEST").read_text(encoding="utf-8").strip() == "20260101_010203_deadbeef"

    om.write_manifest({"hello": "world"})
    data = json.loads((om.run_root / "manifest.json").read_text(encoding="utf-8"))
    assert data["run_id"] == "20260101_010203_deadbeef"
    assert data["hello"] == "world"

    om.update_global_index()
    idx = (tmp_path / "index.md").read_text(encoding="utf-8")
    assert "20260101_010203_deadbeef" in idx
