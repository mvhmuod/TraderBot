"""Tests for the Windows launcher run.bat (runs it against a stand-in app, no GUI)."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

pytestmark = [
    pytest.mark.skipif(sys.platform != "win32", reason="run.bat is Windows only"),
    pytest.mark.skipif(not (Path(sys.prefix) / "pyvenv.cfg").exists(),
                       reason="needs the tests to run inside a virtual environment, like .venv"),
]


def _run_launcher(tmp_path: Path, exit_code: int) -> str:
    """Run a copy of run.bat whose app.py just exits with ``exit_code``; return its output."""
    scripts = tmp_path / ".venv" / "Scripts"
    scripts.mkdir(parents=True)
    # A copy of this virtual environment's launcher (it finds Python through pyvenv.cfg).
    shutil.copy2(sys.executable, scripts / "python.exe")
    shutil.copy2(Path(sys.prefix) / "pyvenv.cfg", tmp_path / ".venv" / "pyvenv.cfg")
    (tmp_path / "app.py").write_text(f"import os\nos._exit({exit_code})\n", encoding="utf-8")
    shutil.copy2(PROJECT_ROOT / "run.bat", tmp_path / "run.bat")
    completed = subprocess.run(["cmd", "/c", str(tmp_path / "run.bat")], stdin=subprocess.DEVNULL,
                               capture_output=True, text=True, timeout=60)
    return completed.stdout + completed.stderr


def test_launcher_is_quiet_after_a_normal_exit(tmp_path: Path) -> None:
    assert "closed with an error" not in _run_launcher(tmp_path, 0)


def test_launcher_pauses_after_a_python_error(tmp_path: Path) -> None:
    assert "closed with an error" in _run_launcher(tmp_path, 1)


def test_launcher_pauses_after_a_native_crash(tmp_path: Path) -> None:
    # An access violation exits with 0xC0000005, which cmd sees as a NEGATIVE errorlevel,
    # so "if errorlevel 1" missed it. (os._exit needs the signed 32-bit value.)
    assert "closed with an error" in _run_launcher(tmp_path, 0xC0000005 - (1 << 32))
