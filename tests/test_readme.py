"""The README examples run as written, in a fresh interpreter."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

README = Path(__file__).resolve().parents[1] / "README.md"


def test_readme_examples_run_in_a_fresh_interpreter():
    if not README.is_file():
        pytest.skip("README.md is not available next to these tests")
    blocks = re.findall(r"```python\n(.*?)```", README.read_text(encoding="utf-8"), re.S)
    assert blocks, "README.md has no Python examples"
    result = subprocess.run(
        [sys.executable, "-c", "\n".join(blocks)],
        cwd=README.parent,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr
