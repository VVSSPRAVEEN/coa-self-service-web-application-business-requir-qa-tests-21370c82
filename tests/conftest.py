"""Configuration shared by every test module under ``tests/``.

The test tree is executed from several working directories (repository root
through ``pytest``, and ``tests/`` through ``python -m pytest tests``), so the
same import roots the containers use are established here as well. Doing it in
one place keeps collection identical for the API, document and notification
suites and avoids per-test ``sys.path`` juggling.

Command line options and markers are NOT declared here: only the root
``conftest.py`` owns ``pytest_addoption`` / ``pytest_configure``. A second
``pytest_addoption`` anywhere in the conftest chain makes pytest abort during
startup with a usage error, so this module intentionally limits itself to the
import roots.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

#: Import roots, in priority order, mirroring the container PYTHONPATH.
_IMPORT_ROOTS = (
    str(_REPO_ROOT),
    str(_REPO_ROOT / "src"),
    str(_REPO_ROOT / "tests"),
)

for _entry in reversed(_IMPORT_ROOTS):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)
