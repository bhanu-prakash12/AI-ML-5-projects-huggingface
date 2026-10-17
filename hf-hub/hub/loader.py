"""Load a staged project script (projects/pN/scripts/<name>.py) under a unique module name.

Every project ships `scripts/ui.py`; importing them by bare name would collide, so each gets `pN_<name>`.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from hub.paths import PROJECTS


def load_script(project: str, name: str = "ui") -> ModuleType:
    key = f"{project}_{name}"
    if key in sys.modules:
        return sys.modules[key]
    scripts = PROJECTS / project / "scripts"
    spec = importlib.util.spec_from_file_location(key, scripts / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    sys.path.insert(0, str(scripts))  # scripts may `import _boot`
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(scripts))
    return mod


def project_dir(project: str) -> Path:
    return PROJECTS / project
