"""Make every project's `src/` package importable from one process (package names are unique)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = ROOT / "projects"
for _p in ("p1", "p2", "p3", "p4", "p5"):
    _s = str(PROJECTS / _p / "src")
    if _s not in sys.path:
        sys.path.insert(0, _s)
