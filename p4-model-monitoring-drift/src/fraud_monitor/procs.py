"""Cross-platform process helpers (stdlib only): venv python, background API, PID files, health wait."""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def is_windows() -> bool:
    return os.name == "nt"


def venv_python(root: Path, windows: bool | None = None) -> Path:
    win = is_windows() if windows is None else windows
    return root / ".venv" / ("Scripts/python.exe" if win else "bin/python")


def resolve_python(root: Path) -> str:
    """Project venv python if it exists, else the interpreter running this script."""
    p = venv_python(root)
    return str(p) if p.exists() else sys.executable


def kill_cmd(pid: int, windows: bool | None = None) -> list[str] | None:
    """Command that kills a process tree on Windows; None means use os.kill (POSIX)."""
    win = is_windows() if windows is None else windows
    return ["taskkill", "/PID", str(pid), "/T", "/F"] if win else None


def terminate(pid: int) -> None:
    cmd = kill_cmd(pid)
    try:
        if cmd:
            subprocess.run(cmd, capture_output=True, check=False)
        else:
            os.kill(pid, 15)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def stop_pid_files(run_dir: Path) -> list[int]:
    """Kill every PID recorded under run_dir/*.pid and delete the files."""
    killed = []
    for f in sorted(run_dir.glob("*.pid")):
        try:
            pid = int(f.read_text().strip())
            terminate(pid)
            killed.append(pid)
        except ValueError:
            pass
        f.unlink(missing_ok=True)
    return killed


def start_background(cmd: list[str], env: dict[str, str], cwd: Path, log_path: Path, pid_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "wb")
    kw: dict[str, Any] = {}
    if is_windows():
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    else:
        kw["start_new_session"] = True
    p = subprocess.Popen(cmd, env=env, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, **kw)
    pid_path.write_text(str(p.pid))
    return p


def http_json(url: str, method: str = "GET", body: Any = None, timeout: float = 30) -> Any:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method, headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - localhost only
        return json.loads(r.read().decode())


def wait_health(url: str, timeout: float = 90, proc: subprocess.Popen | None = None) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            if http_json(url, timeout=2).get("status") == "ok":
                return True
        except (urllib.error.URLError, OSError, ValueError):
            pass
        time.sleep(1)
    return False


def port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def tail(path: Path, n: int = 20) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return ""
