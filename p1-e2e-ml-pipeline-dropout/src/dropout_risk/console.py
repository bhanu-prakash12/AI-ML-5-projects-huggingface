"""Tiny narration helper on `rich` with a plain-print fallback (respects NO_COLOR).

Colors: cyan=operation, green=ok, yellow=warn/tradeoff, red=fail, dim=expected-output hints.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from typing import Any

try:  # rich is optional at runtime: fall back to print
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    _RICH = True
except ImportError:  # pragma: no cover - exercised via monkeypatch in tests
    _RICH = False

_console: Any = None


def use_rich() -> bool:
    """True when rich is importable (colors are then disabled by NO_COLOR via rich itself)."""
    return _RICH


def _con() -> Any:
    global _console
    if _console is None and _RICH:
        _console = Console(no_color=bool(os.environ.get("NO_COLOR")))
    return _console


def _say(text: str, style: str = "") -> None:
    c = _con() if _RICH else None
    if c is not None:
        c.print(text, style=style, highlight=False, markup=False)
    else:
        print(text)


def setup_logging(level: int = logging.INFO) -> None:
    """One logging config for every script (idempotent)."""
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")


def banner(title: str, what: str, expect: str) -> None:
    body = f"{what}\nExpect: {expect}"
    c = _con() if _RICH else None
    if c is not None:
        c.print(Panel(body, title=title, border_style="cyan", expand=False))
    else:
        print(f"== {title} ==\n{body}")


def step(n: int, total: int, msg: str) -> None:
    _say(f"[{n}/{total}] {msg}", "cyan")


def ok(msg: str) -> None:
    _say(f"OK   {msg}", "green")


def warn(msg: str) -> None:
    _say(f"WARN {msg}", "yellow")


def fail(msg: str) -> None:
    _say(f"FAIL {msg}", "red")


def hint(msg: str) -> None:
    _say(msg, "dim")


def table(rows: Sequence[Sequence[Any]], headers: Sequence[str] | None = None,
          title: str | None = None) -> None:
    c = _con() if _RICH else None
    if c is not None:
        t = Table(title=title, show_header=headers is not None)
        for h in headers or (["" ] * (len(rows[0]) if rows else 0)):
            t.add_column(str(h))
        for r in rows:
            t.add_row(*[str(v) for v in r])
        c.print(t)
        return
    if title:
        print(title)
    if headers:
        print(" | ".join(str(h) for h in headers))
    for r in rows:
        print(" | ".join(str(v) for v in r))


def kv_table(d: dict[str, Any], title: str | None = None) -> None:
    table([(k, v) for k, v in d.items()], headers=("key", "value"), title=title)


def note(pros: str, cons: str, scale: str) -> None:
    body = f"Pros:     {pros}\nCons:     {cons}\nAt scale: {scale}"
    c = _con() if _RICH else None
    if c is not None:
        c.print(Panel(body, title="Pros / Cons / At scale", border_style="yellow", expand=False))
    else:
        print(f"-- Pros / Cons / At scale --\n{body}")
