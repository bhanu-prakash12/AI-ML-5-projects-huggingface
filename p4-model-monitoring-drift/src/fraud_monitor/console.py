"""Narration helpers on `rich` with a plain-print fallback (respects NO_COLOR).

Colour scheme: cyan = operation, green = ok, yellow = warn / trade-off, red = fail, dim = expected-output hints.
Text stays ASCII-only so a plain Windows console (cp1252) never chokes on it.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Any, Callable, Mapping, Sequence

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    _RICH = True
except ImportError:  # pragma: no cover - exercised via monkeypatch in tests
    _RICH = False

_con: Any = None


def _console():
    """Lazy rich Console (None -> plain print). Re-evaluated per call so tests can flip `_RICH`."""
    global _con
    if not _RICH:
        return None
    if _con is None:
        _con = Console(highlight=False, no_color=bool(os.environ.get("NO_COLOR")))
    return _con


def _out(text: str, style: str | None = None) -> None:
    c = _console()
    if c is None:
        print(text)
    else:
        c.print(text, style=style, markup=False)


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr, force=True)


def banner(title: str, what: str, expect: str) -> None:
    c = _console()
    if c is None:
        print(f"\n=== {title} ===\nWhat: {what}\nExpect: {expect}\n")
        return
    c.print(Panel.fit(f"{what}\n[dim]Expect: {expect}[/dim]", title=title, border_style="cyan"))


def step(n: int, total: int, msg: str) -> None:
    _out(f"[{n}/{total}] {msg}", "cyan")


def ok(msg: str) -> None:
    _out(f"OK   {msg}", "green")


def warn(msg: str) -> None:
    _out(f"WARN {msg}", "yellow")


def fail(msg: str) -> None:
    _out(f"FAIL {msg}", "red")


def hint(msg: str) -> None:
    _out(f"     {msg}", "dim")


def kv_table(d: Mapping[str, Any], title: str | None = None) -> None:
    c = _console()
    if c is None:
        print((title + ":") if title else "")
        for k, v in d.items():
            print(f"  {k}: {v}")
        return
    t = Table(title=title, show_header=False, box=None, pad_edge=False)
    t.add_column(style="cyan")
    t.add_column()
    for k, v in d.items():
        t.add_row(str(k), str(v))
    c.print(t)


def table(rows: Sequence[Mapping[str, Any]], title: str | None = None,
          highlight: Callable[[Mapping[str, Any]], str | None] | None = None) -> None:
    """Print rows (list of dicts, same keys). `highlight(row)` may return a rich style, e.g. 'red'."""
    if not rows:
        _out("(no rows)", "dim")
        return
    cols = list(rows[0].keys())
    c = _console()
    if c is None:
        if title:
            print(title)
        print(" | ".join(cols))
        for r in rows:
            print(" | ".join(str(r.get(k, "")) for k in cols))
        return
    t = Table(title=title)
    for k in cols:
        t.add_column(str(k))
    for r in rows:
        t.add_row(*[str(r.get(k, "")) for k in cols], style=(highlight(r) if highlight else None))
    c.print(t)


def note(pros: str, cons: str, scale: str) -> None:
    c = _console()
    if c is None:
        print(f"Pros: {pros}\nCons: {cons}\nAt scale: {scale}\n")
        return
    c.print(Panel(f"[green]Pros:[/green] {pros}\n[yellow]Cons:[/yellow] {cons}\n[yellow]At scale:[/yellow] {scale}",
                  title="Trade-offs", border_style="yellow"))
