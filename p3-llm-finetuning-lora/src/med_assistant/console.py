"""Tiny console helper on `rich` (plain-print fallback; respects NO_COLOR).

cyan = operation, green = ok, yellow = warn/tradeoff, red = fail, dim = expected-output hints.
"""
from __future__ import annotations

import logging
import os
from typing import Any

try:  # rich is optional at runtime: fall back to print
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    _CON: Any = Console(no_color=bool(os.environ.get("NO_COLOR")))
except Exception:  # noqa: BLE001
    _CON = None


def _p(msg: str = "", style: str = "") -> None:
    if _CON is not None:
        _CON.print(msg, style=style or None, highlight=False, markup=False)
    else:
        print(msg, flush=True)


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)


def banner(title: str, what: str, expect: str) -> None:
    if _CON is not None:
        _CON.print(Panel(f"{what}\n[dim]Expect: {expect}[/dim]", title=f"[bold cyan]{title}[/bold cyan]",
                         border_style="cyan"))
    else:
        print(f"== {title} ==\n{what}\nExpect: {expect}", flush=True)


def step(n: int, total: int, msg: str) -> None:
    _p(f"[{n}/{total}] {msg}", "cyan")


def ok(msg: str) -> None:
    _p(f"OK   {msg}", "green")


def warn(msg: str) -> None:
    _p(f"WARN {msg}", "yellow")


def fail(msg: str) -> None:
    _p(f"FAIL {msg}", "red")


def hint(msg: str) -> None:
    _p(msg, "dim")


def table(rows: list[list[Any]], headers: list[str] | None = None, title: str | None = None) -> None:
    if _CON is not None:
        t = Table(title=title, show_header=bool(headers))
        for h in headers or [""] * (len(rows[0]) if rows else 0):
            t.add_column(str(h), overflow="fold")
        for r in rows:
            t.add_row(*[str(c) for c in r])
        _CON.print(t)
        return
    if title:
        print(title)
    if headers:
        print(" | ".join(map(str, headers)))
    for r in rows:
        print(" | ".join(map(str, r)))


def kv_table(d: dict[str, Any], title: str | None = None) -> None:
    table([[k, v] for k, v in d.items()], ["key", "value"], title)


def note(pros: str, cons: str, scale: str) -> None:
    body = f"Pros:     {pros}\nCons:     {cons}\nAt scale: {scale}"
    if _CON is not None:
        _CON.print(Panel(body, title="[yellow]Pros / Cons / At scale[/yellow]", border_style="yellow"))
    else:
        print(body, flush=True)
