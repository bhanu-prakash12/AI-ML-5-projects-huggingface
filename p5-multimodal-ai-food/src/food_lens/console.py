"""Narration helpers on `rich` with a plain-print fallback (also used when NO_COLOR is set).

Colors: cyan = operation, green = ok, yellow = warn/tradeoff, red = fail, dim = expected-output hint.
"""
from __future__ import annotations

import logging
import os
from typing import Any

try:  # rich is optional at runtime: everything degrades to print()
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    _RICH = True
except ImportError:  # pragma: no cover - exercised via monkeypatch in tests
    _RICH = False


def use_rich() -> bool:
    return _RICH and not os.environ.get("NO_COLOR")


def _con() -> Any:
    return Console(highlight=False, no_color=bool(os.environ.get("NO_COLOR")))


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging once (scripts previously logged silently)."""
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")


def banner(title: str, what: str, expect: str) -> None:
    if use_rich():
        body = f"{what}\n[dim]Expect: {expect}[/dim]"
        _con().print(Panel(body, title=f"[bold cyan]{title}[/bold cyan]", border_style="cyan"))
    else:
        print(f"=== {title} ===\n{what}\nExpect: {expect}")


def step(n: int, total: int, msg: str) -> None:
    if use_rich():
        _con().print(f"[cyan][{n}/{total}] {msg}[/cyan]")
    else:
        print(f"[{n}/{total}] {msg}")


def _say(style: str, tag: str, msg: str) -> None:
    if use_rich():
        _con().print(f"[{style}]{tag}[/{style}] {msg}")
    else:
        print(f"{tag} {msg}")


def ok(msg: str) -> None:
    _say("green", "OK  ", msg)


def warn(msg: str) -> None:
    _say("yellow", "WARN", msg)


def fail(msg: str) -> None:
    _say("red", "FAIL", msg)


def hint(msg: str) -> None:
    if use_rich():
        _con().print(f"[dim]{msg}[/dim]")
    else:
        print(msg)


def kv_table(data: dict, title: str = "") -> None:
    table([[str(k), str(v)] for k, v in data.items()], headers=["key", "value"], title=title)


def table(rows: list[list], headers: list[str] | None = None, title: str = "") -> None:
    rows = [[str(c) for c in r] for r in rows]
    if use_rich():
        t = Table(title=title or None, header_style="bold cyan")
        for h in headers or [""] * (len(rows[0]) if rows else 0):
            t.add_column(h)
        for r in rows:
            t.add_row(*r)
        _con().print(t)
        return
    if title:
        print(title)
    if headers:
        print(" | ".join(headers))
    for r in rows:
        print(" | ".join(r))


def note(pros: str, cons: str, scale: str) -> None:
    """Pros / Cons / At scale panel (1-3 lines each, no invented numbers)."""
    if use_rich():
        body = f"[green]Pros:[/green] {pros}\n[yellow]Cons:[/yellow] {cons}\n[cyan]At scale:[/cyan] {scale}"
        _con().print(Panel(body, title="Pros / Cons / At scale", border_style="yellow"))
    else:
        print(f"Pros: {pros}\nCons: {cons}\nAt scale: {scale}")
