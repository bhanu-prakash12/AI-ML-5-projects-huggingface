"""Narration helpers on `rich` with a plain-print fallback. Respects NO_COLOR."""
from __future__ import annotations

import logging
import os
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
    return _RICH


def _con() -> Any:
    global _console
    if _console is None and _RICH:
        _console = Console(no_color=bool(os.environ.get("NO_COLOR")), highlight=False)
    return _console


def _out(markup: str, plain: str) -> None:
    c = _con() if _RICH else None
    if c is not None:
        c.print(markup)
    else:
        print(plain, flush=True)


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(message)s")


def banner(title: str, what: str, expect: str) -> None:
    c = _con() if _RICH else None
    if c is not None:
        c.print(Panel(f"{what}\n[dim]Expect: {expect}[/dim]", title=f"[bold cyan]{title}[/bold cyan]", border_style="cyan"))
    else:
        print(f"=== {title} ===\n{what}\nExpect: {expect}", flush=True)


def step(n: int, total: int, msg: str) -> None:
    _out(f"[cyan]\\[{n}/{total}] {msg}[/cyan]", f"[{n}/{total}] {msg}")


def ok(msg: str) -> None:
    _out(f"[green]OK[/green] {msg}", f"OK {msg}")


def warn(msg: str) -> None:
    _out(f"[yellow]WARN[/yellow] {msg}", f"WARN {msg}")


def fail(msg: str) -> None:
    _out(f"[red]FAIL[/red] {msg}", f"FAIL {msg}")


def hint(msg: str) -> None:
    _out(f"[dim]{msg}[/dim]", msg)


def table(rows: list[list[Any]], headers: list[str] | None = None, title: str | None = None) -> None:
    rows = [[str(x) for x in r] for r in rows]
    c = _con() if _RICH else None
    if c is not None:
        t = Table(title=title, header_style="bold cyan")
        for h in headers or [""] * (len(rows[0]) if rows else 0):
            t.add_column(h)
        for r in rows:
            t.add_row(*r)
        c.print(t)
    else:
        if title:
            print(title)
        if headers:
            print(" | ".join(headers))
        for r in rows:
            print(" | ".join(r))


def kv_table(d: dict[str, Any], title: str | None = None) -> None:
    table([[k, v] for k, v in d.items()], ["key", "value"], title)


def note(pros: str, cons: str, scale: str) -> None:
    c = _con() if _RICH else None
    if c is not None:
        c.print(Panel(f"[green]Pros:[/green] {pros}\n[yellow]Cons:[/yellow] {cons}\n[yellow]At scale:[/yellow] {scale}",
                      title="Trade-offs", border_style="yellow"))
    else:
        print(f"Pros: {pros}\nCons: {cons}\nAt scale: {scale}", flush=True)
