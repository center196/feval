from __future__ import annotations

import argparse
import os
import sys
from typing import Any


ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "cyan": "\033[36m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
}


def _use_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def color(text: str, name: str) -> str:
    if not _use_color():
        return text
    return f"{ANSI[name]}{text}{ANSI['reset']}"


class FevalHelpFormatter(argparse.RawTextHelpFormatter):
    def start_section(self, heading: str | None) -> None:
        if heading:
            heading = color(heading.upper(), "cyan")
        super().start_section(heading)


def banner() -> str:
    return "\n".join([
        color("Feval Subnet", "bold"),
        color("cheap evaluation, probabilistic rollout audits, champion rewards", "dim"),
    ])


def print_table(title: str, rows: dict[str, Any]) -> None:
    print(color(title, "green"))
    if not rows:
        return
    width = max(len(str(key)) for key in rows)
    for key, value in rows.items():
        if value is None:
            continue
        print(f"  {color(str(key).ljust(width), 'cyan')}  {value}")


def fail(message: str, exit_code: int = 1) -> None:
    print(color(f"error: {message}", "red"), file=sys.stderr)
    raise SystemExit(exit_code)


