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


def print_rows_table(
    title: str,
    columns: list[tuple[str, str, str]],
    rows: list[dict[str, Any]],
    *,
    stream: Any = None,
) -> None:
    """Print a compact dependency-free table.

    Column specs are ``(field, heading, alignment)`` where alignment is
    ``left`` or ``right``. ANSI color is applied after padding so widths remain
    stable in interactive terminals.
    """

    output = stream or sys.stdout
    if not rows:
        return
    rendered = [
        {field: "-" if row.get(field) is None else str(row.get(field)) for field, _, _ in columns}
        for row in rows
    ]
    widths = {
        field: max(len(heading), *(len(row[field]) for row in rendered))
        for field, heading, _ in columns
    }

    def cell(value: str, field: str, alignment: str) -> str:
        return value.rjust(widths[field]) if alignment == "right" else value.ljust(widths[field])

    print(color(title, "green"), file=output)
    print(
        "  "
        + "  ".join(
            color(cell(heading, field, alignment), "cyan")
            for field, heading, alignment in columns
        ),
        file=output,
    )
    print(
        "  " + "  ".join("-" * widths[field] for field, _, _ in columns),
        file=output,
    )
    for row in rendered:
        values = []
        for field, _, alignment in columns:
            value = cell(row[field], field, alignment)
            if field == "outcome":
                status = row[field].lower()
                tone = "green" if status in {"valid", "pass"} else (
                    "yellow" if status in {"auditing", "retry"} else "red"
                )
                value = color(value, tone)
            elif field.startswith("validator_"):
                status = row[field].lower()
                tone = (
                    "green"
                    if status.startswith(("king", "valid"))
                    else "yellow"
                    if status.startswith(("audit", "retry"))
                    else "red"
                    if status.startswith(("invalid", "blacklist"))
                    else "dim"
                )
                value = color(value, tone)
            values.append(value)
        print("  " + "  ".join(values), file=output)
    output.flush()


def fail(message: str, exit_code: int = 1) -> None:
    print(color(f"error: {message}", "red"), file=sys.stderr)
    raise SystemExit(exit_code)


