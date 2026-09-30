from __future__ import annotations

import difflib


def summarize_diff(before: str, after: str) -> str:
    diff = list(difflib.unified_diff(before.splitlines(), after.splitlines(), lineterm=""))
    changed = sum(1 for line in diff if line.startswith("+") or line.startswith("-"))
    return f"changed_lines={changed}"


def to_markdown(before: str, after: str) -> str:
    diff = "\n".join(difflib.unified_diff(before.splitlines(), after.splitlines(), fromfile="before", tofile="after", lineterm=""))
    return f"```diff\n{diff}\n```"
