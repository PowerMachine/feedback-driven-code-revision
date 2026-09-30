from __future__ import annotations

from pathlib import Path


def write_patch_diff(path: str | Path, diff: dict[str, tuple[float, float]]) -> None:
    lines = ['# Patch Diff', '']
    if not diff:
        lines.append('- No changes applied.')
    else:
        for name, (old, new) in diff.items():
            lines.append(f'- {name} = {old}')
            lines.append(f'+ {name} = {new}')
            lines.append('')
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
