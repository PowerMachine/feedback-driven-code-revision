from __future__ import annotations

from pathlib import Path
import re


_PATTERN = re.compile(r'^(?P<name>[A-Z_]+)\s*=\s*(?P<value>.+)$')


def read_rules(rule_path: str | Path) -> dict:
    out = {}
    for raw in Path(rule_path).read_text(encoding='utf-8').splitlines():
        m = _PATTERN.match(raw.strip())
        if not m:
            continue
        name = m.group('name')
        value = m.group('value').strip()
        try:
            out[name] = int(value)
        except ValueError:
            try:
                out[name] = float(value)
            except ValueError:
                pass
    return out


def apply_patch(rule_path: str | Path, payload: dict) -> dict[str, tuple[float, float]]:
    path = Path(rule_path)
    lines = path.read_text(encoding='utf-8').splitlines()
    changes = {c['name']: c['new_value'] for c in payload.get('changes', [])}
    old_vals = read_rules(path)
    for idx, line in enumerate(lines):
        m = _PATTERN.match(line.strip())
        if not m:
            continue
        name = m.group('name')
        if name in changes:
            lines[idx] = f"{name} = {changes[name]}"
    new_text = "\n".join(lines) + "\n"
    compile(new_text, str(path), 'exec')
    path.write_text(new_text, encoding='utf-8')
    diff = {}
    for name, new_val in changes.items():
        diff[name] = (old_vals.get(name), new_val)
    return diff
