from __future__ import annotations

from pathlib import Path
import json
import pandas as pd


def write_failure_report_json(path: str | Path, report: dict) -> None:
    Path(path).write_text(json.dumps(report, indent=2), encoding='utf-8')


def write_comparison_csv(path: str | Path, row: dict) -> None:
    pd.DataFrame([row]).to_csv(path, index=False)


def write_summary_md(path: str | Path, baseline: dict, patched: dict, patch_payload: dict, improved: bool) -> None:
    lines = [
        "# Constraint-Aware GridWorld Patch Summary",
        "",
        "## Baseline 결과",
        f"- success: {baseline['success']}",
        f"- path_length: {baseline['path_length']}",
        f"- keepout_violations: {baseline['keepout_violations']}",
        f"- replan_count: {baseline['replan_count']}",
        f"- battery_remaining: {baseline['battery_remaining']}",
        "",
        "## Patched 결과",
        f"- success: {patched['success']}",
        f"- path_length: {patched['path_length']}",
        f"- keepout_violations: {patched['keepout_violations']}",
        f"- replan_count: {patched['replan_count']}",
        f"- battery_remaining: {patched['battery_remaining']}",
        "",
        "## 변경 규칙",
    ]
    for c in patch_payload.get('changes', []):
        lines.append(f"- {c['name']} -> {c['new_value']} ({c.get('reason','')})")
    if not patch_payload.get('changes'):
        lines.append('- 변경 없음 (baseline 유지)')
    lines += ["", "## 개선 여부", f"- improved: {improved}"]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
