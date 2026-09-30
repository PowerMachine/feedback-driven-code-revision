from __future__ import annotations

from pathlib import Path
from typing import Any


def build_explanation(
    *,
    scenario: dict,
    failure_report: dict,
    patch_payload: dict,
    diff: dict[str, tuple[Any, Any]],
    baseline_metrics: dict,
    patched_metrics: dict,
    improved: bool,
    validator_reason: str,
) -> str:
    lines: list[str] = []
    lines.append(f"# Run Explanation: {scenario.get('scenario_id', 'unknown')}")
    lines.append("")

    lines.append("## Why the baseline failed")
    lines.append(f"- success: {failure_report.get('success')}")
    lines.append(f"- path_length: {failure_report.get('path_length')}")
    lines.append(f"- keepout_violations: {failure_report.get('keepout_violations')}")
    lines.append(f"- battery_exhausted: {failure_report.get('battery_exhausted')}")
    lines.append(f"- replan_count: {failure_report.get('replan_count')}")
    lines.append(f"- failure_reason: {failure_report.get('failure_reason')}")
    lines.append(f"- suggested_focus: {failure_report.get('suggested_focus')}")
    lines.append("")

    lines.append("## What the patcher proposed")
    lines.append(f"- validator_reason: {validator_reason}")
    if not patch_payload.get("changes"):
        lines.append("- No changes (baseline kept).")
    else:
        for c in patch_payload["changes"]:
            old = diff.get(c["name"], (None, c.get("new_value")))[0]
            lines.append(
                f"- {c['name']}: {old} -> {c['new_value']}  \n  reason: {c.get('reason', '')}"
            )
    lines.append("")

    lines.append("## Expected effect (stated by patcher)")
    lines.append(f"- {patch_payload.get('expected_effect', 'n/a')}")
    lines.append("")

    lines.append("## Observed effect (baseline vs patched)")
    metric_keys = [
        "success",
        "path_length",
        "keepout_violations",
        "replan_count",
        "battery_remaining",
    ]
    lines.append("| metric | baseline | patched | delta |")
    lines.append("|---|---|---|---|")
    for key in metric_keys:
        b = baseline_metrics.get(key)
        p = patched_metrics.get(key)
        try:
            delta = p - b  # type: ignore[operator]
            delta_str = f"{delta:+}"
        except Exception:
            delta_str = "n/a"
        lines.append(f"| {key} | {b} | {p} | {delta_str} |")
    lines.append("")

    lines.append("## Verdict")
    lines.append(f"- improved: {improved}")
    if improved:
        lines.append(
            "- 해석: 변경한 상수가 planner의 cost 지형을 재구성해, keepout/재계획 비용을 높이고 안전한 우회 경로를 선택하게 만든 것으로 보입니다."
        )
    else:
        lines.append(
            "- 해석: patch가 거부되었거나 cost 변화가 경로 선택을 바꿀 만큼 크지 않았습니다. 값 크기/조합 조정을 다음 단계 후보로 제시합니다."
        )

    return "\n".join(lines) + "\n"


def write_explanation_md(path: str | Path, content: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")
