from __future__ import annotations


def build_failure_report(scenario_id: str, metrics: dict) -> dict:
    suggested_focus = "No change needed."
    if metrics["keepout_violations"] > 0:
        suggested_focus = "Increase keepout penalty or risk weight."
    elif metrics["battery_remaining"] < 2:
        suggested_focus = "Increase low-battery penalty for safer budget usage."
    elif not metrics["success"]:
        suggested_focus = "Adjust risk and replan thresholds to avoid dead-ends."

    return {
        "scenario_id": scenario_id,
        "success": metrics["success"],
        "path_length": metrics["path_length"],
        "keepout_violations": metrics["keepout_violations"],
        "battery_exhausted": metrics["battery_exhausted"],
        "replan_count": metrics["replan_count"],
        "failure_reason": metrics["failure_reason"] or "No major failure.",
        "suggested_focus": suggested_focus,
    }
