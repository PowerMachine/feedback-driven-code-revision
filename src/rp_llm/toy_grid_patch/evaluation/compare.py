from __future__ import annotations


def comparison_row(scenario_id: str, before: dict, after: dict, improved: bool) -> dict:
    return {
        "scenario_id": scenario_id,
        "baseline_success": before["success"],
        "patched_success": after["success"],
        "baseline_path_length": before["path_length"],
        "patched_path_length": after["path_length"],
        "baseline_keepout_violations": before["keepout_violations"],
        "patched_keepout_violations": after["keepout_violations"],
        "baseline_replan_count": before["replan_count"],
        "patched_replan_count": after["replan_count"],
        "baseline_battery_remaining": before["battery_remaining"],
        "patched_battery_remaining": after["battery_remaining"],
        "improved": improved,
    }
