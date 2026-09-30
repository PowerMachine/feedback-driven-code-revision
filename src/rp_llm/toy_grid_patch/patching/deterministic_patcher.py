from __future__ import annotations


def propose_patch(current_rules: dict, baseline_metrics: dict) -> dict:
    changes: list[dict] = []

    if baseline_metrics["keepout_violations"] > 0:
        changes.append(
            {
                "name": "KEEP_OUT_PENALTY",
                "new_value": min(5000, max(300, int(current_rules["KEEP_OUT_PENALTY"] * 4))),
                "reason": "Reduce keepout violations by increasing safety cost.",
            }
        )

    if baseline_metrics["battery_remaining"] < 2:
        changes.append(
            {
                "name": "LOW_BATTERY_PENALTY",
                "new_value": min(1000, int(current_rules["LOW_BATTERY_PENALTY"] + 60)),
                "reason": "Bias planning away from near-budget risky paths.",
            }
        )

    if baseline_metrics["replan_count"] > 0 and len(changes) < 3:
        changes.append(
            {
                "name": "RISK_WEIGHT",
                "new_value": min(10.0, round(float(current_rules["RISK_WEIGHT"]) + 0.8, 2)),
                "reason": "Increase penalty impact for risky detours.",
            }
        )

    return {
        "changes": changes[:3],
        "expected_effect": "Prefer safer detour with fewer keepout and replan events.",
    }
