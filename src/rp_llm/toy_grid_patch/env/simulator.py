from __future__ import annotations

from rp_llm.toy_grid_patch.env.gridworld import GridWorld


def simulate_path(env: GridWorld, path: list[tuple[int, int]], replan_count: int) -> dict:
    steps_taken = max(0, len(path) - 1)
    keepout_violations = sum(1 for p in path if p in env.keepout_zones)
    success = bool(path) and path[-1] == env.goal and steps_taken <= env.battery_budget
    battery_remaining = env.battery_budget - steps_taken
    battery_exhausted = battery_remaining < 0
    failure_reason = ""
    if not success:
        if not path or path[-1] != env.goal:
            failure_reason = "Goal was not reached."
        elif battery_exhausted:
            failure_reason = "Battery budget exceeded before reaching goal."
    elif keepout_violations > 0:
        failure_reason = "Reached goal but violated keepout safety preference."

    return {
        "success": success,
        "path_length": steps_taken,
        "keepout_violations": keepout_violations,
        "replan_count": replan_count,
        "battery_remaining": battery_remaining,
        "battery_exhausted": battery_exhausted,
        "failure_reason": failure_reason,
    }
