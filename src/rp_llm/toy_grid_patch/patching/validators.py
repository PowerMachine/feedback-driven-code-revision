from __future__ import annotations

ALLOWED_CONSTANTS = {
    "MOVE_COST": (0, 1000),
    "KEEP_OUT_PENALTY": (0, 5000),
    "OBSTACLE_NEAR_PENALTY": (0, 1000),
    "LOW_BATTERY_PENALTY": (0, 1000),
    "RISK_WEIGHT": (0.0, 10.0),
    "REPLAN_THRESHOLD": (1, 20),
}


def validate_patch_payload(payload: dict) -> tuple[bool, str]:
    changes = payload.get('changes', [])
    if not isinstance(changes, list):
        return False, 'changes must be a list'
    if len(changes) > 3:
        return False, 'At most 3 constants can be changed'

    touched = set()
    for item in changes:
        name = item.get('name')
        if name not in ALLOWED_CONSTANTS:
            return False, f'Constant not allowed: {name}'
        if name in touched:
            return False, f'Duplicate change for constant: {name}'
        touched.add(name)
        lo, hi = ALLOWED_CONSTANTS[name]
        val = item.get('new_value')
        if not isinstance(val, (int, float)):
            return False, f'new_value must be numeric for {name}'
        if val < lo or val > hi:
            return False, f'new_value out of range for {name}: {val}'

    return True, 'ok'
