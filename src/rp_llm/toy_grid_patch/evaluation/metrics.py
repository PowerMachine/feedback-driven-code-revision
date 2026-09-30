from __future__ import annotations


def is_improved(before: dict, after: dict) -> bool:
    before_tuple = (
        int(before["success"]),
        -int(before["keepout_violations"]),
        -int(before["path_length"]),
        -int(before["replan_count"]),
        int(before["battery_remaining"]),
    )
    after_tuple = (
        int(after["success"]),
        -int(after["keepout_violations"]),
        -int(after["path_length"]),
        -int(after["replan_count"]),
        int(after["battery_remaining"]),
    )
    return after_tuple > before_tuple
