from __future__ import annotations


def canary_ok(events: list[dict], max_unsafe_ratio: float = 0.1) -> bool:
    if not events:
        return False
    unsafe = sum(1 for e in events if e.get("unsafe", False))
    return (unsafe / len(events)) <= max_unsafe_ratio
