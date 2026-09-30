from __future__ import annotations


def replay_ok(events: list[dict]) -> bool:
    return len(events) > 0 and not all(e.get("unsafe", False) for e in events)
