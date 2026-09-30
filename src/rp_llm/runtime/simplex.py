from __future__ import annotations


class SimplexGuard:
    def __init__(self, fallback_id: str = "gold") -> None:
        self.fallback_id = fallback_id

    def should_rollback(self, unsafe: bool, score: float) -> bool:
        return unsafe or score >= 0.95
