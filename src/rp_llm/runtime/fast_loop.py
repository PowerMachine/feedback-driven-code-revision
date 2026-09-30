from __future__ import annotations

from rp_llm.runtime.rollback import rollback_to_gold
from rp_llm.runtime.simplex import SimplexGuard


class FastLoopRuntime:
    def __init__(self, verified_variants: set[str] | None = None) -> None:
        self.verified_variants = verified_variants or {"gold"}
        self.guard = SimplexGuard()

    def enforce_verified_only(self, variant_id: str) -> str:
        if variant_id not in self.verified_variants:
            return "gold"
        return variant_id

    def handle_fault(self, unsafe: bool, risk_score: float, variant_id: str) -> str:
        if self.guard.should_rollback(unsafe=unsafe, score=risk_score):
            return rollback_to_gold()
        return self.enforce_verified_only(variant_id)
