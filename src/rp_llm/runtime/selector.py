from __future__ import annotations

from rp_llm.common.schemas import EditCritique


class RuntimeSelector:
    def __init__(self) -> None:
        self.active_variant_id = "gold"

    def select(self, critique: EditCritique, verified_variants: set[str]) -> str:
        if critique.suggested_depth in {"L3", "L4"} and critique.target_module in verified_variants:
            self.active_variant_id = critique.target_module
        return self.active_variant_id
