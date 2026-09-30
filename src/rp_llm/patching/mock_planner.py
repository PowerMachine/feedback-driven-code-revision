from __future__ import annotations

from uuid import uuid4

from rp_llm.common.schemas import EditCritique, PatchPlan


class MockPlanner:
    def make_plan(self, critique: EditCritique, source_variant_id: str) -> PatchPlan:
        gain_scale = 1.15 if critique.failure_type == "performance_drift" else 0.9
        return PatchPlan(
            critique_id=uuid4().hex,
            source_variant_id=source_variant_id,
            target_module=critique.target_module,
            operator=critique.suggested_operator,
            edit_depth=critique.suggested_depth,
            objective=f"Improve {critique.failure_type} while preserving safety",
            patch_spec={"gain_scale": gain_scale, "operator": critique.suggested_operator},
            verifier_constraints=critique.constraints,
        )
