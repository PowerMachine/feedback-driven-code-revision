from __future__ import annotations

from rp_llm.common.schemas import EditCritique, PatchPlan


class LLMPlannerAdapter:
    """
    TODO: Connect external LLM backend with bounded retries and verifier feedback.
    """

    def propose(self, critique: EditCritique, source_variant_id: str) -> PatchPlan:
        return PatchPlan(
            critique_id="llm_stub",
            source_variant_id=source_variant_id,
            target_module=critique.target_module,
            operator=critique.suggested_operator,
            edit_depth=critique.suggested_depth,
            objective="TODO: fill by real LLM planner",
            patch_spec={},
            verifier_constraints=critique.constraints,
        )
