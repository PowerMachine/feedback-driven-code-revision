from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AxisScores(BaseModel):
    safety: float = 0.0
    environment: float = 0.0
    context: float = 0.0
    performance: float = 0.0
    internal: float = 0.0
    uncertainty: float = 0.0


class EditCritique(BaseModel):
    axis_scores: AxisScores
    failure_type: str
    target_module: str
    suggested_operator: Literal[
        "keep",
        "param_patch",
        "skill_patch",
        "rule_patch",
        "code_patch",
        "reward_patch",
    ]
    suggested_depth: Literal["L0", "L1", "L2", "L3", "L4"]
    constraints: list[str] = Field(default_factory=list)
    evidence_summary: str
    edit_recommendation: str = ""
    confidence: float = 0.0


class PatchPlan(BaseModel):
    critique_id: str
    source_variant_id: str
    target_module: str
    operator: str
    edit_depth: str
    objective: str
    patch_spec: dict = Field(default_factory=dict)
    verifier_constraints: list[str] = Field(default_factory=list)
    rationale: str = ""


class PatchCandidate(BaseModel):
    candidate_id: str
    parent_variant_id: str
    plan: PatchPlan
    source_before: str
    source_after: str
    diff_summary: str
    candidate_path: str = ""


class VerificationBundle(BaseModel):
    candidate_id: str
    syntax_ok: bool
    type_ok: bool
    unit_ok: bool
    replay_ok: bool
    safety_ok: bool
    regression_ok: bool
    canary_ok: bool
    fail_summary: str | None = None
    verifier_hash: str
    details: dict = Field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(
            [
                self.syntax_ok,
                self.type_ok,
                self.unit_ok,
                self.replay_ok,
                self.safety_ok,
                self.regression_ok,
                self.canary_ok,
            ]
        )


class VariantManifest(BaseModel):
    variant_id: str
    parent_id: str | None
    promotion_level: Literal["gold", "silver", "canary", "quarantine"]
    signature: dict = Field(default_factory=dict)
    preconditions: list[str] = Field(default_factory=list)
    fallback_id: str
    verifier_hash: str
    success_rate: float = 0.0
    unsafe_events: int = 0
    rollback_rate: float = 0.0
    p95_cycle_time_ms: float = 0.0


class RuntimeEvent(BaseModel):
    event: str
    variant_id: str
    message: str
    payload: dict = Field(default_factory=dict)
