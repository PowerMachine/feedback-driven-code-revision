from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field, ValidationError

from rp_llm.common.schemas import EditCritique, PatchPlan
from rp_llm.llm.client import QwenLocalClient
from rp_llm.llm.prompts import build_planner_prompt

ALLOWED_TARGET_MODULES = {"v_conservative"}
FORBIDDEN_KEYWORDS = {"safety_filter", "rollback", "verifier", "logging", "subprocess", "socket", "requests", "os.system"}


class PlannerOutput(BaseModel):
    target_module: str
    operator: str
    edit_depth: str
    objective: str
    patch_spec: dict = Field(default_factory=dict)
    verifier_constraints: list[str] = Field(default_factory=list)
    rationale: str


class LLMPatchPlanner:
    def __init__(self, llm_client: QwenLocalClient | None = None) -> None:
        self.client = llm_client or QwenLocalClient()

    def _validate_output(self, raw: dict) -> PlannerOutput:
        parsed = PlannerOutput.model_validate(raw)
        if parsed.target_module not in ALLOWED_TARGET_MODULES:
            raise ValueError(f"target_module '{parsed.target_module}' is not whitelisted")
        if parsed.operator not in {"keep", "param_patch", "rule_patch", "code_patch"}:
            raise ValueError(f"Unsupported operator: {parsed.operator}")
        if parsed.edit_depth not in {"L0", "L1", "L2", "L3", "L4"}:
            raise ValueError(f"Unsupported edit_depth: {parsed.edit_depth}")
        gain_x = float(parsed.patch_spec.get("controller_gain_x", 0.12))
        gain_y = float(parsed.patch_spec.get("controller_gain_y", 0.12))
        if not (0.05 <= gain_x <= 0.35 and 0.05 <= gain_y <= 0.35):
            raise ValueError("controller gains must stay in bounded range [0.05, 0.35]")
        return parsed

    def make_plan(self, critique: EditCritique, source_variant_id: str, source_before: str) -> PatchPlan:
        prompt = build_planner_prompt(critique, source_before)
        raw = self.client.generate_json(prompt)
        try:
            parsed = self._validate_output(raw)
        except (ValidationError, ValueError) as exc:
            raise RuntimeError(f"Invalid LLM patch plan JSON: {exc}") from exc
        return PatchPlan(
            critique_id=uuid4().hex,
            source_variant_id=source_variant_id,
            target_module=parsed.target_module,
            operator=parsed.operator,
            edit_depth=parsed.edit_depth,
            objective=parsed.objective,
            patch_spec=parsed.patch_spec,
            verifier_constraints=parsed.verifier_constraints,
            rationale=parsed.rationale,
        )

    def synthesize_controller_source(self, plan: PatchPlan) -> str:
        gain_x = float(plan.patch_spec.get("controller_gain_x", 0.12))
        gain_y = float(plan.patch_spec.get("controller_gain_y", 0.12))
        candidate_code = "\n".join(
            [
                "from __future__ import annotations",
                "",
                "class PatchedController:",
                "    def action(self, obs: dict) -> tuple[float, float]:",
                f"        gain_x = {gain_x}",
                f"        gain_y = {gain_y}",
                "        return obs['dx'] * gain_x, obs['dy'] * gain_y",
                "",
                "def build_controller() -> PatchedController:",
                "    return PatchedController()",
                "",
            ]
        )
        for token in FORBIDDEN_KEYWORDS:
            if token in candidate_code:
                raise RuntimeError(f"Generated code contains forbidden token: {token}")
        return candidate_code

    def write_candidate_file(self, candidate_id: str, source_after: str) -> Path:
        out_dir = Path("artifacts/candidates")
        out_dir.mkdir(parents=True, exist_ok=True)
        candidate_path = out_dir / f"{candidate_id}.py"
        candidate_path.write_text(source_after, encoding="utf-8")
        return candidate_path

