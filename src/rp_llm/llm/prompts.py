from __future__ import annotations

import json

from rp_llm.common.schemas import EditCritique


PATCH_PLAN_JSON_SCHEMA = {
    "type": "object",
    "required": [
        "target_module",
        "operator",
        "edit_depth",
        "objective",
        "patch_spec",
        "verifier_constraints",
        "rationale",
    ],
    "properties": {
        "target_module": {"type": "string", "enum": ["v_conservative"]},
        "operator": {"type": "string", "enum": ["keep", "param_patch", "rule_patch", "code_patch"]},
        "edit_depth": {"type": "string", "enum": ["L0", "L1", "L2", "L3", "L4"]},
        "objective": {"type": "string"},
        "patch_spec": {
            "type": "object",
            "required": ["controller_gain_x", "controller_gain_y"],
            "properties": {
                "controller_gain_x": {"type": "number", "minimum": 0.05, "maximum": 0.35},
                "controller_gain_y": {"type": "number", "minimum": 0.05, "maximum": 0.35},
            },
        },
        "verifier_constraints": {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": "string"},
    },
}


def build_planner_prompt(critique: EditCritique, source_before: str) -> str:
    return "\n".join(
        [
            "You are a bounded RP-LLM patch planner.",
            "Output STRICT JSON only, no markdown.",
            "Do not add keys outside the schema.",
            "Never request shell commands, network calls, or dependency changes.",
            "Forbidden edit targets: safety_filter, rollback, verifier, logging, low-level control.",
            "Allowed target_module: v_conservative.",
            "Use minimal patch surface by changing only controller gains.",
            f"Critique JSON: {critique.model_dump_json()}",
            f"Source-before snippet:\n{source_before}",
            f"JSON schema: {json.dumps(PATCH_PLAN_JSON_SCHEMA)}",
        ]
    )

