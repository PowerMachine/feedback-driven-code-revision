from __future__ import annotations

TEMPLATES: dict[str, str] = {
    "param_patch": "gain = max(0.05, gain * {gain_scale})",
    "rule_patch": "if risk_score > 0.8: use_variant = 'conservative'",
    "skill_patch": "sequence = ['scan', 'approach', 'act', 'verify']",
    "code_patch": "controller_gain = controller_gain * {gain_scale}",
}
