from __future__ import annotations

from rp_llm.common.schemas import PatchPlan
from rp_llm.patching.templates import TEMPLATES


class TemplateSynthesizer:
    def apply(self, plan: PatchPlan, source_code: str) -> str:
        template = TEMPLATES.get(plan.operator, "# keep")
        generated = template.format(**plan.patch_spec) if "{" in template else template
        return source_code + "\n# PATCH\n" + generated + "\n"
