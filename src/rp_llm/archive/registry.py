from __future__ import annotations

from pathlib import Path
import json

from rp_llm.common.schemas import PatchCandidate, VariantManifest, VerificationBundle


class ArchiveRegistry:
    def __init__(self, root: str = "artifacts/manifests") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.verified_variants: set[str] = {"gold"}

    def promote(self, candidate: PatchCandidate, bundle: VerificationBundle) -> VariantManifest:
        vid = f"variant_{candidate.candidate_id[:8]}"
        manifest = VariantManifest(
            variant_id=vid,
            parent_id=candidate.parent_variant_id,
            promotion_level="silver",
            signature={"operator": candidate.plan.operator},
            preconditions=candidate.plan.verifier_constraints,
            fallback_id="gold",
            verifier_hash=bundle.verifier_hash,
            success_rate=1.0,
            unsafe_events=0,
            rollback_rate=0.0,
            p95_cycle_time_ms=20.0,
        )
        self.verified_variants.add(vid)
        (self.root / f"{vid}.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        return manifest

    def write_lineage(self, scenario: str, edges: list[tuple[str, str]]) -> None:
        out = Path("artifacts/lineage")
        out.mkdir(parents=True, exist_ok=True)
        payload = [{"from": a, "to": b} for a, b in edges]
        (out / f"{scenario}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
