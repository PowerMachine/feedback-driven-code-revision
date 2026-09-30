from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path

from rp_llm.common.schemas import PatchCandidate, VerificationBundle
from rp_llm.executor.canary_runner import canary_ok
from rp_llm.executor.process_backend import ProcessBackend
from rp_llm.executor.replay_runner import replay_ok


class VerifierPipeline:
    def __init__(self, backend: ProcessBackend | None = None) -> None:
        self.backend = backend or ProcessBackend()

    def _load_candidate_controller(self, candidate_path: Path):
        spec = importlib.util.spec_from_file_location(f"candidate_{candidate_path.stem}", candidate_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("Unable to create module spec from candidate file")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "build_controller"):
            raise RuntimeError("Candidate module must define build_controller()")
        controller = module.build_controller()
        return controller

    def load_candidate_controller(self, candidate_path: str):
        return self._load_candidate_controller(Path(candidate_path))

    def _interface_ok(self, controller) -> bool:
        if not hasattr(controller, "action"):
            return False
        probe = controller.action({"dx": 0.5, "dy": -0.5, "distance": 0.7})
        return isinstance(probe, tuple) and len(probe) == 2 and all(isinstance(v, float) for v in probe)

    def _unit_checks(self, controller) -> bool:
        obs = {"dx": 1.0, "dy": 1.0, "distance": 1.0}
        ax, ay = controller.action(obs)
        # Bounded actuator command for safe exploration.
        return abs(ax) <= 1.0 and abs(ay) <= 1.0

    def _run_eval(self, controller, seeds: list[int], max_steps: int, noise: float, goal: tuple[float, float], disturbance: tuple[float, float]) -> list[list[dict]]:
        episodes = []
        for seed in seeds:
            episodes.append(
                self.backend.run_episode(
                    controller,
                    max_steps=max_steps,
                    noise=noise,
                    goal=goal,
                    disturbance=disturbance,
                    seed=seed,
                )
            )
        return episodes

    def _source_policy_ok(self, source_after: str) -> bool:
        forbidden_tokens = [
            "import requests",
            "import socket",
            "import subprocess",
            "os.system",
            "safety_filter",
            "rollback",
            "verifier",
            "logging",
        ]
        return all(token not in source_after for token in forbidden_tokens)

    def run(self, candidate: PatchCandidate, profile: dict) -> VerificationBundle:
        syntax_ok = False
        type_ok = False
        unit_ok = False
        replay_check = False
        safety_ok = False
        regression_ok = False
        canary_check = False
        details: dict = {}
        fail_summary = None
        candidate_path = Path(candidate.candidate_path)
        try:
            ast.parse(candidate.source_after)
            compile(candidate.source_after, filename=str(candidate_path), mode="exec")
            syntax_ok = True
            policy_ok = self._source_policy_ok(candidate.source_after)
            if not policy_ok:
                raise RuntimeError("candidate source violates patch policy")
            controller = self._load_candidate_controller(candidate_path)
            type_ok = self._interface_ok(controller)
            unit_ok = self._unit_checks(controller)
            replay_episodes = self._run_eval(
                controller,
                seeds=[101, 102, 103, 104],
                max_steps=profile["max_steps"],
                noise=profile["noise"],
                goal=tuple(profile["goal"]),
                disturbance=tuple(profile["disturbance"]),
            )
            replay_check = all(replay_ok(ep) for ep in replay_episodes)
            heldout = self._run_eval(
                controller,
                seeds=[201, 202, 203, 204],
                max_steps=profile["max_steps"],
                noise=profile["noise"],
                goal=tuple(profile["goal"]),
                disturbance=tuple(profile["disturbance"]),
            )
            canary_episodes = self._run_eval(
                controller,
                seeds=[301, 302],
                max_steps=profile["max_steps"],
                noise=profile["noise"],
                goal=tuple(profile["goal"]),
                disturbance=tuple(profile["disturbance"]),
            )
            safety_ok = all(sum(1 for e in ep if e.get("unsafe", False)) == 0 for ep in replay_episodes + heldout)
            regression_ok = all(ep and ep[-1].get("distance", 1.0) < 0.2 for ep in heldout)
            canary_check = all(canary_ok(ep, max_unsafe_ratio=0.0) for ep in canary_episodes)
            details = {
                "replay_episode_count": len(replay_episodes),
                "heldout_episode_count": len(heldout),
                "canary_episode_count": len(canary_episodes),
            }
        except Exception as exc:  # pragma: no cover - defensive at runtime.
            fail_summary = f"verification_exception:{exc}"

        checks = [syntax_ok, type_ok, unit_ok, replay_check, safety_ok, regression_ok, canary_check]
        if fail_summary is None and not all(checks):
            fail_summary = "verification_failed"
        vhash = hashlib.sha256(candidate.source_after.encode("utf-8")).hexdigest()[:16]
        return VerificationBundle(
            candidate_id=candidate.candidate_id,
            syntax_ok=syntax_ok,
            type_ok=type_ok,
            unit_ok=unit_ok,
            replay_ok=replay_check,
            safety_ok=safety_ok,
            regression_ok=regression_ok,
            canary_ok=canary_check,
            fail_summary=fail_summary,
            verifier_hash=vhash,
            details=details,
        )
