from __future__ import annotations

from uuid import uuid4

from rp_llm.archive.registry import ArchiveRegistry
from rp_llm.baselines.modes import heuristic_only, rp_llm_full, static_gold, switch_only
from rp_llm.critic.critique import SixAxisCritic
from rp_llm.common.schemas import PatchCandidate, RuntimeEvent
from rp_llm.executor.process_backend import ProcessBackend
from rp_llm.patching.diff_utils import summarize_diff
from rp_llm.patching.llm_planner import LLMPatchPlanner
from rp_llm.policies.gold_controller import ConservativeVariant, GoldController, HighNoiseVariant, LowLightVariant
from rp_llm.verifier.pipeline import VerifierPipeline
from rp_llm.visualization.reporting import (
    plot_distance_before_after,
    plot_critique_timeline,
    plot_lineage_graph,
    plot_trajectory_overlay,
    write_before_after_video_placeholder,
    write_casebook_page,
    write_comparison_csv,
    write_patch_diff,
    write_summary_json,
)

# TODO: optional gpt-oss reviewer integration for secondary plan vetting.
# TODO: future C++ plugin runtime adapter for low-latency execution path.
# TODO: future hardware-in-the-loop execution backend.


def _scenario_profile(scenario: str) -> dict:
    profiles = {
        "simple_nav": {
            "episodes": 24,
            "max_steps": 120,
            "noise": 0.02,
            "goal": (1.0, 1.0),
            "disturbance": (0.0, 0.0),
        },
        "simple_nav_shift": {
            "episodes": 36,
            "max_steps": 160,
            "noise": 0.08,
            "goal": (1.2, 0.8),
            "disturbance": (0.015, -0.01),
        },
        "test_smoke": {
            "episodes": 6,
            "max_steps": 80,
            "noise": 0.03,
            "goal": (1.0, 1.0),
            "disturbance": (0.0, 0.0),
        },
    }
    return profiles.get(scenario, profiles["simple_nav"])


def _run_trials(backend: ProcessBackend, controller, profile: dict, seed_base: int) -> list[list[dict]]:
    episodes: list[list[dict]] = []
    for idx in range(profile["episodes"]):
        episodes.append(
            backend.run_episode(
                controller,
                max_steps=profile["max_steps"],
                noise=profile["noise"],
                goal=profile["goal"],
                disturbance=profile["disturbance"],
                seed=seed_base + idx,
            )
        )
    return episodes


def _trajectory(events: list[dict]) -> list[tuple[float, float]]:
    return [(0.0, 0.0)] + [(float(e.get("x", 0.0)), float(e.get("y", 0.0))) for e in events]


def _representative_episode(episodes: list[list[dict]]) -> list[dict]:
    if not episodes:
        return []
    with_distance = [ep for ep in episodes if ep]
    if not with_distance:
        return []
    ranked = sorted(with_distance, key=lambda ep: ep[-1].get("distance", 999.0))
    return ranked[len(ranked) // 2]


def _as_event(event: str, variant_id: str, message: str, payload: dict | None = None) -> RuntimeEvent:
    return RuntimeEvent(event=event, variant_id=variant_id, message=message, payload=payload or {})


def _source_before_stub() -> str:
    return "\n".join(
        [
            "class ConservativeVariant:",
            "    def action(self, obs: dict) -> tuple[float, float]:",
            "        return obs['dx'] * 0.12, obs['dy'] * 0.12",
            "",
        ]
    )


def run_pipeline(scenario: str = "simple_nav") -> dict:
    profile = _scenario_profile(scenario)
    backend = ProcessBackend()
    critic = SixAxisCritic()
    planner = LLMPatchPlanner()
    verifier = VerifierPipeline()
    archive = ArchiveRegistry()
    runtime_events: list[RuntimeEvent] = []

    controllers = {
        "gold": GoldController(),
        "v_conservative": ConservativeVariant(),
        "v_low_light": LowLightVariant(),
        "v_high_noise": HighNoiseVariant(),
    }

    baseline_before = _run_trials(backend, controllers["v_conservative"], profile, seed_base=1000)
    critique = critic.critique(baseline_before[0], target_module="v_conservative")
    accepted = 0
    lineage = [("gold", "gold")]
    before_src = _source_before_stub()
    after_src = before_src
    active_controller = controllers["v_conservative"]
    promotion_status = "not_attempted"

    if critique.suggested_operator != "keep":
        plan = planner.make_plan(critique, source_variant_id="gold", source_before=before_src)
        after_src = planner.synthesize_controller_source(plan)
        candidate_id = uuid4().hex
        candidate_path = planner.write_candidate_file(candidate_id, after_src)
        candidate = PatchCandidate(
            candidate_id=candidate_id,
            parent_variant_id="gold",
            plan=plan,
            source_before=before_src,
            source_after=after_src,
            diff_summary=summarize_diff(before_src, after_src),
            candidate_path=str(candidate_path),
        )
        runtime_events.append(
            _as_event(
                "patch_generated",
                variant_id="candidate",
                message="LLM planner produced bounded controller patch.",
                payload={"candidate_id": candidate_id, "path": str(candidate_path)},
            )
        )
        # New candidates are validated offline; never execute in fast loop directly.
        bundle = verifier.run(candidate, profile=profile)
        if bundle.passed:
            manifest = archive.promote(candidate, bundle)
            accepted = 1
            promotion_status = "promoted"
            lineage.append(("gold", manifest.variant_id))
            runtime_events.append(
                _as_event(
                    "promotion",
                    variant_id=manifest.variant_id,
                    message="Candidate passed full verification and was promoted.",
                    payload={"verifier_hash": bundle.verifier_hash},
                )
            )
            active_controller = verifier.load_candidate_controller(candidate.candidate_path)
            runtime_events.append(
                _as_event(
                    "variant_switch",
                    variant_id=manifest.variant_id,
                    message="Evaluation switched to promoted verified variant.",
                )
            )
        else:
            promotion_status = "verification_failed"
            runtime_events.append(
                _as_event(
                    "verification_failed",
                    variant_id="candidate",
                    message=bundle.fail_summary or "verification_failed",
                    payload=bundle.details,
                )
            )
            runtime_events.append(
                _as_event("rollback", variant_id="gold", message="Candidate rejected; rollback to trusted baseline.")
            )
    else:
        plan = None
        bundle = None
        runtime_events.append(_as_event("variant_switch", variant_id="gold", message="No patch required; KEEP decision."))

    episodes_gold = _run_trials(backend, controllers["gold"], profile, seed_base=1000)
    episodes_switch = _run_trials(backend, controllers["v_conservative"], profile, seed_base=1000)
    episodes_heuristic = _run_trials(backend, controllers["v_high_noise"], profile, seed_base=1000)
    episodes_full = _run_trials(backend, active_controller, profile, seed_base=1000)

    results = [
        static_gold(episodes_gold).__dict__,
        switch_only(episodes_switch).__dict__,
        heuristic_only(episodes_heuristic).__dict__,
        rp_llm_full(episodes_full, accepted_patch_count=accepted).__dict__,
    ]

    by_name = {row["name"]: row for row in results}
    full_result = by_name["rp_llm_full"]
    switch_result = by_name["switch_only"]
    success_gain = full_result["success_rate"] - switch_result["success_rate"]
    distance_delta = switch_result["mean_final_distance"] - full_result["mean_final_distance"]
    reward_delta = full_result["reward_mean"] - switch_result["reward_mean"]

    write_summary_json(
        scenario,
        {
            "scenario": scenario,
            "profile": profile,
            "accepted_patch_count": accepted,
            "verifier_passed": bool(bundle.passed) if bundle else False,
            "promotion_status": promotion_status,
            "episodes": profile["episodes"],
            "critique_summary": critique.edit_recommendation,
            "runtime_events": [event.model_dump() for event in runtime_events],
            "plan": plan.model_dump() if plan else None,
            "verification": bundle.model_dump() if bundle else None,
            "improvement": {
                "success_rate_delta_vs_switch_only": success_gain,
                "final_distance_reduction_vs_switch_only": distance_delta,
                "reward_gain_vs_switch_only": reward_delta,
            },
            "ranking_by_final_distance": sorted(
                [(row["name"], row["mean_final_distance"]) for row in results],
                key=lambda pair: pair[1],
            ),
            "critique": critique.model_dump(),
        },
    )
    write_comparison_csv(scenario, results)
    write_patch_diff(scenario, before_src, after_src)
    axis_rows = [critic.infer_axis(ep).model_dump() for ep in baseline_before[: min(12, len(baseline_before))]]
    plot_critique_timeline(scenario, axis_rows)
    rep_before = _representative_episode(episodes_switch)
    rep_after = _representative_episode(episodes_full)
    before_xy = _trajectory(rep_before)
    after_xy = _trajectory(rep_after)
    plot_trajectory_overlay(scenario, before_xy, after_xy)
    plot_distance_before_after(scenario, rep_before, rep_after)
    plot_lineage_graph(scenario, lineage)
    write_casebook_page(
        scenario,
        [
            "# Casebook",
            f"- scenario: {scenario}",
            f"- episodes: {profile['episodes']}",
            f"- failure_type: {critique.failure_type}",
            f"- operator: {critique.suggested_operator}",
            f"- depth: {critique.suggested_depth}",
            f"- critique_summary: {critique.edit_recommendation}",
            f"- promotion_status: {promotion_status}",
            f"- verifier_passed: {bool(bundle.passed) if bundle else False}",
            f"- success_delta_vs_switch_only: {success_gain:+.3f}",
            f"- final_distance_delta_vs_switch_only: {distance_delta:+.3f}",
            f"- reward_delta_vs_switch_only: {reward_delta:+.3f}",
            f"- runtime_events: {[event.event for event in runtime_events]}",
            "- TODO: optional gpt-oss reviewer integration hook",
            "- TODO: future C++ plugin runtime for low-latency deployment",
            "- TODO: future hardware-in-the-loop execution pipeline",
        ],
    )
    write_before_after_video_placeholder(scenario)
    return {
        "scenario": scenario,
        "accepted_patch_count": accepted,
        "results": results,
        "improvement": {"success_gain": success_gain, "distance_delta": distance_delta, "reward_delta": reward_delta},
    }
