from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import statistics

import pandas as pd
from rp_llm.toy_grid_patch.env.gridworld import GridWorld
from rp_llm.toy_grid_patch.env.perturb import perturb_scenario
from rp_llm.toy_grid_patch.env.scenario_loader import load_scenario
from rp_llm.toy_grid_patch.env.simulator import simulate_path
from rp_llm.toy_grid_patch.evaluation.compare import comparison_row
from rp_llm.toy_grid_patch.evaluation.explainability import build_explanation, write_explanation_md
from rp_llm.toy_grid_patch.evaluation.metrics import is_improved
from rp_llm.toy_grid_patch.evaluation.report import write_failure_report_json, write_summary_md
from rp_llm.toy_grid_patch.patching.apply_patch import apply_patch, read_rules
from rp_llm.toy_grid_patch.patching.deterministic_patcher import propose_patch
from rp_llm.toy_grid_patch.patching.failure_report import build_failure_report
from rp_llm.toy_grid_patch.patching.llm_patcher import propose_patch_with_llm
from rp_llm.toy_grid_patch.patching.validators import validate_patch_payload
from rp_llm.toy_grid_patch.planner.planner_astar import plan_path
from rp_llm.toy_grid_patch.planner import planner_rules
from rp_llm.toy_grid_patch.visualization.before_after import save_before_after, save_cost_heatmaps
from rp_llm.toy_grid_patch.visualization.patch_diff import write_patch_diff


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="configs/scenarios/scenario_keepout_01.yaml")
    parser.add_argument("--patcher", default="deterministic", choices=["deterministic", "llm"])
    parser.add_argument("--rules-file", default="src/rp_llm/toy_grid_patch/planner/planner_rules.py")
    parser.add_argument("--artifacts-root", default="artifacts/toy_grid_patch")
    parser.add_argument("--runs", type=int, default=1)

    # Variability knobs (3 independent sources).
    parser.add_argument("--perturb", action="store_true", help="Per-run obstacle/keepout perturbation.")
    parser.add_argument("--perturb-seed-base", type=int, default=7000)
    parser.add_argument("--astar-tiebreak", action="store_true", help="Randomize A* tiebreak across equal costs.")
    parser.add_argument("--astar-seed-base", type=int, default=9000)

    # LLM backend.
    parser.add_argument("--llm-backend", default="stub", choices=["stub", "qwen"])
    parser.add_argument(
        "--llm-model-path",
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
    )
    parser.add_argument("--llm-temperature", type=float, default=0.7)
    parser.add_argument("--llm-top-p", type=float, default=0.95)
    parser.add_argument("--llm-max-new-tokens", type=int, default=350)
    parser.add_argument("--llm-seed-base", type=int, default=1000)
    parser.add_argument("--llm-jitter", action="store_true")
    args = parser.parse_args()

    if args.runs < 1:
        raise ValueError("--runs must be >= 1")

    base_scenario = load_scenario(args.scenario)
    rules_path = Path(args.rules_file)
    baseline_rules_text = rules_path.read_text(encoding="utf-8")

    scenario_id = base_scenario["scenario_id"]
    patcher_tag = args.patcher
    if args.patcher == "llm" and args.llm_backend == "qwen":
        patcher_tag = "llm_qwen"
    run_root = Path(args.artifacts_root) / scenario_id / patcher_tag
    (run_root / "runs").mkdir(parents=True, exist_ok=True)
    (run_root / "reports").mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    per_run_records: list[dict] = []

    for run_idx in range(args.runs):
        run_dir = run_root / "runs" / f"run_{run_idx:02d}"
        (run_dir / "figures").mkdir(parents=True, exist_ok=True)
        (run_dir / "reports").mkdir(parents=True, exist_ok=True)

        if args.perturb:
            scenario = perturb_scenario(base_scenario, seed=args.perturb_seed_base + run_idx)
        else:
            scenario = base_scenario
        env = GridWorld.from_scenario(scenario)

        rules_path.write_text(baseline_rules_text, encoding="utf-8")
        importlib.reload(planner_rules)
        baseline_rules = read_rules(args.rules_file)

        tiebreak_seed = (args.astar_seed_base + run_idx) if args.astar_tiebreak else None
        baseline_plan = plan_path(env, tiebreak_seed=tiebreak_seed)
        baseline_metrics = simulate_path(env, baseline_plan.path, baseline_plan.replan_count)
        failure_report = build_failure_report(scenario["scenario_id"], baseline_metrics)

        current_rules = read_rules(args.rules_file)
        llm_payload, llm_trace = None, {}
        if args.patcher == "llm":
            llm_payload, llm_trace = propose_patch_with_llm(
                current_rules,
                scenario,
                baseline_metrics,
                failure_report,
                run_idx=run_idx,
                jitter=args.llm_jitter,
                llm_backend=args.llm_backend,
                llm_model_path=args.llm_model_path,
                llm_temperature=args.llm_temperature,
                llm_top_p=args.llm_top_p,
                llm_max_new_tokens=args.llm_max_new_tokens,
                llm_seed_base=args.llm_seed_base,
            )
        patch_payload = llm_payload if llm_payload else propose_patch(current_rules, baseline_metrics)
        fallback_used = args.patcher == "llm" and llm_payload is None

        valid, reason = validate_patch_payload(patch_payload)
        diff: dict = {}
        if valid and patch_payload.get("changes"):
            diff = apply_patch(args.rules_file, patch_payload)
            importlib.reload(planner_rules)
        else:
            patch_payload = {"changes": [], "expected_effect": f"Patch rejected: {reason}"}

        patched_rules = read_rules(args.rules_file)
        patched_plan = plan_path(env, tiebreak_seed=tiebreak_seed)
        patched_metrics = simulate_path(env, patched_plan.path, patched_plan.replan_count)
        improved = is_improved(baseline_metrics, patched_metrics)

        row = comparison_row(scenario["scenario_id"], baseline_metrics, patched_metrics, improved)
        row["run_idx"] = run_idx
        row["patcher"] = patcher_tag
        row["perturb"] = bool(args.perturb)
        row["astar_tiebreak"] = bool(args.astar_tiebreak)
        row["llm_fallback"] = fallback_used
        rows.append(row)

        subtitle = (
            f"{patcher_tag} | run {run_idx:02d} | improved={improved} | "
            f"keepout {baseline_metrics.get('keepout_violations')}->{patched_metrics.get('keepout_violations')} | "
            f"len {baseline_metrics.get('path_length')}->{patched_metrics.get('path_length')}"
        )
        save_before_after(
            run_dir / "figures" / "before_after_paths.png",
            env,
            baseline_plan.path,
            patched_plan.path,
            subtitle=subtitle,
        )
        save_cost_heatmaps(
            run_dir / "figures" / "cost_heatmap.png",
            env,
            baseline_rules,
            patched_rules,
            subtitle=f"{patcher_tag} | run {run_idx:02d} | cost landscape + path overlay",
            baseline_path=baseline_plan.path,
            patched_path=patched_plan.path,
        )
        write_patch_diff(run_dir / "reports" / "patch_diff.md", diff)
        write_failure_report_json(run_dir / "reports" / "failure_report.json", failure_report)
        write_summary_md(
            run_dir / "reports" / "summary_report.md",
            baseline_metrics,
            patched_metrics,
            patch_payload,
            improved,
        )
        explanation = build_explanation(
            scenario=scenario,
            failure_report=failure_report,
            patch_payload=patch_payload,
            diff=diff,
            baseline_metrics=baseline_metrics,
            patched_metrics=patched_metrics,
            improved=improved,
            validator_reason=reason if not valid else "ok",
        )
        write_explanation_md(run_dir / "reports" / "explanation.md", explanation)

        if llm_trace:
            (run_dir / "reports" / "llm_trace.json").write_text(
                json.dumps({k: v for k, v in llm_trace.items() if k != "raw_output"}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            if llm_trace.get("raw_output"):
                (run_dir / "reports" / "llm_raw_output.txt").write_text(
                    str(llm_trace["raw_output"]), encoding="utf-8"
                )
        (run_dir / "reports" / "scenario_used.json").write_text(
            json.dumps(scenario, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )

        per_run_records.append(
            {
                "run_idx": run_idx,
                "scenario_id": scenario["scenario_id"],
                "improved": improved,
                "validator_reason": reason if not valid else "ok",
                "patch_payload": patch_payload,
                "diff": {k: list(v) for k, v in diff.items()},
                "baseline_metrics": baseline_metrics,
                "patched_metrics": patched_metrics,
                "llm_fallback": fallback_used,
                "llm_seed": llm_trace.get("seed"),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(run_root / "reports" / "comparison_metrics.csv", index=False)

    improved_flags = [bool(r["improved"]) for r in rows]
    keepout_delta = [r["baseline_keepout_violations"] - r["patched_keepout_violations"] for r in rows]
    path_delta = [r["baseline_path_length"] - r["patched_path_length"] for r in rows]
    summary = {
        "patcher": patcher_tag,
        "llm_backend": args.llm_backend if args.patcher == "llm" else None,
        "scenario_id": base_scenario["scenario_id"],
        "perturb": bool(args.perturb),
        "astar_tiebreak": bool(args.astar_tiebreak),
        "runs": args.runs,
        "improved_runs": int(sum(improved_flags)),
        "improved_rate": float(sum(improved_flags) / len(improved_flags)),
        "keepout_delta_mean": float(statistics.mean(keepout_delta)),
        "keepout_delta_stdev": float(statistics.pstdev(keepout_delta)) if len(keepout_delta) > 1 else 0.0,
        "path_delta_mean": float(statistics.mean(path_delta)),
        "path_delta_stdev": float(statistics.pstdev(path_delta)) if len(path_delta) > 1 else 0.0,
        "llm_fallback_count": int(sum(bool(r.get("llm_fallback")) for r in per_run_records)),
    }
    (run_root / "reports" / "aggregate_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_root / "reports" / "per_run_records.json").write_text(
        json.dumps(per_run_records, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    rules_path.write_text(baseline_rules_text, encoding="utf-8")
    importlib.reload(planner_rules)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
