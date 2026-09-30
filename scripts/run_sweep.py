from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_SCENARIOS = [
    "configs/scenarios/scenario_keepout_01.yaml",
    "configs/scenarios/scenario_battery_tight.yaml",
    "configs/scenarios/scenario_obstacle_dense.yaml",
]

DEFAULT_PATCHERS = ["deterministic", "llm"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS)
    parser.add_argument("--patchers", nargs="+", default=DEFAULT_PATCHERS)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--artifacts-root", default="artifacts/toy_grid_patch")

    parser.add_argument("--perturb", action="store_true", help="Per-run obstacle/keepout perturbation.")
    parser.add_argument("--perturb-seed-base", type=int, default=7000)
    parser.add_argument("--astar-tiebreak", action="store_true", help="Random A* tiebreak per run.")
    parser.add_argument("--astar-seed-base", type=int, default=9000)

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

    root = Path(args.artifacts_root)
    summary = {"sweeps": []}

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", "src")

    for scenario in args.scenarios:
        for patcher in args.patchers:
            cmd = [
                sys.executable,
                "scripts/run_patch_cycle.py",
                "--scenario",
                scenario,
                "--patcher",
                patcher,
                "--runs",
                str(args.runs),
                "--artifacts-root",
                args.artifacts_root,
            ]
            if args.perturb:
                cmd += ["--perturb", "--perturb-seed-base", str(args.perturb_seed_base)]
            if args.astar_tiebreak:
                cmd += ["--astar-tiebreak", "--astar-seed-base", str(args.astar_seed_base)]
            if patcher == "llm":
                cmd += [
                    "--llm-backend",
                    args.llm_backend,
                    "--llm-model-path",
                    args.llm_model_path,
                    "--llm-temperature",
                    str(args.llm_temperature),
                    "--llm-top-p",
                    str(args.llm_top_p),
                    "--llm-max-new-tokens",
                    str(args.llm_max_new_tokens),
                    "--llm-seed-base",
                    str(args.llm_seed_base),
                ]
                if args.llm_jitter:
                    cmd.append("--llm-jitter")

            print(f"\n>>> Running: {' '.join(cmd)}", flush=True)
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
            sys.stdout.write(result.stdout)
            sys.stderr.write(result.stderr)
            summary["sweeps"].append(
                {
                    "scenario": scenario,
                    "patcher": patcher,
                    "runs": args.runs,
                    "perturb": bool(args.perturb),
                    "astar_tiebreak": bool(args.astar_tiebreak),
                    "llm_backend": args.llm_backend if patcher == "llm" else None,
                    "llm_jitter": args.llm_jitter and patcher == "llm",
                }
            )

    root.mkdir(parents=True, exist_ok=True)
    (root / "sweep_manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nSweep complete. Manifest: {root / 'sweep_manifest.json'}")


if __name__ == "__main__":
    main()
