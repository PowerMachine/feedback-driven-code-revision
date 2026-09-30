from __future__ import annotations

import argparse
import json
from pathlib import Path

from rp_llm.toy_grid_patch.env.gridworld import GridWorld
from rp_llm.toy_grid_patch.env.scenario_loader import load_scenario
from rp_llm.toy_grid_patch.env.simulator import simulate_path
from rp_llm.toy_grid_patch.planner.planner_astar import plan_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', default='configs/scenarios/scenario_keepout_01.yaml')
    parser.add_argument('--out', default='artifacts/baseline/baseline_result.json')
    args = parser.parse_args()

    scenario = load_scenario(args.scenario)
    env = GridWorld.from_scenario(scenario)
    plan = plan_path(env)
    metrics = simulate_path(env, plan.path, plan.replan_count)

    payload = {
        'scenario': scenario,
        'path': plan.path,
        'total_cost': plan.total_cost,
        'metrics': metrics,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
