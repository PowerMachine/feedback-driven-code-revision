from __future__ import annotations

import argparse
from pathlib import Path
import importlib.util
import json
import shutil
import sys

import numpy as np

from rp_llm.llm.client import QwenLocalClient


def _extract_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)
    start = text.find("{")
    if start == -1:
        raise ValueError(f"Could not find JSON object in model output: {text[:400]}")
    decoder = json.JSONDecoder()
    parsed, _ = decoder.raw_decode(text[start:])
    return parsed


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _prepare_module_dir(target_dir: Path, gridworld_src: str, pa2_src: str) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "gridworld.py").write_text(gridworld_src, encoding="utf-8")
    (target_dir / "pa2.py").write_text(pa2_src, encoding="utf-8")
    (target_dir / "gridworld_render.py").write_text(
        "\n".join(
            [
                "class Renderer:",
                "    def __init__(self, *args, **kwargs):",
                "        pass",
                "    def render_v(self, *args, **kwargs):",
                "        return None",
                "    def render_q(self, *args, **kwargs):",
                "        return None",
            ]
        ),
        encoding="utf-8",
    )


def _greedy_route(pa2_module, env, q_values: dict, max_steps: int = 20) -> list[tuple[int, int]]:
    route = [env.reset()]
    state = route[0]
    for _ in range(max_steps):
        action = pa2_module.greedy_action(q_values, state)
        next_state = env.next_state(state, action)
        route.append(next_state)
        if next_state == state:
            break
        state = next_state
        if state == env.goal_state:
            break
    return route


def _evaluate_variant(module_dir: Path, module_tag: str) -> dict:
    sys.path.insert(0, str(module_dir))
    try:
        sys.modules.pop("gridworld", None)
        sys.modules.pop("pa2", None)
        gridworld_module = _load_module(f"gridworld_{module_tag}", module_dir / "gridworld.py")
        sys.modules["gridworld"] = gridworld_module
        pa2_module = _load_module(f"pa2_{module_tag}", module_dir / "pa2.py")
        env = gridworld_module.GridWorld()
        np.random.seed(0)
        q_td, returns_td = pa2_module.sarsa_td_control(env, episodes=1500, gamma=0.99, alpha=0.15, max_steps=100)
        policy_ascii = pa2_module.ascii_policy(q_td, env)
        route = _greedy_route(pa2_module, env, q_td)
        return {
            "ascii_policy": policy_ascii,
            "route": route,
            "route_length": len(route) - 1,
            "start_state": env.start_state,
            "goal_state": env.goal_state,
            "wall_state": getattr(env, "wall_state", None),
            "extra_blocked": getattr(env, "extra_wall_state", None),
            "final_return_mean_last100": float(np.mean(returns_td[-100:])),
        }
    finally:
        sys.path = [p for p in sys.path if p != str(module_dir)]
        sys.modules.pop("gridworld", None)
        sys.modules.pop("pa2", None)


def _build_prompt(gridworld_src: str, pa2_src: str, obstacle: tuple[int, int]) -> str:
    return "\n".join(
        [
            "You are editing a reinforcement-learning GridWorld homework baseline.",
            "Return STRICT JSON only. No markdown fences.",
            f"Task: modify the provided gridworld so that one additional obstacle is added at cell {obstacle}, which lies on the original route from start to goal.",
            "Keep the original API as much as possible.",
            "Update only what is necessary for the environment to respect the new obstacle.",
            "Do not change reward semantics, start_state, goal_state, or action definitions.",
            f"You may add `extra_wall_state = {obstacle}` and adjust transition blocking logic.",
            "Return JSON with keys:",
            "- change_summary: short string",
            "- updated_gridworld_py: full runnable python for gridworld.py",
            "- expected_effect: short string",
            "Reference gridworld.py:",
            gridworld_src,
            "Reference pa2.py:",
            pa2_src,
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--obstacle-row", type=int, default=1)
    parser.add_argument("--obstacle-col", type=int, default=0)
    args = parser.parse_args()
    obstacle = (args.obstacle_row, args.obstacle_col)

    root = Path("artifacts/qwen_reference_gridworld")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    ref_dir = Path("Reference/Gridworld")
    gridworld_src = (ref_dir / "gridworld.py").read_text(encoding="utf-8")
    pa2_src = (ref_dir / "pa2.py").read_text(encoding="utf-8")

    baseline_dir = root / "baseline"
    _prepare_module_dir(baseline_dir, gridworld_src, pa2_src)
    baseline = _evaluate_variant(baseline_dir, "baseline")

    client = QwenLocalClient()
    prompt = _build_prompt(gridworld_src, pa2_src, obstacle)
    raw_text = client.generate_text(prompt, max_new_tokens=2200)
    parsed = _extract_json_object(raw_text)
    updated_gridworld = parsed["updated_gridworld_py"]

    modified_dir = root / "modified"
    _prepare_module_dir(modified_dir, updated_gridworld, pa2_src)
    modified = _evaluate_variant(modified_dir, "modified")

    summary = {
        "model": "Qwen3-Coder-30B-A3B-Instruct",
        "requested_obstacle": obstacle,
        "change_summary": parsed.get("change_summary", ""),
        "expected_effect": parsed.get("expected_effect", ""),
        "baseline": baseline,
        "modified": modified,
        "delta": {
            "route_length": modified["route_length"] - baseline["route_length"],
            "last100_return_mean": modified["final_return_mean_last100"] - baseline["final_return_mean_last100"],
        },
    }

    (root / "prompt.txt").write_text(prompt, encoding="utf-8")
    (root / "qwen_raw_response.txt").write_text(raw_text, encoding="utf-8")
    (root / "baseline" / "gridworld.py").write_text(gridworld_src, encoding="utf-8")
    (root / "modified" / "gridworld.py").write_text(updated_gridworld, encoding="utf-8")
    (root / "result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
