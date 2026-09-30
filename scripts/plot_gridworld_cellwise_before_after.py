from __future__ import annotations

from collections import defaultdict
import importlib.util
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


ARROWS = {0: "↑", 1: "↓", 2: "←", 3: "→"}


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _train_q(module_dir: Path, seed: int, episodes: int):
    sys.path.insert(0, str(module_dir))
    try:
        sys.modules.pop("gridworld", None)
        sys.modules.pop("pa2", None)
        gridworld_module = _load_module(f"gridworld_{module_dir.name}", module_dir / "gridworld.py")
        sys.modules["gridworld"] = gridworld_module
        pa2_module = _load_module(f"pa2_{module_dir.name}", module_dir / "pa2.py")
        env = gridworld_module.GridWorld()
        np.random.seed(seed)
        q_td, returns = pa2_module.sarsa_td_control(env, episodes=episodes, gamma=0.99, max_steps=100)
        return env, pa2_module, q_td, returns
    finally:
        sys.path = [p for p in sys.path if p != str(module_dir)]
        sys.modules.pop("gridworld", None)
        sys.modules.pop("pa2", None)


def _route_and_visits(env, pa2_module, q_values, eval_episodes: int = 100):
    route = [env.reset()]
    state = route[0]
    for _ in range(30):
        action = pa2_module.greedy_action(q_values, state)
        nxt = env.next_state(state, action)
        route.append(nxt)
        if nxt == state or nxt == env.goal_state:
            break
        state = nxt

    visits: defaultdict[tuple[int, int], int] = defaultdict(int)
    for _ in range(eval_episodes):
        s = env.reset()
        visits[s] += 1
        for _ in range(30):
            a = pa2_module.greedy_action(q_values, s)
            s_next = env.next_state(s, a)
            visits[s_next] += 1
            if s_next == s or s_next == env.goal_state:
                break
            s = s_next
    return route, visits


def _state_tables(env, pa2_module, q_values):
    h, w = env.height, env.width
    values = np.full((h, w), np.nan)
    policy = np.full((h, w), -1, dtype=int)
    for y in range(h):
        for x in range(w):
            s = (y, x)
            if s == env.wall_state:
                continue
            if getattr(env, "extra_wall_state", None) == s:
                continue
            if s == env.goal_state:
                values[y, x] = 1.0
                policy[y, x] = -2
                continue
            if tuple(s) == (1, 3):
                values[y, x] = -1.0
                policy[y, x] = -3
                continue
            qs = [q_values[(s, a)] for a in [0, 1, 2, 3]]
            best = int(np.argmax(qs))
            values[y, x] = float(qs[best])
            policy[y, x] = best
    return values, policy


def _draw_grid(ax, env, values, policy, route, visits, title: str):
    h, w = env.height, env.width
    shown_values = np.nan_to_num(values, nan=0.0)
    ax.imshow(shown_values, cmap="YlGnBu", vmin=np.nanmin(shown_values), vmax=np.nanmax(shown_values))
    ax.set_xticks(range(w))
    ax.set_yticks(range(h))
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_title(title)
    ax.grid(True, color="#555", linewidth=1.0)

    for y in range(h):
        for x in range(w):
            s = (y, x)
            if s == env.wall_state or getattr(env, "extra_wall_state", None) == s:
                ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color="#666"))
                ax.text(x, y, "W", ha="center", va="center", color="white", fontsize=11, fontweight="bold")
                continue
            if s == env.goal_state:
                ax.text(x, y - 0.12, "G", ha="center", va="center", color="darkred", fontsize=11, fontweight="bold")
            elif s == (1, 3):
                ax.text(x, y - 0.12, "B", ha="center", va="center", color="black", fontsize=11, fontweight="bold")
            elif s == env.start_state:
                ax.text(x, y - 0.12, "S", ha="center", va="center", color="navy", fontsize=11, fontweight="bold")

            if policy[y, x] >= 0:
                ax.text(x, y + 0.10, ARROWS[policy[y, x]], ha="center", va="center", fontsize=12, color="black")

            if not np.isnan(values[y, x]):
                ax.text(x, y + 0.32, f"V={values[y, x]:.2f}", ha="center", va="center", fontsize=8, color="black")
            ax.text(x, y + 0.44, f"N={visits.get(s, 0)}", ha="center", va="center", fontsize=8, color="black")

    route_xy = [(p[1], p[0]) for p in route]
    if len(route_xy) >= 2:
        xs, ys = zip(*route_xy)
        ax.plot(xs, ys, color="crimson", linewidth=2.2, marker="o", markersize=4)
    ax.set_xticklabels([])
    ax.set_yticklabels([])


def _draw_delta(ax, env, before_values, after_values, title: str):
    delta = after_values - before_values
    filled = np.nan_to_num(delta, nan=0.0)
    vmax = max(abs(np.nanmin(filled)), abs(np.nanmax(filled)), 1e-6)
    im = ax.imshow(filled, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    h, w = env.height, env.width
    ax.set_xticks(range(w))
    ax.set_yticks(range(h))
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.grid(True, color="#555", linewidth=1.0)
    ax.set_title(title)
    for y in range(h):
        for x in range(w):
            s = (y, x)
            if s == env.wall_state or getattr(env, "extra_wall_state", None) == s:
                ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color="#666"))
                continue
            if not np.isnan(delta[y, x]):
                ax.text(x, y, f"{delta[y, x]:+.2f}", ha="center", va="center", fontsize=10, color="black")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    return im


def main() -> None:
    root = Path("artifacts/qwen_problem_solving_demo")
    plot_dir = root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir = root / "baseline"
    modified_dir = root / "modified"

    env_b, pa2_b, q_b, ret_b = _train_q(baseline_dir, seed=0, episodes=1000)
    env_a, pa2_a, q_a, ret_a = _train_q(modified_dir, seed=0, episodes=1000)
    route_b, visits_b = _route_and_visits(env_b, pa2_b, q_b, eval_episodes=120)
    route_a, visits_a = _route_and_visits(env_a, pa2_a, q_a, eval_episodes=120)
    values_b, policy_b = _state_tables(env_b, pa2_b, q_b)
    values_a, policy_a = _state_tables(env_a, pa2_a, q_a)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    _draw_grid(axes[0], env_b, values_b, policy_b, route_b, visits_b, "Before: Weak Solver")
    _draw_grid(axes[1], env_a, values_a, policy_a, route_a, visits_a, "After: Qwen-Improved Solver")
    plt.tight_layout()
    plt.savefig(plot_dir / "grid_policy_route_before_after.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    im = _draw_delta(axes[0], env_b, values_b, values_a, "Cell Value Delta (After - Before)")
    axes[1].plot(ret_b, label="before", alpha=0.8)
    axes[1].plot(ret_a, label="after", alpha=0.8)
    axes[1].set_title("Episode Return Curve (seed=0)")
    axes[1].set_xlabel("episode")
    axes[1].set_ylabel("discounted return")
    axes[1].legend()
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(plot_dir / "grid_value_delta_and_learning_curve.png", dpi=160)
    plt.close(fig)

    print(
        {
            "saved": [
                str(plot_dir / "grid_policy_route_before_after.png"),
                str(plot_dir / "grid_value_delta_and_learning_curve.png"),
            ]
        }
    )


if __name__ == "__main__":
    main()

