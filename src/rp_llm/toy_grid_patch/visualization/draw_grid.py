from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def draw_single(ax, env, path, title: str, annotate_steps: bool = True) -> None:
    rows, cols = env.grid_size
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    ax.grid(True, linewidth=0.4)
    ax.set_title(title)
    ax.set_aspect("equal")

    for r, c in env.obstacles:
        ax.add_patch(
            plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor="black", edgecolor="black", alpha=0.85)
        )
    for r, c in env.keepout_zones:
        ax.add_patch(
            plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor="orange", edgecolor="darkorange", alpha=0.35)
        )

    sr, sc = env.start
    gr, gc = env.goal
    ax.scatter([sc], [sr], color="green", s=120, zorder=5, label="Start")
    ax.scatter([gc], [gr], color="red", s=120, zorder=5, label="Goal")
    ax.text(sc, sr, "S", color="white", ha="center", va="center", fontsize=9, fontweight="bold", zorder=6)
    ax.text(gc, gr, "G", color="white", ha="center", va="center", fontsize=9, fontweight="bold", zorder=6)

    if path:
        xs = [c for r, c in path]
        ys = [r for r, c in path]
        ax.plot(xs, ys, color="blue", linewidth=2.2, alpha=0.9, zorder=3)
        ax.scatter(xs, ys, color="blue", s=18, zorder=4)
        if annotate_steps:
            for i, (r, c) in enumerate(path):
                ax.text(c + 0.18, r + 0.18, str(i), color="navy", fontsize=6.5, zorder=6)

    proxies = [
        Patch(facecolor="black", alpha=0.85, label="Obstacle"),
        Patch(facecolor="orange", alpha=0.35, label="Keepout"),
    ]
    existing = ax.get_legend_handles_labels()
    handles = list(existing[0]) + proxies
    labels = list(existing[1]) + [p.get_label() for p in proxies]
    ax.legend(handles, labels, loc="upper right", fontsize=7, framealpha=0.9)


def draw_cost_heatmap(ax, env, rules_snapshot, title: str, path=None) -> None:
    """Render per-cell cost contribution using a rules snapshot; optionally overlay a path."""
    rows, cols = env.grid_size
    grid = [[0.0 for _ in range(cols)] for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            cell = (r, c)
            if cell in env.obstacles:
                grid[r][c] = float("nan")
                continue
            cost = float(rules_snapshot.get("MOVE_COST", 1))
            if cell in env.keepout_zones:
                cost += float(rules_snapshot.get("KEEP_OUT_PENALTY", 0))
            if any(
                (r + dr, c + dc) in env.obstacles
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
            ):
                cost += float(rules_snapshot.get("OBSTACLE_NEAR_PENALTY", 0))
            grid[r][c] = cost * float(rules_snapshot.get("RISK_WEIGHT", 1.0))
    im = ax.imshow(grid, origin="upper", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    for r, c in env.obstacles:
        ax.add_patch(
            plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor="black", edgecolor="black", alpha=0.9)
        )
    if path:
        xs = [c for r, c in path]
        ys = [r for r, c in path]
        ax.plot(xs, ys, color="red", linewidth=2.2, alpha=0.95, zorder=5)
        ax.scatter(xs, ys, color="red", s=20, zorder=6)
        sr, sc = path[0]
        gr, gc = path[-1]
        ax.scatter([sc], [sr], color="lime", s=80, edgecolors="black", zorder=7, label="Start")
        ax.scatter([gc], [gr], color="magenta", s=80, edgecolors="black", zorder=7, label="Goal")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="per-cell cost")
