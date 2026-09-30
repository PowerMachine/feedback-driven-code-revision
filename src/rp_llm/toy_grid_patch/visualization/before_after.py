from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from rp_llm.toy_grid_patch.visualization.draw_grid import draw_single, draw_cost_heatmap


def save_before_after(path: str | Path, env, baseline_path, patched_path, *, subtitle: str | None = None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    draw_single(axes[0], env, baseline_path, "Baseline path")
    draw_single(axes[1], env, patched_path, "Patched path")
    if subtitle:
        fig.suptitle(subtitle, fontsize=11)
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def save_cost_heatmaps(
    path: str | Path,
    env,
    baseline_rules: dict,
    patched_rules: dict,
    *,
    subtitle: str | None = None,
    baseline_path: list | None = None,
    patched_path: list | None = None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    draw_cost_heatmap(axes[0], env, baseline_rules, "Baseline cost + path", path=baseline_path)
    draw_cost_heatmap(axes[1], env, patched_rules, "Patched cost + path", path=patched_path)
    if subtitle:
        fig.suptitle(subtitle, fontsize=11)
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
