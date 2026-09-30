from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from rp_llm.patching.diff_utils import to_markdown


def _scenario_dir(scenario: str) -> Path:
    out = Path("artifacts/reports") / scenario
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_summary_json(scenario: str, payload: dict) -> None:
    (_scenario_dir(scenario) / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_comparison_csv(scenario: str, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(_scenario_dir(scenario) / "comparison_metrics.csv", index=False)


def write_patch_diff(scenario: str, before: str, after: str) -> None:
    (_scenario_dir(scenario) / "patch_diff.md").write_text(to_markdown(before, after), encoding="utf-8")


def plot_trajectory_overlay(scenario: str, before: list[tuple[float, float]], after: list[tuple[float, float]]) -> None:
    plt.figure(figsize=(5, 5))
    if before:
        bx, by = zip(*before)
        plt.plot(bx, by, label="before")
    if after:
        ax, ay = zip(*after)
        plt.plot(ax, ay, label="after")
    plt.legend()
    plt.title("Trajectory Overlay")
    plt.savefig(_scenario_dir(scenario) / "trajectory_overlay.png")
    plt.close()


def plot_distance_before_after(scenario: str, before_events: list[dict], after_events: list[dict]) -> None:
    plt.figure(figsize=(8, 3))
    bx = list(range(len(before_events)))
    ax = list(range(len(after_events)))
    plt.plot(bx, [row.get("distance", 0.0) for row in before_events], label="before_distance")
    plt.plot(ax, [row.get("distance", 0.0) for row in after_events], label="after_distance")
    plt.legend()
    plt.title("Distance To Goal (Before vs After)")
    plt.xlabel("step")
    plt.ylabel("distance")
    plt.savefig(_scenario_dir(scenario) / "distance_before_after.png")
    plt.close()


def plot_critique_timeline(scenario: str, axis_rows: list[dict]) -> None:
    plt.figure(figsize=(8, 3))
    x = list(range(len(axis_rows)))
    plt.plot(x, [row.get("safety", 0.0) for row in axis_rows], label="safety")
    plt.plot(x, [row.get("performance", 0.0) for row in axis_rows], label="performance")
    plt.legend()
    plt.title("Critique Timeline")
    plt.savefig(_scenario_dir(scenario) / "critique_timeline.png")
    plt.close()


def plot_lineage_graph(scenario: str, edges: list[tuple[str, str]]) -> None:
    g = nx.DiGraph()
    g.add_edges_from(edges)
    plt.figure(figsize=(6, 4))
    nx.draw_networkx(g, with_labels=True, node_size=900, font_size=8)
    plt.savefig(_scenario_dir(scenario) / "lineage_graph.png")
    plt.close()


def write_casebook_page(scenario: str, lines: list[str]) -> None:
    (_scenario_dir(scenario) / "casebook_page.md").write_text("\n".join(lines), encoding="utf-8")


def write_before_after_video_placeholder(scenario: str) -> None:
    # ffmpeg/imageio integration is left as a TODO hook.
    (_scenario_dir(scenario) / "before_after.mp4").write_bytes(b"")
