from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt


def _load_result(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _plot_summary_bars(result: dict, out_dir: Path) -> None:
    before = result["before"]
    after = result["after"]
    labels = ["Last100 Mean", "Learning Gain", "Success Rate", "Route Length Mean"]
    before_values = [
        before["last100_mean"],
        before["learning_gain_mean"],
        before["success_rate"],
        before["route_length_mean"],
    ]
    after_values = [
        after["last100_mean"],
        after["learning_gain_mean"],
        after["success_rate"],
        after["route_length_mean"],
    ]

    x = list(range(len(labels)))
    width = 0.35
    plt.figure(figsize=(10, 4))
    plt.bar([xi - width / 2 for xi in x], before_values, width=width, label="before")
    plt.bar([xi + width / 2 for xi in x], after_values, width=width, label="after")
    plt.xticks(x, labels, rotation=15)
    plt.title("Qwen Problem-Solving Demo: Before vs After")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "summary_before_after.png")
    plt.close()


def _plot_per_seed(result: dict, out_dir: Path) -> None:
    before_rows = result["before"]["seed_metrics"]
    after_rows = result["after"]["seed_metrics"]
    seeds = list(range(len(before_rows)))

    before_last = [row["last100_mean"] for row in before_rows]
    after_last = [row["last100_mean"] for row in after_rows]
    before_gain = [row["gain"] for row in before_rows]
    after_gain = [row["gain"] for row in after_rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(seeds, before_last, marker="o", label="before")
    axes[0].plot(seeds, after_last, marker="o", label="after")
    axes[0].set_title("Per-Seed Last100 Mean")
    axes[0].set_xlabel("seed index")
    axes[0].set_ylabel("value")
    axes[0].legend()

    axes[1].plot(seeds, before_gain, marker="o", label="before")
    axes[1].plot(seeds, after_gain, marker="o", label="after")
    axes[1].set_title("Per-Seed Learning Gain")
    axes[1].set_xlabel("seed index")
    axes[1].set_ylabel("value")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_dir / "per_seed_trends.png")
    plt.close()


def main() -> None:
    root = Path("artifacts/qwen_problem_solving_demo")
    out_dir = root / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = _load_result(root / "result.json")
    _plot_summary_bars(result, out_dir)
    _plot_per_seed(result, out_dir)
    print({"saved": [str(out_dir / "summary_before_after.png"), str(out_dir / "per_seed_trends.png")]})


if __name__ == "__main__":
    main()

