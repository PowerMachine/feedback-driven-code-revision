from __future__ import annotations

import argparse
from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-subdir", default="prompt_robustness")
    parser.add_argument("--output-file", default="prompt_robustness.png")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = Path("artifacts/model_benchmark_coders")
    run_root = root / args.input_subdir
    plots = run_root / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    payload = json.loads((run_root / "benchmark_results.json").read_text(encoding="utf-8"))
    df = pd.DataFrame(payload["results"])
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        raise RuntimeError("No successful rows found in benchmark results.")

    grouped = (
        ok.groupby(["model", "prompt_variant"], as_index=False)
        .agg(
            pass_rate_mean=("accuracy_score", "mean"),
            pass_rate_std=("accuracy_score", "std"),
            gen_time_mean=("generate_seconds", "mean"),
            gen_time_std=("generate_seconds", "std"),
        )
        .fillna(0.0)
    )

    pivot_pass_mean = grouped.pivot(index="model", columns="prompt_variant", values="pass_rate_mean")
    pivot_pass_std = grouped.pivot(index="model", columns="prompt_variant", values="pass_rate_std")
    pivot_gen_mean = grouped.pivot(index="model", columns="prompt_variant", values="gen_time_mean")
    pivot_gen_std = grouped.pivot(index="model", columns="prompt_variant", values="gen_time_std")

    prompt_variants = sorted(ok["prompt_variant"].unique())
    x = range(len(pivot_pass_mean.index))
    width = 0.22

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for idx, pv in enumerate(prompt_variants):
        offset = (idx - (len(prompt_variants) - 1) / 2) * width
        xs = [v + offset for v in x]
        axes[0].bar(
            xs,
            pivot_pass_mean[pv],
            yerr=pivot_pass_std[pv],
            width=width,
            capsize=3,
            label=pv,
        )
        axes[1].bar(
            xs,
            pivot_gen_mean[pv],
            yerr=pivot_gen_std[pv],
            width=width,
            capsize=3,
            label=pv,
        )

    axes[0].set_title("Pass Rate by Prompt Variant")
    axes[0].set_ylabel("pass rate")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(pivot_pass_mean.index, rotation=20, ha="right")

    axes[1].set_title("Generation Time by Prompt Variant")
    axes[1].set_ylabel("seconds")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(pivot_gen_mean.index, rotation=20, ha="right")
    axes[1].legend(title="prompt")
    plt.tight_layout()
    out_file = plots / args.output_file
    plt.savefig(out_file, dpi=160)
    plt.close(fig)

    grouped.to_csv(run_root / "prompt_robustness_summary.csv", index=False)
    print({"saved": [str(out_file), str(run_root / "prompt_robustness_summary.csv")]})


if __name__ == "__main__":
    main()

