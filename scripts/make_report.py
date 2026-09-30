from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

import pandas as pd


def _aggregate_patcher(patcher_dir: Path, scenario_id: str) -> dict | None:
    csv_path = patcher_dir / "reports" / "comparison_metrics.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if df.empty:
        return None
    improved_rate = float(df["improved"].mean())
    keepout_delta = df["baseline_keepout_violations"] - df["patched_keepout_violations"]
    path_delta = df["baseline_path_length"] - df["patched_path_length"]

    records_path = patcher_dir / "reports" / "per_run_records.json"
    patch_signatures: Counter = Counter()
    per_constant_values: dict[str, list] = {}
    fallback_count = 0
    if records_path.exists():
        records = json.loads(records_path.read_text(encoding="utf-8"))
        for rec in records:
            if rec.get("llm_fallback"):
                fallback_count += 1
            payload = rec.get("patch_payload") or {}
            changes = payload.get("changes") or []
            sig_items = tuple(sorted((c.get("name"), c.get("new_value")) for c in changes))
            patch_signatures[json.dumps(sig_items, ensure_ascii=False, default=str)] += 1
            for c in changes:
                name = c.get("name")
                val = c.get("new_value")
                if isinstance(val, (int, float)) and name:
                    per_constant_values.setdefault(name, []).append(float(val))

    const_stats = {
        name: {
            "n": len(vs),
            "mean": float(statistics.mean(vs)) if vs else None,
            "stdev": float(statistics.pstdev(vs)) if len(vs) > 1 else 0.0,
            "min": float(min(vs)) if vs else None,
            "max": float(max(vs)) if vs else None,
        }
        for name, vs in per_constant_values.items()
    }
    unique_sigs = len(patch_signatures)
    total_sigs = sum(patch_signatures.values())

    return {
        "scenario_id": scenario_id,
        "patcher": patcher_dir.name,
        "runs": int(len(df)),
        "improved_rate": improved_rate,
        "keepout_delta_mean": float(keepout_delta.mean()),
        "keepout_delta_stdev": float(keepout_delta.std(ddof=0)) if len(df) > 1 else 0.0,
        "path_delta_mean": float(path_delta.mean()),
        "path_delta_stdev": float(path_delta.std(ddof=0)) if len(df) > 1 else 0.0,
        "baseline_success_rate": float(df["baseline_success"].mean()) if "baseline_success" in df else None,
        "patched_success_rate": float(df["patched_success"].mean()) if "patched_success" in df else None,
        "llm_fallback_count": fallback_count,
        "unique_patch_signatures": unique_sigs,
        "total_patches": total_sigs,
        "signature_diversity": (unique_sigs / total_sigs) if total_sigs else None,
        "per_constant_stats": const_stats,
    }


def _write_markdown(path: Path, aggregates: list[dict]) -> None:
    lines = ["# GridWorld Rule Patch Aggregate Report", ""]
    lines.append(
        "| scenario | patcher | runs | improved_rate | keepout_delta (mean±std) | "
        "path_delta (mean±std) | unique_patches | fallback |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for agg in aggregates:
        uniq = (
            f"{agg['unique_patch_signatures']}/{agg['total_patches']}"
            if agg.get("total_patches")
            else "n/a"
        )
        lines.append(
            "| {sc} | {pa} | {r} | {rate:.2f} | {kd:+.2f}±{kds:.2f} | {pd:+.2f}±{pds:.2f} | {uq} | {fb} |".format(
                sc=agg["scenario_id"],
                pa=agg["patcher"],
                r=agg["runs"],
                rate=agg["improved_rate"],
                kd=agg["keepout_delta_mean"],
                kds=agg["keepout_delta_stdev"],
                pd=agg["path_delta_mean"],
                pds=agg["path_delta_stdev"],
                uq=uniq,
                fb=agg.get("llm_fallback_count", 0),
            )
        )
    lines.append("")
    lines.append(
        "unique_patches = distinct (name,new_value) signatures across runs; "
        "higher = LLM varied its choices more."
    )
    lines.append("")
    lines.append("## LLM choice consistency per constant")
    for agg in aggregates:
        if not agg.get("per_constant_stats"):
            continue
        lines.append(f"### {agg['scenario_id']} / {agg['patcher']}")
        lines.append("| constant | n | mean | stdev | min | max |")
        lines.append("|---|---|---|---|---|---|")
        for name, s in agg["per_constant_stats"].items():
            lines.append(
                "| {n} | {k} | {m:.2f} | {sd:.2f} | {lo:.2f} | {hi:.2f} |".format(
                    n=name,
                    k=s["n"],
                    m=s.get("mean") or 0.0,
                    sd=s.get("stdev") or 0.0,
                    lo=s.get("min") or 0.0,
                    hi=s.get("max") or 0.0,
                )
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="artifacts/toy_grid_patch")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Missing root: {root}")

    aggregates: list[dict] = []
    for scenario_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for patcher_dir in sorted(p for p in scenario_dir.iterdir() if p.is_dir()):
            agg = _aggregate_patcher(patcher_dir, scenario_dir.name)
            if agg:
                aggregates.append(agg)

    if not aggregates:
        raise SystemExit("No patcher run data found under root.")

    (root / "aggregate_summary.json").write_text(
        json.dumps({"aggregates": aggregates}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_markdown(root / "aggregate_report.md", aggregates)
    print(json.dumps({"aggregates": aggregates}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
