from __future__ import annotations

import argparse
import json
from pathlib import Path

from rp_llm.orchestrator import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="simple_nav")
    args = parser.parse_args()
    result = run_pipeline(args.scenario)
    out = Path("artifacts/reports") / args.scenario / "baseline_compare.json"
    out.write_text(json.dumps(result["results"], indent=2), encoding="utf-8")
    print({"saved": str(out)})


if __name__ == "__main__":
    main()
