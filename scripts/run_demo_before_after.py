from __future__ import annotations

import argparse

from rp_llm.orchestrator import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="simple_nav")
    args = parser.parse_args()
    result = run_pipeline(args.scenario)
    print(result)


if __name__ == "__main__":
    main()
