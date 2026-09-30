from __future__ import annotations

from rp_llm.orchestrator import run_pipeline


def main() -> None:
    for scenario in ["simple_nav", "simple_nav_shift"]:
        run_pipeline(scenario)


if __name__ == "__main__":
    main()
