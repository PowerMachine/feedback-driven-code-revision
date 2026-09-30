# Feedback-Driven Code Revision

[![CI](https://github.com/PowerMachine/feedback-driven-code-revision/actions/workflows/ci.yml/badge.svg)](https://github.com/PowerMachine/feedback-driven-code-revision/actions/workflows/ci.yml)

A simulation-only research prototype for testing whether an LLM can propose bounded code or
configuration revisions, pass them through deterministic verification, and promote only changes
that improve a closed-world control task.

## Why this repository exists

The system separates an adaptive proposal path from a verified runtime path:

```text
failure evidence -> critique -> candidate patch -> static/runtime checks
                 -> before/after evaluation -> promote or roll back
```

It is not a self-modifying production controller. Generated changes are restricted to a small
GridWorld testbed, and the runtime keeps a deterministic fallback.

## Recorded study

The checked-in aggregate covers 60 runs: three scenarios, two patchers, and ten seeds. In this
testbed, the Qwen patcher produced more varied patch signatures than the deterministic baseline,
but did not consistently improve more often. Four of 30 LLM responses required fallback, and one
LLM intervention regressed the safety metric. Those negative results motivate the verifier and
rollback path rather than a model-quality claim.

- [Aggregate report](public-evidence/aggregate_report.md)
- [Machine-readable aggregate](public-evidence/aggregate_summary.json)
- [Behavioral study notes](public-evidence/llm_behavioral_study_v2.md)

## Run the safe smoke path

```bash
python -m pip install -e ".[dev]"
RP_LLM_ALLOW_MOCK_LLM=1 pytest -q
RP_LLM_ALLOW_MOCK_LLM=1 python scripts/run_demo_before_after.py --scenario test_smoke
```

For a real local model, install `.[llm]` and pass a model path explicitly to the sweep or patch
runner. Model weights are not included.

## Public-release boundary

This repository contains authored source, tests, configuration, and aggregate measurements. It
excludes model weights, raw model generations, internal reports, environment-specific paths,
credentials, and third-party datasets.

MIT licensed. See [LICENSE](LICENSE).
