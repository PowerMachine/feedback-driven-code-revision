from __future__ import annotations

from rp_llm.evaluation.benchmark import summarize


def static_gold(events: list[dict]):
    return summarize("static_gold", events, accepted_patch_count=0)


def switch_only(events: list[dict]):
    return summarize("switch_only", events, accepted_patch_count=0)


def heuristic_only(events: list[dict]):
    return summarize("heuristic_only", events, accepted_patch_count=0)


def rp_llm_full(events: list[dict], accepted_patch_count: int):
    return summarize("rp_llm_full", events, accepted_patch_count=accepted_patch_count)
