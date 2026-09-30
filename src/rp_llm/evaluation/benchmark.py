from __future__ import annotations

from dataclasses import dataclass
import statistics


@dataclass
class BenchmarkResult:
    name: str
    success_rate: float
    success_std: float
    unsafe_rate: float
    recovery_time: float
    p95_cycle_time_ms: float
    rollback_rate: float
    mean_final_distance: float
    reward_mean: float
    reward_std: float
    episode_count: int
    accepted_patch_count: int


def _normalize_episodes(events_or_episodes: list[dict] | list[list[dict]]) -> list[list[dict]]:
    if not events_or_episodes:
        return []
    first = events_or_episodes[0]
    if isinstance(first, dict):
        return [events_or_episodes]  # type: ignore[list-item]
    return events_or_episodes  # type: ignore[return-value]


def summarize(name: str, events_or_episodes: list[dict] | list[list[dict]], accepted_patch_count: int = 0) -> BenchmarkResult:
    episodes = _normalize_episodes(events_or_episodes)
    if not episodes:
        return BenchmarkResult(name, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 999.0, 0.0, 0.0, 0, accepted_patch_count)

    success_rows: list[float] = []
    unsafe_rows: list[float] = []
    recovery_rows: list[float] = []
    p95_rows: list[float] = []
    rollback_rows: list[float] = []
    final_distance_rows: list[float] = []
    reward_rows: list[float] = []

    for events in episodes:
        if not events:
            continue
        unsafe = sum(1 for e in events if e.get("unsafe", False))
        success_rows.append(1.0 if events[-1].get("distance", 999.0) < 0.2 else 0.0)
        unsafe_rows.append(unsafe / len(events))
        recovery_rows.append(float(len(events)))
        latency = [e.get("latency_ms", 0.0) for e in events]
        p95 = statistics.quantiles(latency, n=20)[-1] if len(latency) > 1 else latency[0]
        p95_rows.append(float(p95))
        rollback_rows.append(1.0 if unsafe > 0 else 0.0)
        final_distance_rows.append(float(events[-1].get("distance", 999.0)))
        reward_rows.append(float(sum(e.get("reward", 0.0) for e in events)))

    if not success_rows:
        return BenchmarkResult(name, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 999.0, 0.0, 0.0, 0, accepted_patch_count)

    return BenchmarkResult(
        name=name,
        success_rate=statistics.fmean(success_rows),
        success_std=statistics.pstdev(success_rows) if len(success_rows) > 1 else 0.0,
        unsafe_rate=statistics.fmean(unsafe_rows),
        recovery_time=statistics.fmean(recovery_rows),
        p95_cycle_time_ms=statistics.fmean(p95_rows),
        rollback_rate=statistics.fmean(rollback_rows),
        mean_final_distance=statistics.fmean(final_distance_rows),
        reward_mean=statistics.fmean(reward_rows),
        reward_std=statistics.pstdev(reward_rows) if len(reward_rows) > 1 else 0.0,
        episode_count=len(success_rows),
        accepted_patch_count=accepted_patch_count,
    )
