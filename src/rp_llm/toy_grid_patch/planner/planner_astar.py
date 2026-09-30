from __future__ import annotations

import heapq
import random
from dataclasses import dataclass

from rp_llm.toy_grid_patch.env.gridworld import GridWorld
from rp_llm.toy_grid_patch.planner import planner_rules as rules
from rp_llm.toy_grid_patch.planner.planner_utils import manhattan


@dataclass
class PlanResult:
    path: list[tuple[int, int]]
    replan_count: int
    total_cost: float


def _step_cost(env: GridWorld, cell: tuple[int, int], step_idx: int) -> float:
    cost = float(rules.MOVE_COST)
    if cell in env.keepout_zones:
        cost += float(rules.KEEP_OUT_PENALTY)
    if env.near_obstacle(cell):
        cost += float(rules.OBSTACLE_NEAR_PENALTY)
    battery_remaining = env.battery_budget - step_idx
    if battery_remaining <= max(2, int(rules.REPLAN_THRESHOLD)):
        cost += float(rules.LOW_BATTERY_PENALTY)
    return cost * float(rules.RISK_WEIGHT)


def plan_path(env: GridWorld, *, tiebreak_seed: int | None = None) -> PlanResult:
    """A* with optional per-call tiebreak randomization.

    When `tiebreak_seed` is provided, equal f-scores are broken by a small
    deterministic-per-seed random tiebreak key. This preserves optimality
    (the tiebreak is strictly less than MOVE_COST) while exposing path
    diversity across seeds.
    """
    start, goal = env.start, env.goal
    rng = random.Random(tiebreak_seed) if tiebreak_seed is not None else None
    tiebreak_scale = max(1e-6, 0.01 * float(rules.MOVE_COST))

    open_heap: list[tuple[float, float, tuple[int, int]]] = []
    heapq.heappush(open_heap, (0.0, 0.0, start))
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    g_score: dict[tuple[int, int], float] = {start: 0.0}

    while open_heap:
        _, _, cur = heapq.heappop(open_heap)
        if cur == goal:
            break
        step_idx = int(g_score[cur] // max(1.0, float(rules.MOVE_COST)))
        for nxt in env.neighbors4(cur):
            tentative = g_score[cur] + _step_cost(env, nxt, step_idx + 1)
            if tentative < g_score.get(nxt, float("inf")):
                came_from[nxt] = cur
                g_score[nxt] = tentative
                f_score = tentative + manhattan(nxt, goal)
                tiebreak = rng.uniform(0.0, tiebreak_scale) if rng is not None else 0.0
                heapq.heappush(open_heap, (f_score, tiebreak, nxt))

    if goal not in came_from:
        return PlanResult(path=[start], replan_count=0, total_cost=float("inf"))

    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = came_from[cur]
    path.reverse()

    keepout_hits = sum(1 for p in path if p in env.keepout_zones)
    replan_count = 1 if keepout_hits >= int(rules.REPLAN_THRESHOLD) else 0
    return PlanResult(path=path, replan_count=replan_count, total_cost=g_score.get(goal, float("inf")))
