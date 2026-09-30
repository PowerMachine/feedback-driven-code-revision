from __future__ import annotations

import random
from copy import deepcopy


_SHIFTS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1), (0, 0)]


def _in_bounds(pos: tuple[int, int], rows: int, cols: int) -> bool:
    return 0 <= pos[0] < rows and 0 <= pos[1] < cols


def perturb_scenario(scenario: dict, seed: int, *, shift_prob: float = 0.5) -> dict:
    """Deterministic per-seed perturbation of obstacle/keepout positions.

    Rules:
      - start, goal, grid_size, battery_budget are immutable.
      - Each obstacle/keepout cell is shifted by one of _SHIFTS with prob=shift_prob;
        otherwise kept in place. Shifts that collide with start/goal, escape the grid,
        or produce duplicates with another occupied cell are dropped (keep original).
      - Obstacles and keepouts never overlap each other after perturbation.
    """
    rng = random.Random(seed)
    out = deepcopy(scenario)
    rows, cols = out["grid_size"]
    start = tuple(out["start"])
    goal = tuple(out["goal"])

    occupied: set[tuple[int, int]] = {start, goal}
    result_obs: list[list[int]] = []
    for cell in out.get("obstacles", []):
        orig = (int(cell[0]), int(cell[1]))
        shift = rng.choices(_SHIFTS, weights=[1] * (len(_SHIFTS) - 1) + [int(round((1 - shift_prob) * 8))])[0]
        cand = (orig[0] + shift[0], orig[1] + shift[1])
        if not _in_bounds(cand, rows, cols) or cand in occupied:
            cand = orig if orig not in occupied else None  # type: ignore[assignment]
        if cand is None:
            continue
        occupied.add(cand)
        result_obs.append(list(cand))
    out["obstacles"] = result_obs

    result_keep: list[list[int]] = []
    for cell in out.get("keepout_zones", []):
        orig = (int(cell[0]), int(cell[1]))
        shift = rng.choices(_SHIFTS, weights=[1] * (len(_SHIFTS) - 1) + [int(round((1 - shift_prob) * 8))])[0]
        cand = (orig[0] + shift[0], orig[1] + shift[1])
        if not _in_bounds(cand, rows, cols) or cand in occupied:
            cand = orig if orig not in occupied else None  # type: ignore[assignment]
        if cand is None:
            continue
        occupied.add(cand)
        result_keep.append(list(cand))
    out["keepout_zones"] = result_keep

    out["scenario_id"] = f"{scenario['scenario_id']}__p{seed}"
    return out
