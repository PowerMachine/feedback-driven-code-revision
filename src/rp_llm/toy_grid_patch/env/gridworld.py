from __future__ import annotations

from dataclasses import dataclass


Coord = tuple[int, int]


@dataclass
class GridWorld:
    grid_size: tuple[int, int]
    start: Coord
    goal: Coord
    obstacles: set[Coord]
    keepout_zones: set[Coord]
    battery_budget: int

    @classmethod
    def from_scenario(cls, scenario: dict) -> "GridWorld":
        rows, cols = scenario["grid_size"]
        return cls(
            grid_size=(rows, cols),
            start=tuple(scenario["start"]),
            goal=tuple(scenario["goal"]),
            obstacles={tuple(v) for v in scenario["obstacles"]},
            keepout_zones={tuple(v) for v in scenario["keepout_zones"]},
            battery_budget=int(scenario["battery_budget"]),
        )

    def in_bounds(self, pos: Coord) -> bool:
        r, c = pos
        rows, cols = self.grid_size
        return 0 <= r < rows and 0 <= c < cols

    def is_blocked(self, pos: Coord) -> bool:
        return pos in self.obstacles

    def neighbors4(self, pos: Coord) -> list[Coord]:
        r, c = pos
        out: list[Coord] = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (r + dr, c + dc)
            if self.in_bounds(nxt) and not self.is_blocked(nxt):
                out.append(nxt)
        return out

    def near_obstacle(self, pos: Coord) -> bool:
        r, c = pos
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            cand = (r + dr, c + dc)
            if cand in self.obstacles:
                return True
        return False
