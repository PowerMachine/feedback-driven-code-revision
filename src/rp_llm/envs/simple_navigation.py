from __future__ import annotations

from dataclasses import dataclass
import math
import random


@dataclass
class NavState:
    x: float
    y: float
    goal_x: float
    goal_y: float
    step: int = 0


class SimpleNavigationEnv:
    def __init__(
        self,
        max_steps: int = 120,
        noise: float = 0.0,
        start: tuple[float, float] = (0.0, 0.0),
        goal: tuple[float, float] = (1.0, 1.0),
        disturbance: tuple[float, float] = (0.0, 0.0),
        seed: int | None = None,
    ) -> None:
        self.max_steps = max_steps
        self.noise = noise
        self.start = start
        self.goal = goal
        self.disturbance = disturbance
        self.rng = random.Random(seed)
        self.state = NavState(start[0], start[1], goal[0], goal[1], 0)

    def reset(self) -> dict:
        self.state = NavState(self.start[0], self.start[1], self.goal[0], self.goal[1], 0)
        return self.observe()

    def observe(self) -> dict:
        dx = self.state.goal_x - self.state.x
        dy = self.state.goal_y - self.state.y
        dist = math.sqrt(dx * dx + dy * dy)
        return {"x": self.state.x, "y": self.state.y, "dx": dx, "dy": dy, "distance": dist}

    def step(self, action: tuple[float, float]) -> tuple[dict, float, bool, dict]:
        ax, ay = action
        self.state.x += ax + self.disturbance[0] + self.rng.uniform(-self.noise, self.noise)
        self.state.y += ay + self.disturbance[1] + self.rng.uniform(-self.noise, self.noise)
        self.state.step += 1
        obs = self.observe()
        reward = -obs["distance"]
        done = obs["distance"] < 0.05 or self.state.step >= self.max_steps
        info = {"unsafe": abs(self.state.x) > 2.5 or abs(self.state.y) > 2.5}
        return obs, reward, done, info
