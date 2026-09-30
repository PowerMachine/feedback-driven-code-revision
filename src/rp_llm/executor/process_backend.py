from __future__ import annotations

from rp_llm.envs.simple_navigation import SimpleNavigationEnv


class ProcessBackend:
    def run_episode(
        self,
        controller,
        max_steps: int = 120,
        noise: float = 0.02,
        start: tuple[float, float] = (0.0, 0.0),
        goal: tuple[float, float] = (1.0, 1.0),
        disturbance: tuple[float, float] = (0.0, 0.0),
        seed: int | None = None,
    ) -> list[dict]:
        env = SimpleNavigationEnv(
            max_steps=max_steps,
            noise=noise,
            start=start,
            goal=goal,
            disturbance=disturbance,
            seed=seed,
        )
        obs = env.reset()
        events: list[dict] = []
        for step in range(max_steps):
            action = controller.action(obs)
            action_mag = abs(action[0]) + abs(action[1])
            obs, reward, done, info = env.step(action)
            latency_ms = 10.0 + (step % 5) + (action_mag * 6.0) + (obs["distance"] * 2.0)
            events.append(
                {
                    "step": step,
                    "reward": reward,
                    "latency_ms": latency_ms,
                    "unsafe": info.get("unsafe", False),
                    "distance": obs["distance"],
                    "x": obs["x"],
                    "y": obs["y"],
                    "done": done,
                }
            )
            if done:
                break
        return events
