from __future__ import annotations

from pathlib import Path
import json
import textwrap

from rp_llm.llm.client import QwenLocalClient


BASELINE_CODE = textwrap.dedent(
    """
    ACTIONS = {
        "U": (-1, 0),
        "D": (1, 0),
        "L": (0, -1),
        "R": (0, 1),
    }


    class GridWorld:
        def __init__(self):
            self.rows = 3
            self.cols = 4
            self.start = (2, 0)
            self.goal = (0, 3)
            self.bomb = (1, 3)
            self.blocked = {(1, 1)}
            self.gamma = 0.95
            self.step_cost = -0.04

        def in_bounds(self, state):
            r, c = state
            return 0 <= r < self.rows and 0 <= c < self.cols

        def terminal_reward(self, state):
            if state == self.goal:
                return 1.0
            if state == self.bomb:
                return -1.0
            return None

        def next_state(self, state, action):
            if self.terminal_reward(state) is not None:
                return state
            dr, dc = ACTIONS[action]
            candidate = (state[0] + dr, state[1] + dc)
            if not self.in_bounds(candidate) or candidate in self.blocked:
                return state
            return candidate

        def value_iteration(self, iterations=200):
            states = [
                (r, c)
                for r in range(self.rows)
                for c in range(self.cols)
                if (r, c) not in self.blocked
            ]
            values = {s: 0.0 for s in states}
            for _ in range(iterations):
                new_values = values.copy()
                for state in states:
                    reward = self.terminal_reward(state)
                    if reward is not None:
                        new_values[state] = reward
                        continue
                    q_values = []
                    for action in ACTIONS:
                        nxt = self.next_state(state, action)
                        terminal = self.terminal_reward(nxt)
                        if terminal is not None:
                            q_values.append(terminal)
                        else:
                            q_values.append(self.step_cost + self.gamma * values[nxt])
                    new_values[state] = max(q_values)
                values = new_values
            return values

        def greedy_policy(self):
            values = self.value_iteration()
            policy = {}
            for r in range(self.rows):
                for c in range(self.cols):
                    state = (r, c)
                    reward = self.terminal_reward(state)
                    if state in self.blocked:
                        policy[state] = "WALL"
                        continue
                    if reward is not None:
                        policy[state] = "GOAL" if reward > 0 else "BOMB"
                        continue
                    best_action = None
                    best_value = -10**9
                    for action in ACTIONS:
                        nxt = self.next_state(state, action)
                        terminal = self.terminal_reward(nxt)
                        score = terminal if terminal is not None else self.step_cost + self.gamma * values[nxt]
                        if score > best_value:
                            best_value = score
                            best_action = action
                    policy[state] = best_action
            return policy

        def route_from_start(self, max_steps=20):
            policy = self.greedy_policy()
            state = self.start
            route = [state]
            for _ in range(max_steps):
                action = policy[state]
                if action in {"GOAL", "BOMB", "WALL"}:
                    break
                state = self.next_state(state, action)
                route.append(state)
                if state == self.goal or state == self.bomb:
                    break
            return route

        def render_policy_table(self):
            policy = self.greedy_policy()
            rows = []
            for r in range(self.rows):
                row = []
                for c in range(self.cols):
                    row.append(policy[(r, c)])
                rows.append(row)
            return rows


    def solve():
        env = GridWorld()
        return {
            "blocked": sorted(list(env.blocked)),
            "route": env.route_from_start(),
            "policy_table": env.render_policy_table(),
        }
    """
)


def _extract_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Could not find JSON object in model output: {text[:300]}")
    return json.loads(text[start : end + 1])


def _execute_solution(code: str) -> dict:
    namespace: dict = {}
    exec(code, namespace)
    return namespace["solve"]()


def build_prompt() -> str:
    return "\n".join(
        [
            "You are modifying a Python gridworld baseline for a reinforcement-learning homework style task.",
            "Return STRICT JSON only. Do not wrap with markdown fences.",
            "Goal: keep the same 3x4 gridworld structure from the baseline, but add one new obstacle on the shortest baseline route.",
            "Use new obstacle at cell (2, 1).",
            "Keep the existing wall at (1, 1), goal at (0, 3), bomb at (1, 3), start at (2, 0).",
            "Return JSON with keys: change_summary, updated_code, expected_effect.",
            "The updated_code must be complete runnable Python and must preserve solve().",
            "Baseline code:",
            BASELINE_CODE,
        ]
    )


def main() -> None:
    out_dir = Path("artifacts/qwen_gridworld_demo")
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_result = _execute_solution(BASELINE_CODE)
    (out_dir / "baseline_gridworld.py").write_text(BASELINE_CODE, encoding="utf-8")

    client = QwenLocalClient()
    prompt = build_prompt()
    raw_text = client.generate_text(prompt, max_new_tokens=1400)
    parsed = _extract_json_object(raw_text)
    updated_code = parsed["updated_code"]
    modified_result = _execute_solution(updated_code)

    (out_dir / "qwen_raw_response.txt").write_text(raw_text, encoding="utf-8")
    (out_dir / "modified_gridworld.py").write_text(updated_code, encoding="utf-8")

    summary = {
        "model": "Qwen3-Coder-30B-A3B-Instruct",
        "baseline": baseline_result,
        "modified": modified_result,
        "change_summary": parsed.get("change_summary", ""),
        "expected_effect": parsed.get("expected_effect", ""),
        "baseline_route_length": len(baseline_result["route"]) - 1,
        "modified_route_length": len(modified_result["route"]) - 1,
    }
    (out_dir / "result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
