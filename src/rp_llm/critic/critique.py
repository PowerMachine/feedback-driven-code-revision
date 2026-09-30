from __future__ import annotations

import statistics
from uuid import uuid4

from rp_llm.common.schemas import AxisScores, EditCritique


def _depth_from_axis(axis: AxisScores) -> str:
    if axis.safety >= 0.8:
        return "L0"
    if axis.internal >= 0.75 or axis.uncertainty >= 0.7:
        return "L1"
    if axis.environment >= 0.65 or axis.context >= 0.65:
        return "L2"
    if axis.performance >= 0.7:
        return "L4"
    return "L0"


def _operator_from_depth(depth: str) -> str:
    mapping = {
        "L0": "keep",
        "L1": "param_patch",
        "L2": "rule_patch",
        "L3": "skill_patch",
        "L4": "code_patch",
    }
    return mapping[depth]


class SixAxisCritic:
    def infer_axis(self, episode_log: list[dict]) -> AxisScores:
        if not episode_log:
            return AxisScores()
        unsafe_ratio = sum(1 for row in episode_log if row.get("unsafe", False)) / len(episode_log)
        latency = [float(row.get("latency_ms", 10.0)) for row in episode_log]
        reward = [float(row.get("reward", 0.0)) for row in episode_log]
        distance = [float(row.get("distance", 1.0)) for row in episode_log]
        x_values = [abs(float(row.get("x", 0.0))) for row in episode_log]
        y_values = [abs(float(row.get("y", 0.0))) for row in episode_log]

        safety = min(1.0, unsafe_ratio * 2.0)
        environment = min(1.0, max(max(x_values), max(y_values)) / 2.5)
        context = min(1.0, distance[-1] / max(distance[0], 1e-6))
        performance = min(1.0, statistics.fmean(latency) / 22.0)
        internal = min(1.0, abs(reward[-1] - reward[0]) / 2.0)
        uncertainty = min(1.0, statistics.pstdev(distance) / 0.25 if len(distance) > 1 else 0.0)
        return AxisScores(
            safety=safety,
            environment=environment,
            context=context,
            performance=performance,
            internal=internal,
            uncertainty=uncertainty,
        )

    def critique(self, episode_log: list[dict], target_module: str = "gold") -> EditCritique:
        axis = self.infer_axis(episode_log)
        depth = _depth_from_axis(axis)
        operator = _operator_from_depth(depth)  # type: ignore[arg-type]
        exceeded = []
        if axis.safety >= 0.8:
            exceeded.append("safety>=0.8")
        if axis.internal >= 0.75:
            exceeded.append("internal>=0.75")
        if axis.uncertainty >= 0.7:
            exceeded.append("uncertainty>=0.7")
        if axis.environment >= 0.65:
            exceeded.append("environment>=0.65")
        if axis.context >= 0.65:
            exceeded.append("context>=0.65")
        if axis.performance >= 0.7:
            exceeded.append("performance>=0.7")
        if not exceeded:
            operator = "keep"
            depth = "L0"
        failure_type = "keep" if operator == "keep" else "performance_or_stability_regression"
        recommendation = "No patch needed; keep current verified variant." if operator == "keep" else (
            f"Edit {target_module} with {operator} at {depth}; prioritize " + ", ".join(exceeded)
        )
        return EditCritique(
            axis_scores=axis,
            failure_type=failure_type,
            target_module=target_module,
            suggested_operator=operator,  # type: ignore[arg-type]
            suggested_depth=depth,  # type: ignore[arg-type]
            constraints=["verified_only_runtime", "no_fast_loop_candidate_execution"],
            evidence_summary=(
                f"critique_id={uuid4().hex[:8]} "
                f"axes={axis.model_dump()} exceeded={exceeded if exceeded else ['none']}"
            ),
            edit_recommendation=recommendation,
            confidence=0.75,
        )
