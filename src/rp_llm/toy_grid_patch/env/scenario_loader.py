from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_scenario(path: str | Path) -> dict[str, Any]:
    scenario_path = Path(path)
    data = yaml.safe_load(scenario_path.read_text(encoding='utf-8'))
    required = {"scenario_id", "grid_size", "start", "goal", "obstacles", "keepout_zones", "battery_budget"}
    missing = required.difference(data.keys())
    if missing:
        raise ValueError(f"Scenario missing keys: {sorted(missing)}")
    return data
