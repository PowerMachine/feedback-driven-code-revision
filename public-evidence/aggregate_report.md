# GridWorld Rule Patch Aggregate Report

| scenario | patcher | runs | improved_rate | keepout_delta (mean±std) | path_delta (mean±std) | unique_patches | fallback |
|---|---|---|---|---|---|---|---|
| scenario_battery_tight | deterministic | 10 | 0.50 | +0.80±0.87 | +0.00±0.00 | 2/10 | 0 |
| scenario_battery_tight | llm_qwen | 10 | 0.50 | +0.60±1.20 | +0.00±0.00 | 9/10 | 1 |
| scenario_keepout_01 | deterministic | 10 | 0.10 | +0.10±0.30 | +0.00±0.00 | 2/10 | 0 |
| scenario_keepout_01 | llm_qwen | 10 | 0.10 | +0.10±0.30 | +0.00±0.00 | 3/10 | 2 |
| scenario_obstacle_dense | deterministic | 10 | 0.50 | +1.40±1.74 | +0.20±0.60 | 3/10 | 0 |
| scenario_obstacle_dense | llm_qwen | 10 | 0.30 | +1.10±1.81 | +0.00±0.00 | 6/10 | 1 |

unique_patches = distinct (name,new_value) signatures across runs; higher = LLM varied its choices more.

## LLM choice consistency per constant
### scenario_battery_tight / deterministic
| constant | n | mean | stdev | min | max |
|---|---|---|---|---|---|
| LOW_BATTERY_PENALTY | 10 | 80.00 | 0.00 | 80.00 | 80.00 |
| KEEP_OUT_PENALTY | 5 | 300.00 | 0.00 | 300.00 | 300.00 |

### scenario_battery_tight / llm_qwen
| constant | n | mean | stdev | min | max |
|---|---|---|---|---|---|
| LOW_BATTERY_PENALTY | 6 | 446.67 | 219.60 | 80.00 | 800.00 |
| RISK_WEIGHT | 6 | 2.17 | 0.37 | 2.00 | 3.00 |
| KEEP_OUT_PENALTY | 5 | 101.00 | 103.94 | 5.00 | 300.00 |
| REPLAN_THRESHOLD | 3 | 2.67 | 0.94 | 2.00 | 4.00 |

### scenario_keepout_01 / deterministic
| constant | n | mean | stdev | min | max |
|---|---|---|---|---|---|
| KEEP_OUT_PENALTY | 1 | 300.00 | 0.00 | 300.00 | 300.00 |

### scenario_keepout_01 / llm_qwen
| constant | n | mean | stdev | min | max |
|---|---|---|---|---|---|
| KEEP_OUT_PENALTY | 2 | 5.00 | 0.00 | 5.00 | 5.00 |
| LOW_BATTERY_PENALTY | 1 | 50.00 | 0.00 | 50.00 | 50.00 |
| RISK_WEIGHT | 1 | 2.00 | 0.00 | 2.00 | 2.00 |

### scenario_obstacle_dense / deterministic
| constant | n | mean | stdev | min | max |
|---|---|---|---|---|---|
| KEEP_OUT_PENALTY | 5 | 300.00 | 0.00 | 300.00 | 300.00 |
| RISK_WEIGHT | 2 | 1.80 | 0.00 | 1.80 | 1.80 |

### scenario_obstacle_dense / llm_qwen
| constant | n | mean | stdev | min | max |
|---|---|---|---|---|---|
| KEEP_OUT_PENALTY | 6 | 44.17 | 42.57 | 5.00 | 100.00 |
| OBSTACLE_NEAR_PENALTY | 2 | 10.00 | 5.00 | 5.00 | 15.00 |
| LOW_BATTERY_PENALTY | 2 | 80.00 | 70.00 | 10.00 | 150.00 |
| RISK_WEIGHT | 4 | 2.00 | 0.00 | 2.00 | 2.00 |

