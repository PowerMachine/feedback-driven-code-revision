from __future__ import annotations

import json
import os
import random
import re
from copy import deepcopy
from typing import Any


# ---------------------------------------------------------------------------
# Static stub (kept for regression / fully offline runs).
# ---------------------------------------------------------------------------


def _extract_json_obj(text: str) -> dict[str, Any]:
    s = text.strip()
    if s.startswith("{") and s.endswith("}"):
        return json.loads(s)
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("LLM output is not valid JSON object")
    candidate = s[start : end + 1]
    candidate = re.sub(r",\s*}", "}", candidate)
    candidate = re.sub(r",\s*]", "]", candidate)
    return json.loads(candidate)


def _jitter_value(name: str, value: Any, rng: random.Random) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        delta = max(1, int(abs(value) * 0.30))
        pct = rng.uniform(-0.30, 0.30)
        delta = int(round(value * pct)) if abs(value) >= 4 else rng.choice([-delta, 0, delta])
        return max(0, value + delta)
    if isinstance(value, float):
        pct = rng.uniform(-0.20, 0.20)
        return round(max(0.0, value * (1.0 + pct)), 2)
    return value


def apply_jitter(patch_payload: dict, seed: int) -> dict:
    rng = random.Random(seed)
    out = deepcopy(patch_payload)
    for change in out.get("changes", []):
        change["new_value"] = _jitter_value(change["name"], change["new_value"], rng)
        change["reason"] = f"{change.get('reason', '')} [jitter seed={seed}]"
    out["expected_effect"] = f"{out.get('expected_effect', '')} (seed={seed})"
    return out


# ---------------------------------------------------------------------------
# Real Qwen call
# ---------------------------------------------------------------------------


ALLOWED_CONSTANTS = [
    "MOVE_COST",
    "KEEP_OUT_PENALTY",
    "OBSTACLE_NEAR_PENALTY",
    "LOW_BATTERY_PENALTY",
    "RISK_WEIGHT",
    "REPLAN_THRESHOLD",
]

ALLOWED_RANGES = {
    "MOVE_COST": (1, 5),
    "KEEP_OUT_PENALTY": (0, 5000),
    "OBSTACLE_NEAR_PENALTY": (0, 500),
    "LOW_BATTERY_PENALTY": (0, 1000),
    "RISK_WEIGHT": (0.5, 10.0),
    "REPLAN_THRESHOLD": (1, 10),
}


SYSTEM_PROMPT = """You are a strict planning-rule editor for a GridWorld A* planner.
You DO NOT write code. You only propose changes to a whitelisted set of numeric constants that the A* cost function reads.

Rules you must follow:
1. Output MUST be a single JSON object with this exact shape and NOTHING else:
   {
     "changes": [ {"name": "<CONSTANT_NAME>", "new_value": <number>, "reason": "<one short sentence>"} ],
     "expected_effect": "<one or two sentence description of what the patch should achieve>"
   }
2. "changes" length MUST be between 0 and 3.
3. Only these constants are allowed to change: MOVE_COST, KEEP_OUT_PENALTY, OBSTACLE_NEAR_PENALTY, LOW_BATTERY_PENALTY, RISK_WEIGHT, REPLAN_THRESHOLD.
4. Stay within allowed ranges.
5. Do not add any prose, markdown fences, or keys outside the schema.
6. You may choose to return zero changes if the baseline is already good."""


def _build_user_prompt(current_rules: dict, scenario: dict, baseline_metrics: dict, failure_report: dict) -> str:
    rules_lines = "\n".join(f"  {k} = {current_rules.get(k)}" for k in ALLOWED_CONSTANTS)
    range_lines = "\n".join(f"  {k}: {lo}..{hi}" for k, (lo, hi) in ALLOWED_RANGES.items())
    scenario_summary = {
        "scenario_id": scenario.get("scenario_id"),
        "grid_size": scenario.get("grid_size"),
        "start": scenario.get("start"),
        "goal": scenario.get("goal"),
        "num_obstacles": len(scenario.get("obstacles", [])),
        "num_keepouts": len(scenario.get("keepout_zones", [])),
        "battery_budget": scenario.get("battery_budget"),
    }
    return (
        f"Current rule values:\n{rules_lines}\n\n"
        f"Allowed ranges:\n{range_lines}\n\n"
        f"Scenario summary:\n{json.dumps(scenario_summary, ensure_ascii=False)}\n\n"
        f"Baseline metrics:\n{json.dumps(baseline_metrics, ensure_ascii=False)}\n\n"
        f"Failure report:\n{json.dumps(failure_report, ensure_ascii=False)}\n\n"
        "Given this situation, propose the minimal patch (0-3 changes) that would most likely reduce keepout violations, avoid battery exhaustion, and keep the plan safe. Return ONLY the JSON object."
    )


_MODEL_CACHE: dict = {}


def _load_qwen(model_path: str):
    """Lazy-load and cache a Qwen Instruct model across runs."""
    import torch  # Imported lazily so the stub path has no torch dependency.
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if model_path in _MODEL_CACHE:
        return _MODEL_CACHE[model_path]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    _MODEL_CACHE[model_path] = (tokenizer, model)
    return tokenizer, model


def _call_qwen(
    model_path: str,
    system_prompt: str,
    user_prompt: str,
    seed: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> str:
    import torch

    tokenizer, model = _load_qwen(model_path)
    device = next(model.parameters()).device
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    chat_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if not torch.is_tensor(chat_ids):
        chat_ids = chat_ids["input_ids"]
    input_ids = chat_ids.to(device)
    attn = torch.ones_like(input_ids, dtype=torch.long, device=device)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    out = model.generate(
        input_ids=input_ids,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    generated_ids = out[0][input_ids.shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def propose_patch_with_qwen(
    current_rules: dict,
    scenario: dict,
    baseline_metrics: dict,
    failure_report: dict,
    *,
    model_path: str,
    run_idx: int,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_new_tokens: int = 350,
    seed_base: int = 1000,
) -> tuple[dict | None, str]:
    """Query an Instruct LLM for a patch JSON. Returns (parsed_payload_or_None, raw_text)."""
    seed = seed_base + int(run_idx)
    user_prompt = _build_user_prompt(current_rules, scenario, baseline_metrics, failure_report)
    raw = _call_qwen(
        model_path=model_path,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        seed=seed,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )
    try:
        parsed = _extract_json_obj(raw)
    except Exception:
        return None, raw
    if "changes" not in parsed or "expected_effect" not in parsed:
        return None, raw
    return parsed, raw


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def propose_patch_with_llm(
    current_rules: dict,
    scenario: dict,
    baseline_metrics: dict,
    failure_report: dict,
    *,
    run_idx: int | None = None,
    jitter: bool = False,
    llm_backend: str = "stub",
    llm_model_path: str | None = None,
    llm_temperature: float = 0.7,
    llm_top_p: float = 0.95,
    llm_max_new_tokens: int = 350,
    llm_seed_base: int = 1000,
) -> tuple[dict | None, dict]:
    """Return (payload, trace).

    backend = "stub": read RP_LLM_TOY_PATCH_JSON env var (offline, scripted).
    backend = "qwen": actually call a Qwen Instruct model with temperature>0.
    """
    trace: dict = {"backend": llm_backend}

    if llm_backend == "qwen":
        if not llm_model_path:
            raise ValueError("llm_model_path is required when llm_backend='qwen'")
        payload, raw = propose_patch_with_qwen(
            current_rules,
            scenario,
            baseline_metrics,
            failure_report,
            model_path=llm_model_path,
            run_idx=int(run_idx or 0),
            temperature=llm_temperature,
            top_p=llm_top_p,
            max_new_tokens=llm_max_new_tokens,
            seed_base=llm_seed_base,
        )
        trace.update(
            {
                "model_path": llm_model_path,
                "temperature": llm_temperature,
                "top_p": llm_top_p,
                "seed": llm_seed_base + int(run_idx or 0),
                "raw_output": raw,
                "parsed_ok": payload is not None,
            }
        )
        if payload is None:
            return None, trace
        if jitter:
            payload = apply_jitter(payload, seed=trace["seed"])
        return payload, trace

    raw = os.environ.get("RP_LLM_TOY_PATCH_JSON", "").strip()
    trace["raw_output"] = raw
    if not raw:
        trace["parsed_ok"] = False
        return None, trace
    parsed = _extract_json_obj(raw)
    if "changes" not in parsed or "expected_effect" not in parsed:
        trace["parsed_ok"] = False
        raise ValueError("LLM patch JSON must contain changes and expected_effect")
    trace["parsed_ok"] = True
    if jitter:
        try:
            base_seed = int(os.environ.get("RP_LLM_TOY_PATCH_SEED", "0"))
        except ValueError:
            base_seed = 0
        parsed = apply_jitter(parsed, base_seed + int(run_idx or 0))
    return parsed, trace
