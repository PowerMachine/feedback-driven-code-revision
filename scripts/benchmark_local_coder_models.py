from __future__ import annotations

import ast
import argparse
from dataclasses import dataclass
import gc
import json
from pathlib import Path
import time

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llama_cpp import Llama


BASE_TASK_BLOCK = """
def sum_primes_upto(n: int) -> int:
    pass

def rotate_matrix_90(mat: list[list[int]]) -> list[list[int]]:
    pass
"""

PROMPT_VARIANTS = {
    "P0": (
        "Solve the coding task below.\n"
        "You may think briefly, but final answer must include valid Python implementations.\n"
        "If you include explanation, keep it very short.\n\n"
        f"{BASE_TASK_BLOCK}\n"
        "Requirements:\n"
        "- no external dependencies\n"
        "- deterministic and correct\n"
        "- keep code reasonably concise\n"
    ),
    "P1": (
        "Solve the coding task below.\n\n"
        "Requirements:\n"
        "- deterministic and correct\n"
        "- no external dependencies\n"
        "- keep code reasonably concise\n\n"
        "Final answer must include valid Python implementations only.\n\n"
        f"{BASE_TASK_BLOCK}\n"
    ),
    "P2": (
        "You are writing utility functions for a coding interview prep notebook.\n"
        "Keep behavior deterministic and correctness-first.\n"
        "Return runnable Python implementations.\n\n"
        f"{BASE_TASK_BLOCK}\n"
        "Constraints:\n"
        "- no external dependencies\n"
        "- concise code preferred\n"
    ),
}


def reference_rotate_matrix_90(mat: list[list[int]]) -> list[list[int]]:
    n = len(mat)
    return [[mat[n - 1 - r][c] for r in range(n)] for c in range(n)]


TESTS = [
    ("sum_primes_upto(10)", lambda ns: ns["sum_primes_upto"](10) == 17),
    ("sum_primes_upto(100)", lambda ns: ns["sum_primes_upto"](100) == 1060),
    ("sum_primes_upto(1000)", lambda ns: ns["sum_primes_upto"](1000) == 76127),
    (
        "rotate_matrix_90_case_2x2",
        lambda ns: ns["rotate_matrix_90"]([[1, 2], [3, 4]]) == reference_rotate_matrix_90([[1, 2], [3, 4]]),
    ),
    (
        "rotate_matrix_90_case_3x3",
        lambda ns: ns["rotate_matrix_90"]([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        == reference_rotate_matrix_90([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
    ),
    (
        "rotate_matrix_90_case_4x4",
        lambda ns: ns["rotate_matrix_90"](
            [
                [5, 1, 9, 11],
                [2, 4, 8, 10],
                [13, 3, 6, 7],
                [15, 14, 12, 16],
            ]
        )
        == reference_rotate_matrix_90(
            [
                [5, 1, 9, 11],
                [2, 4, 8, 10],
                [13, 3, 6, 7],
                [15, 14, 12, 16],
            ]
        ),
    ),
]


@dataclass
class ModelSpec:
    label: str
    kind: str
    path: str


MODEL_SPECS = [
    ModelSpec("Qwen2.5-Coder-3B-Instruct", "transformers", "Qwen/Qwen2.5-Coder-3B-Instruct"),
    ModelSpec("Qwen2.5-Coder-7B-Instruct", "transformers", "Qwen/Qwen2.5-Coder-7B-Instruct"),
    ModelSpec("Qwen2.5-Coder-14B-Instruct", "transformers", "Qwen/Qwen2.5-Coder-14B-Instruct"),
    ModelSpec("Qwen3-Coder-30B-A3B-Instruct", "transformers", "Qwen/Qwen3-Coder-30B-A3B-Instruct"),
]


def extract_python_code(text: str) -> str:
    s = text.strip()
    if "```" in s:
        parts = s.split("```")
        for part in parts:
            p = part.strip()
            if p.startswith("python"):
                return p[len("python") :].strip()
            if p.startswith("def "):
                return p
    first_def = s.find("def ")
    if first_def >= 0:
        return s[first_def:]
    # Fallback: remove leading bullet/meta lines and keep python-like lines.
    lines = []
    for line in s.splitlines():
        stripped = line.strip()
        if stripped.startswith(("-", "*")) and "def " not in stripped:
            continue
        lines.append(line)
    s2 = "\n".join(lines).strip()
    first_def = s2.find("def ")
    if first_def >= 0:
        return s2[first_def:]
    return s


def evaluate_code(code: str) -> tuple[int, int, list[str], str | None]:
    namespace: dict = {}
    try:
        ast.parse(code)
        exec(code, namespace)
    except Exception as exc:
        return 0, len(TESTS), [], f"compile_error: {exc}"
    if "sum_primes_upto" not in namespace or "rotate_matrix_90" not in namespace:
        return 0, len(TESTS), [], "missing_required_functions"
    passed = []
    first_runtime_error: str | None = None
    for name, fn in TESTS:
        try:
            if fn(namespace):
                passed.append(name)
        except Exception as exc:
            if first_runtime_error is None:
                first_runtime_error = f"runtime_error_in_{name}: {exc}"
    return len(passed), len(TESTS), passed, first_runtime_error


def run_transformers(model_path: str, prompt: str, max_new_tokens: int = 520) -> tuple[str, int, float, float]:
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    t1 = time.perf_counter()
    device = next(model.parameters()).device
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        chat_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if not torch.is_tensor(chat_ids):
            chat_ids = chat_ids["input_ids"]
        input_ids = chat_ids.to(device)
    else:
        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
    attn = torch.ones_like(input_ids, dtype=torch.long, device=device)
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    t2 = time.perf_counter()
    generated_ids = out[0][input_ids.shape[1] :]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    token_count = int(generated_ids.shape[0])
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return text, token_count, (t1 - t0), (t2 - t1)


def run_llama_cpp(gguf_path: str, prompt: str, max_new_tokens: int = 520) -> tuple[str, int, float, float]:
    t0 = time.perf_counter()
    llm = Llama(
        model_path=gguf_path,
        n_ctx=4096,
        n_threads=8,
        n_gpu_layers=-1,
        verbose=False,
    )
    t1 = time.perf_counter()
    gemma_chat_prompt = f"<start_of_turn>user\n{prompt}\n<end_of_turn>\n<start_of_turn>model\n"
    res = llm.create_completion(
        prompt=gemma_chat_prompt,
        max_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
    )
    t2 = time.perf_counter()
    text = res["choices"][0]["text"]
    usage = res.get("usage", {})
    token_count = int(usage.get("completion_tokens", 0))
    del llm
    gc.collect()
    return text, token_count, (t1 - t0), (t2 - t1)


def build_repair_prompt(raw_text: str, eval_error: str | None = None) -> str:
    error_hint = f"\nValidation error to fix: {eval_error}\n" if eval_error else ""
    return (
        "Convert the following answer into valid Python code only.\n"
        "Output exactly two functions: sum_primes_upto and rotate_matrix_90.\n"
        "rotate_matrix_90 must correctly rotate an N x N matrix 90 degrees clockwise.\n"
        "No markdown fences, no explanations.\n\n"
        f"{error_hint}"
        f"{raw_text}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["Qwen2.5-Coder-7B-Instruct", "Qwen3-Coder-30B-A3B-Instruct"],
        help="Model labels to run.",
    )
    parser.add_argument(
        "--prompt-variants",
        nargs="+",
        default=["P0", "P1", "P2"],
        choices=sorted(PROMPT_VARIANTS.keys()),
        help="Prompt variants to benchmark.",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Repeats per (model, prompt_variant).")
    parser.add_argument("--max-new-tokens", type=int, default=520)
    parser.add_argument("--repair-max-new-tokens", type=int, default=420)
    parser.add_argument(
        "--out-subdir",
        default="prompt_robustness",
        help="Output subdirectory under artifacts/model_benchmark_coders.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    available_specs = {spec.label: spec for spec in MODEL_SPECS}
    selected_specs: list[ModelSpec] = []
    for label in args.models:
        if label not in available_specs:
            raise ValueError(f"Unknown model label: {label}")
        selected_specs.append(available_specs[label])

    selected_prompt_variants = args.prompt_variants
    out_root = Path("artifacts/model_benchmark_coders")
    run_root = out_root / args.out_subdir
    run_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for spec in selected_specs:
        for prompt_variant in selected_prompt_variants:
            prompt = PROMPT_VARIANTS[prompt_variant]
            for repeat_idx in range(args.repeats):
                row = {
                    "model": spec.label,
                    "kind": spec.kind,
                    "model_path": spec.path,
                    "prompt_variant": prompt_variant,
                    "repeat_idx": repeat_idx,
                    "status": "ok",
                    "run_ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                run_tag = f"{spec.label}__{prompt_variant}__r{repeat_idx}"
                try:
                    if spec.kind == "transformers":
                        raw_text, tokens, load_s, gen_s = run_transformers(spec.path, prompt, max_new_tokens=args.max_new_tokens)
                    else:
                        raw_text, tokens, load_s, gen_s = run_llama_cpp(spec.path, prompt, max_new_tokens=args.max_new_tokens)
                    code = extract_python_code(raw_text)
                    passed, total, passed_names, err = evaluate_code(code)
                    repair_used = False
                    # Trigger one repair pass whenever output is invalid or incomplete.
                    if err is not None or passed < total:
                        repair_prompt = build_repair_prompt(raw_text, err)
                        if spec.kind == "transformers":
                            repaired_text, repair_tokens, l2, g2 = run_transformers(
                                spec.path, repair_prompt, max_new_tokens=args.repair_max_new_tokens
                            )
                        else:
                            repaired_text, repair_tokens, l2, g2 = run_llama_cpp(
                                spec.path, repair_prompt, max_new_tokens=args.repair_max_new_tokens
                            )
                        repaired_code = extract_python_code(repaired_text)
                        p2, total2, names2, err2 = evaluate_code(repaired_code)
                        # Keep better result and include additional call time.
                        if p2 >= passed:
                            raw_text = repaired_text
                            code = repaired_code
                            passed, total, passed_names, err = p2, total2, names2, err2
                            repair_used = True
                        load_s += l2
                        gen_s += g2
                        tokens += repair_tokens
                    row.update(
                        {
                            "load_seconds": load_s,
                            "generate_seconds": gen_s,
                            "total_call_seconds": load_s + gen_s,
                            "completion_tokens": tokens,
                            "token_per_second": (tokens / gen_s) if gen_s > 0 else 0.0,
                            "accuracy_score": passed / total,
                            "passed_tests": passed,
                            "total_tests": total,
                            "eval_error": err,
                            "passed_test_names": passed_names,
                            "repair_used": repair_used,
                        }
                    )
                    (run_root / f"{run_tag}.raw.txt").write_text(raw_text, encoding="utf-8")
                    (run_root / f"{run_tag}.code.py").write_text(code, encoding="utf-8")
                except Exception as exc:  # pragma: no cover
                    row.update(
                        {
                            "status": "error",
                            "load_seconds": None,
                            "generate_seconds": None,
                            "total_call_seconds": None,
                            "completion_tokens": None,
                            "token_per_second": None,
                            "accuracy_score": None,
                            "passed_tests": 0,
                            "total_tests": len(TESTS),
                            "eval_error": str(exc),
                            "passed_test_names": [],
                            "repair_used": False,
                        }
                    )
                rows.append(row)
    payload = {
        "prompt_variants": {k: PROMPT_VARIANTS[k] for k in selected_prompt_variants},
        "test_count": len(TESTS),
        "models": [s.label for s in selected_specs],
        "repeats": args.repeats,
        "results": rows,
    }
    (run_root / "benchmark_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(run_root / "benchmark_results.csv", index=False)
    print({"saved": str(run_root / "benchmark_results.json"), "rows": len(rows)})


if __name__ == "__main__":
    main()
