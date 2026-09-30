from __future__ import annotations

from pathlib import Path
import json
import os


class QwenLocalClient:
    def __init__(
        self,
        model_dir: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        model_name: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    ) -> None:
        self.model_dir = Path(os.environ.get("RP_LLM_MODEL_DIR", model_dir))
        self.model_name = model_name

    def _mock_response(self) -> dict:
        # Test fallback only. Real runs should keep RP_LLM_ALLOW_MOCK_LLM unset.
        return {
            "target_module": "v_conservative",
            "operator": "param_patch",
            "edit_depth": "L1",
            "objective": "Improve tracking speed without violating safety thresholds.",
            "patch_spec": {"controller_gain_x": 0.16, "controller_gain_y": 0.16},
            "verifier_constraints": ["verified_only_runtime", "no_fast_loop_candidate_execution"],
            "rationale": "Conservative controller is too slow under disturbance; modest gain increase should improve convergence.",
        }

    def _load_model_and_tokenizer(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-untyped]
        import torch

        model_path = str(self.model_dir if self.model_dir.exists() else self.model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        return model, tokenizer

    def generate_text(self, prompt: str, max_new_tokens: int = 768) -> str:
        model, tokenizer = self._load_model_and_tokenizer()
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id,
        )
        return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()

    def _generate_with_transformers(self, prompt: str) -> dict:
        model, tokenizer = self._load_model_and_tokenizer()
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        return self._parse_json_object(text)

    def _parse_json_object(self, text: str) -> dict:
        if text.startswith("{") and text.endswith("}"):
            return json.loads(text)
        start = text.find("{")
        if start == -1:
            raise ValueError(f"No JSON object found in model output: {text[:200]}")
        decoder = json.JSONDecoder()
        parsed, _ = decoder.raw_decode(text[start:])
        return parsed

    def generate_json(self, prompt: str) -> dict:
        if os.environ.get("RP_LLM_ALLOW_MOCK_LLM", "0") == "1":
            return self._mock_response()
        try:
            return self._generate_with_transformers(prompt)
        except Exception as exc:  # pragma: no cover - runtime environment dependent
            raise RuntimeError(
                "Qwen local generation failed. Download model to "
                f"{self.model_dir} and install transformers/torch. Original error: {exc}"
            ) from exc
