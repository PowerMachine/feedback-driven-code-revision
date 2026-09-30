from __future__ import annotations

from pathlib import Path
import os


def main() -> None:
    from huggingface_hub import snapshot_download  # type: ignore[import-untyped]

    root = Path(os.environ.get("RP_LLM_MODEL_ROOT", "models/Qwen"))
    try:
        root.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:
        print(
            {
                "error": f"permission denied for {root}",
                "hint": "Set RP_LLM_MODEL_ROOT to a writable path or request write permission.",
            }
        )
        raise SystemExit(1) from exc
    local_dir = root / "Qwen3-Coder-30B-A3B-Instruct"
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )
    print({"downloaded_to": str(local_dir)})


if __name__ == "__main__":
    main()
