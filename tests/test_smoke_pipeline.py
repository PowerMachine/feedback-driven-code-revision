import os

from rp_llm.orchestrator import run_pipeline


def test_smoke_pipeline():
    os.environ["RP_LLM_ALLOW_MOCK_LLM"] = "1"
    result = run_pipeline("test_smoke")
    assert result["accepted_patch_count"] >= 0
    assert any(row["name"] == "heuristic_only" for row in result["results"])
