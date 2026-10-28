import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
S = json.loads((ROOT / "results" / "summary.json").read_text())


def test_vllm_scales_and_sglang_does_not_at_32_users():
    assert S["vllm_agg_32"] > 1.4 * S["sglang_agg_32"]
    assert S["sglang_ttft_p50_32"] > 10 * S["vllm_ttft_p50_32"]


def test_sglang_batch_is_capped_below_the_user_count():
    assert S["sglang_max_concurrent_32"] <= 20 < S["vllm_max_concurrent_32"]


def test_cache_per_token_is_smaller_for_9b():
    assert S["bf16_9b_mib_per_token"] < 0.6 * S["awq27b_mib_per_token"]


def test_only_valid_runs_are_used():
    d = pd.read_csv(ROOT / "data" / "llm_serving.csv")
    assert d.valid_run.dtype == bool and d.valid_run.sum() > 100


def test_cheapest_gpu_within_sla():
    assert S["best_usd_per_m_within_sla"] < S["worst_usd_per_m_within_sla"]
