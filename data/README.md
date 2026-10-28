# Data

Curated CSV files from the campaign (public Kaggle dataset `lucasrangelss/gpu-ai-benchmark-campaign-2026`). One row per measured run or attempt; the columns are described in the dataset page. Host locations and instance identifiers were removed.

- `llm_serving.csv`: load tests of vLLM servers: one row per model build, GPU and concurrency level (`valid_run` marks the usable rows)
- `engine_comparison.csv`: vLLM against SGLang on one RTX 5090 (and the first RTX 4090 attempt)
- `engine_requests.csv`: every request of the engine comparison with start, first-token and end times
- `longctx_runs.csv`: long-context and prefix-cache runs (cold prefill, cached prefill, multi-turn agent sessions)
- `vllm_starts_all.csv`: key figures parsed from every vLLM server start (weights, KV cache, concurrency, failures)
- `soak_requests_5090_awq.csv`: 4,051 requests of the 35-minute stability test
- `soak_gpu_5090.csv`: GPU sampler of the stability test (memory, power, clocks)
