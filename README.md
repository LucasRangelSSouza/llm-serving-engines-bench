# LLM serving engines under load

vLLM and SGLang serving benchmarks (time to first token, per-user decode speed, aggregate throughput) across concurrency levels and GPUs. It measures serving behaviour, not model quality.

Author: Lucas Rangel. All numbers below come from `evidence/results.json` (measured runs on rented Vast.ai GPUs, Oct 2026). Rows with failed or install-only status are not included. Every row is a real run on a shared marketplace host: values are single-session medians, outliers are kept (not filtered), and a different host of the same GPU model can differ noticeably.

## Conclusion

Valid runs: 149. A run is valid when it had no connection errors and at least one successful request; requests the server rejected on purpose (a prompt longer than the context limit it was started with) are not counted against it. Highest aggregate output speed: 1802.3 tok/s on RTX 5090 (b9speed, 32 users). Per-user decode speed falls as concurrency rises; see the tables for the knee on each card.

## Serving engine runs (valid runs: no connection errors and at least one successful request, thinking mode excluded)

| gpu | variant | users | requests | ttft_p50_s | ttft_p95_s | decode_tps_per_user_p50 | agg_out_tps | wall_s |
|---|---|---|---|---|---|---|---|---|
| L40S | fp8_plain | 1 | 5 | 0.142 | 2.33 | 21.7 | 20.3 | 43.9 |
| L40S | fp8_plain | 2 | 8 | 0.185 | 0.241 | 21.3 | 41.4 | 43.3 |
| L40S | fp8_plain | 4 | 17 | 0.189 | 0.444 | 20.6 | 72.9 | 54.8 |
| L40S | fp8_plain | 8 | 34 | 0.197 | 0.643 | 19.5 | 136 | 53.3 |
| L40S | fp8_plain | 16 | 58 | 0.243 | 1.35 | 17.7 | 220 | 59.4 |
| L40S | fp8_plain | 32 | 90 | 0.299 | 2.74 | 14.4 | 338 | 65.5 |
| RTX PRO 6000 S | bf16_dflash | 1 | 15 | 0.136 | 2.04 | 78.7 | 85.3 | 42.1 |
| RTX PRO 6000 S | bf16_dflash | 2 | 23 | 0.171 | 0.401 | 66 | 134 | 41.7 |
| RTX PRO 6000 S | bf16_dflash | 4 | 48 | 0.198 | 3.21 | 92.2 | 280 | 43 |
| RTX PRO 6000 S | bf16_dflash | 8 | 83 | 0.23 | 3.18 | 61 | 402 | 44.7 |
| RTX PRO 6000 S | bf16_dflash | 16 | 117 | 0.318 | 1.74 | 43.2 | 608 | 44.7 |
| RTX PRO 6000 S | bf16_dflash | 32 | 112 | 7.51 | 10.3 | 36.9 | 505 | 55.5 |
| RTX PRO 6000 S | fp8_dflash | 1 | 22 | 0.112 | 0.251 | 174 | 137 | 40.1 |
| RTX PRO 6000 S | fp8_dflash | 2 | 36 | 0.135 | 0.285 | 144 | 214 | 40.8 |
| RTX PRO 6000 S | fp8_dflash | 4 | 73 | 0.146 | 0.349 | 131 | 404 | 42.3 |
| RTX PRO 6000 S | fp8_dflash | 8 | 116 | 0.16 | 2.91 | 78.4 | 583 | 43.1 |
| RTX PRO 6000 S | fp8_dflash | 16 | 160 | 0.228 | 5.06 | 60.2 | 821 | 43.3 |
| RTX PRO 6000 S | fp8_dflash | 32 | 200 | 0.428 | 2.2 | 38.4 | 1.04e+03 | 43.9 |
| RTX PRO 6000 S | nvfp4_dflash | 1 | 20 | 0.11 | 1.11 | 143 | 129 | 40.3 |
| RTX PRO 6000 S | nvfp4_dflash | 2 | 30 | 0.136 | 0.252 | 120 | 195 | 41 |
| RTX PRO 6000 S | nvfp4_dflash | 4 | 61 | 0.141 | 0.293 | 121 | 390 | 41.2 |
| RTX PRO 6000 S | nvfp4_dflash | 8 | 109 | 0.15 | 2.89 | 83.9 | 605 | 42.3 |
| RTX PRO 6000 S | nvfp4_dflash | 16 | 152 | 0.209 | 5.01 | 70.4 | 890 | 43.9 |
| RTX PRO 6000 S | nvfp4_dflash | 32 | 223 | 0.322 | 1.23 | 45.5 | 1.26e+03 | 45.7 |
| H100 NVL | bf16_dflash | 8 | 122 | 0.142 | 3.06 | 87.2 | 625 | 41.1 |
| H100 NVL | bf16_dflash | 32 | 139 | 5.63 | 8.68 | 53.7 | 718 | 47.2 |
| H100 NVL | bf16_dflash | 64 | 183 | 14.9 | 17.1 | 58 | 748 | 55.4 |
| H100 NVL | fp8_dflash | 8 | 155 | 0.142 | 0.906 | 118 | 823 | 41.5 |
| H100 NVL | fp8_dflash | 32 | 259 | 0.335 | 4.4 | 58.8 | 1.4e+03 | 42.9 |
| H100 NVL | fp8_dflash | 64 | 272 | 5.55 | 6.94 | 47.3 | 1.31e+03 | 47.5 |
| RTX PRO 6000 WS | nvfp4_dflash | 16 | 171 | 0.143 | 3.65 | 71 | 971 | 45.5 |
| RTX PRO 6000 WS | nvfp4_dflash | 32 | 246 | 0.255 | 0.976 | 52.1 | 1.47e+03 | 44.3 |
| RTX PRO 6000 WS | nvfp4_dflash | 64 | 269 | 5.05 | 6.56 | 42.8 | 1.39e+03 | 48.6 |
| RTX PRO 6000 WS | nvfp4_dflash | 100 | 295 | 12.1 | 14 | 41.9 | 1.35e+03 | 56.7 |
| RTX PRO 6000 WS | nvfp4_dflash | 128 | 327 | 16.9 | 18.6 | 38 | 1.34e+03 | 60.9 |
| RTX 3090 | plain | 1 | 7 | 0.146 | 8.32 | 46.7 | 36.5 | 42.7 |
| RTX 3090 | plain | 4 | 25 | 0.165 | 0.665 | 39.4 | 144 | 44.5 |
| RTX 3090 | plain | 8 | 26 | 0.671 | 16.8 | 19 | 88.6 | 64.8 |
| RTX 3090 | plain | 16 | 25 | 21.3 | 42.2 | 10.4 | 66.6 | 87.7 |
| L40S | plain | 1 | 5 | 0.083 | 0.122 | 43.2 | 42.3 | 21.8 |
| L40S | plain | 4 | 17 | 0.195 | 0.269 | 38.2 | 126 | 28.7 |
| L40S | plain | 8 | 33 | 0.144 | 0.288 | 35.4 | 216 | 31.7 |
| L40S | fp8_plain | 1 | 4 | 0.126 | 0.148 | 21.7 | 21.4 | 35.2 |
| L40S | fp8_plain | 4 | 12 | 0.183 | 0.357 | 20.9 | 59.3 | 40.4 |
| L40S | fp8_plain | 8 | 22 | 0.309 | 0.608 | 20 | 117 | 34.3 |
| L40S | qwen9b_bf16 | 1 | 6 | 0.075 | 0.081 | 44.7 | 43.9 | 24 |
| L40S | qwen9b_bf16 | 8 | 37 | 0.092 | 0.219 | 38.9 | 232 | 32.4 |
| L40S | qwen9b_bf16 | 16 | 64 | 0.115 | 0.26 | 36.6 | 448 | 31.2 |
| L40S | qwen9b_bf16 | 32 | 72 | 7.85 | 12.3 | 36.8 | 462 | 41.2 |
| A100 PCIE | plain | 1 | 7 | 0.117 | 0.122 | 63.2 | 61.4 | 26.9 |
| A100 PCIE | plain | 4 | 24 | 0.129 | 0.207 | 56 | 199 | 28.3 |
| A100 PCIE | plain | 8 | 44 | 0.132 | 0.261 | 50.1 | 344 | 26.5 |
| A100 PCIE | qwen9b_bf16 | 1 | 7 | 0.055 | 0.062 | 74.2 | 72.9 | 21.5 |
| A100 PCIE | qwen9b_bf16 | 8 | 62 | 0.065 | 0.159 | 67.7 | 448 | 26.6 |
| A100 PCIE | qwen9b_bf16 | 16 | 101 | 0.068 | 0.194 | 62.2 | 847 | 27 |
| A100 PCIE | qwen9b_bf16 | 32 | 112 | 4.32 | 7.22 | 62.4 | 856 | 32 |
| RTX 3090 | plain | 1 | 5 | 0.126 | 0.128 | 46 | 44.9 | 22.2 |
| RTX 3090 | plain | 4 | 18 | 0.157 | 0.318 | 39.9 | 134 | 29.2 |
| RTX 3090 | plain | 8 | 34 | 0.16 | 0.445 | 37.8 | 237 | 30.3 |
| RTX 3090 | plain | 16 | 44 | 4.91 | 10.6 | 37.6 | 257 | 39.6 |
| RTX 3090 | plain | 1 | 6 | 0.126 | 0.135 | 47.2 | 45.7 | 22.2 |
| RTX 3090 | plain | 4 | 18 | 0.149 | 0.315 | 41 | 137 | 28 |
| RTX 3090 | plain | 8 | 33 | 0.153 | 0.459 | 38.4 | 224 | 31.6 |
| RTX 3090 | plain | 16 | 42 | 5.99 | 9.82 | 38.3 | 247 | 37.4 |
| RTX 3090 | qwen9b_bf16 | 1 | 6 | 0.08 | 0.088 | 48.2 | 47.1 | 22.4 |
| RTX 3090 | qwen9b_bf16 | 8 | 44 | 0.094 | 0.221 | 44.9 | 293 | 29.5 |
| RTX 3090 | qwen9b_bf16 | 16 | 68 | 0.123 | 0.398 | 41.4 | 493 | 30.4 |
| RTX 3090 | qwen9b_bf16 | 32 | 79 | 6.81 | 11 | 40.9 | 544 | 37.1 |
| RTX 4090 | plain | 1 | 5 | 0.15 | 0.155 | 53.7 | 52.1 | 21 |
| RTX 4090 | plain | 4 | 19 | 0.186 | 0.328 | 46.2 | 161 | 25.7 |
| RTX 4090 | plain | 8 | 39 | 0.185 | 0.292 | 43.1 | 283 | 30.2 |
| RTX 4090 | plain | 16 | 44 | 5.1 | 8.23 | 42.6 | 296 | 35 |
| RTX 5090 | plain | 1 | 8 | 0.048 | 0.055 | 83.6 | 82.1 | 21.5 |
| RTX 5090 | plain | 4 | 26 | 0.068 | 0.112 | 72.2 | 274 | 22.1 |
| RTX 5090 | plain | 8 | 62 | 0.067 | 0.156 | 73.1 | 495 | 25.6 |
| RTX 5090 | plain | 16 | 97 | 0.072 | 0.283 | 63.7 | 844 | 26.7 |
| L40S | fp8_dflash | 1 | 7 | 0.14 | 2.23 | 110 | 80 | 20.5 |
| L40S | fp8_dflash | 4 | 18 | 0.266 | 2.5 | 43.6 | 176 | 24.4 |
| L40S | fp8_dflash | 8 | 24 | 4.78 | 7.05 | 50.7 | 186 | 26.6 |
| L40S | fp8_dflash | 16 | 32 | 12.2 | 14.2 | 55.3 | 191 | 34.2 |
| L40S | fp8_dflash | 32 | 43 | 31 | 40.6 | 54.2 | 210 | 56.6 |
| L40S | nvfp4_dflash | 1 | 7 | 0.106 | 3.11 | 104 | 80.4 | 23.6 |
| L40S | nvfp4_dflash | 4 | 18 | 0.192 | 5.6 | 51.7 | 177 | 25.6 |
| L40S | nvfp4_dflash | 8 | 26 | 2.56 | 7.77 | 55.4 | 195 | 30.3 |
| L40S | nvfp4_dflash | 16 | 32 | 12.5 | 13.9 | 53.7 | 213 | 33.6 |
| L40S | nvfp4_dflash | 32 | 43 | 25.5 | 34.4 | 55.8 | 246 | 52 |
| L40S | qwen9b_bf16 | 1 | 6 | 0.062 | 0.072 | 50.5 | 49.6 | 21.3 |
| L40S | qwen9b_bf16 | 8 | 45 | 0.075 | 0.122 | 45 | 301 | 28.7 |
| L40S | qwen9b_bf16 | 16 | 70 | 0.1 | 0.197 | 41.7 | 500 | 31.2 |
| L40S | qwen9b_bf16 | 32 | 80 | 6.82 | 10.5 | 41.8 | 559 | 36.6 |
| RTX PRO 6000 S | plain | 1 | 7 | 0.093 | 0.098 | 76.7 | 74.6 | 21.3 |
| RTX PRO 6000 S | plain | 4 | 25 | 0.105 | 0.183 | 64.3 | 234 | 25.4 |
| RTX PRO 6000 S | plain | 8 | 53 | 0.119 | 0.2 | 60.8 | 399 | 27.5 |
| RTX PRO 6000 S | plain | 16 | 83 | 0.127 | 0.271 | 51.9 | 693 | 27.5 |
| RTX PRO 6000 S | fp8_plain | 1 | 4 | 0.099 | 1.06 | 45.4 | 42.4 | 20.5 |
| RTX PRO 6000 S | fp8_plain | 4 | 18 | 0.107 | 0.283 | 42.3 | 140 | 26.7 |
| RTX PRO 6000 S | fp8_plain | 8 | 32 | 0.109 | 0.345 | 39.8 | 278 | 25.6 |
| RTX PRO 6000 S | fp8_plain | 16 | 59 | 0.151 | 0.666 | 35.6 | 419 | 30.3 |
| RTX PRO 6000 S | fp8_plain | 32 | 91 | 0.199 | 1.24 | 30.2 | 694 | 32.1 |
| RTX PRO 6000 S | qwen9b_bf16 | 1 | 8 | 0.065 | 0.069 | 84.6 | 82.5 | 21.2 |
| RTX PRO 6000 S | qwen9b_bf16 | 8 | 68 | 0.08 | 0.157 | 76 | 537 | 26 |
| RTX PRO 6000 S | qwen9b_bf16 | 16 | 108 | 0.099 | 0.18 | 69.1 | 937 | 26.3 |
| RTX PRO 6000 S | qwen9b_bf16 | 32 | 123 | 3.84 | 6.22 | 70.7 | 967 | 30.5 |
| GB10 | plain | 1 | 4 | 0.178 | 0.219 | 13.6 | 13.4 | 56.9 |
| GB10 | plain | 4 | 8 | 0.386 | 0.408 | 13.1 | 39.6 | 39 |
| GB10 | plain | 8 | 18 | 0.403 | 0.587 | 12.3 | 60.3 | 51.7 |
| GB10 | plain | 16 | 30 | 1.04 | 1.07 | 10.7 | 116 | 53.2 |
| GB10 | bf16_dflash | 1 | 3 | 0.624 | 8.31 | 13.1 | 10.8 | 36.5 |
| GB10 | bf16_dflash | 4 | 6 | 4.1 | 7.03 | 12.6 | 40.3 | 38.1 |
| GB10 | bf16_dflash | 8 | 10 | 3.06 | 3.69 | 9.9 | 65.4 | 36.7 |
| GB10 | bf16_dflash | 16 | 20 | 3.81 | 8.69 | 8.2 | 92.2 | 48.1 |
| GB10 | bf16_dflash | 32 | 32 | 9.37 | 18.8 | 5.7 | 125 | 64.9 |
| RTX 4090 | b9speed | 1 | 7 | 0.086 | 0.092 | 56.9 | 55.8 | 28.1 |
| RTX 4090 | b9speed | 4 | 24 | 0.092 | 0.185 | 51.5 | 184 | 31.5 |
| RTX 4090 | b9speed | 16 | 95 | 0.129 | 0.289 | 45.5 | 605 | 34.4 |
| RTX 4090 | b9speed | 32 | 114 | 3.72 | 7.6 | 42.9 | 710 | 39.2 |
| RTX 5090 | b9speed | 1 | 10 | 0.079 | 0.093 | 95.5 | 92.7 | 29.8 |
| RTX 5090 | b9speed | 4 | 33 | 0.089 | 0.135 | 81.7 | 290 | 30.7 |
| RTX 5090 | b9speed | 16 | 133 | 0.112 | 0.237 | 71 | 981 | 31.1 |
| RTX 5090 | b9speed | 32 | 237 | 0.139 | 0.389 | 65.1 | 1.8e+03 | 30.9 |
| RTX 4090 | engine_sglang | 1 | 7 | 0.031 | 0.04 | 57.5 | 57.1 | 29.4 |
| RTX 4090 | engine_sglang | 8 | 29 | 7.71 | 9.84 | 52.9 | 139 | 41.1 |
| RTX 4090 | engine_sglang | 16 | 35 | 16.5 | 22 | 52.8 | 147 | 46.8 |
| RTX 4090 | engine_vllm | 1 | 7 | 0.045 | 0.053 | 56.9 | 56.2 | 27.9 |
| RTX 4090 | engine_vllm | 8 | 58 | 0.069 | 0.084 | 51.2 | 350 | 31.9 |
| RTX 4090 | engine_vllm | 16 | 100 | 0.067 | 0.158 | 48.2 | 657 | 34.6 |
| GB10 | 27b_fp8_dflash | 1 | 6 | 0.381 | 4.97 | 26.7 | 23.7 | 54.4 |
| GB10 | 27b_fp8_dflash | 2 | 8 | 0.566 | 3.25 | 18.1 | 36.8 | 49.3 |
| GB10 | 27b_fp8_dflash | 4 | 15 | 0.813 | 3.13 | 15.6 | 67.1 | 48.2 |
| GB10 | 27b_fp8_dflash | 8 | 29 | 1.28 | 1.95 | 14.3 | 112 | 56.7 |
| GB10 | 27b_fp8_dflash | 16 | 44 | 1.11 | 6.09 | 12.8 | 157 | 64.9 |
| GB10 | 27b_fp8_dflash | 32 | 55 | 2.29 | 11.1 | 10.6 | 215 | 67.7 |
| GB10 | 27b_fp8_plain | 1 | 2 | 0.328 | 0.328 | 8 | 7.9 | 49 |
| GB10 | 27b_fp8_plain | 2 | 2 | 0.423 | 0.423 | 8.1 | 14.6 | 56.9 |
| GB10 | 27b_fp8_plain | 4 | 7 | 1.26 | 5.46 | 7.7 | 24.5 | 70.3 |
| GB10 | 27b_fp8_plain | 8 | 18 | 0.586 | 1.36 | 7.3 | 37.8 | 90.5 |
| GB10 | 27b_fp8_plain | 16 | 31 | 1.25 | 2.7 | 6.7 | 73.7 | 84.1 |
| GB10 | 27b_fp8_plain | 32 | 51 | 2.27 | 5.6 | 5.4 | 122 | 110 |
| GB10 | 27b_nvfp4_dflash | 1 | 4 | 0.325 | 6.49 | 25.5 | 21.3 | 46.2 |
| GB10 | 27b_nvfp4_dflash | 2 | 7 | 0.533 | 3.18 | 15.5 | 34.1 | 52.7 |
| GB10 | 27b_nvfp4_dflash | 4 | 15 | 0.656 | 3 | 14 | 63.3 | 55.3 |
| GB10 | 27b_nvfp4_dflash | 8 | 29 | 0.765 | 1.26 | 14.6 | 106 | 63.6 |
| GB10 | 27b_nvfp4_dflash | 16 | 48 | 0.974 | 3.91 | 12.6 | 168 | 69.7 |
| GB10 | 27b_nvfp4_dflash | 32 | 55 | 1.55 | 6.07 | 10.2 | 221 | 73.4 |
| GB10 | qwen9b_bf16 | 1 | 4 | 0.179 | 2.7 | 12.9 | 12.3 | 70.1 |
| GB10 | qwen9b_bf16 | 2 | 4 | 0.224 | 0.226 | 13.8 | 26.7 | 45.8 |
| GB10 | qwen9b_bf16 | 4 | 13 | 0.237 | 0.317 | 13.5 | 42.5 | 76.3 |
| GB10 | qwen9b_bf16 | 8 | 28 | 0.254 | 0.324 | 13.1 | 87.6 | 68.1 |
| GB10 | qwen9b_bf16 | 16 | 40 | 0.266 | 0.597 | 12.3 | 132 | 64.4 |

## What this cost

Whole campaign: US$ 50.79 (about R$ 249), shared by all repositories of this series. Rental share of the GPUs used in this repo:

| GPU | rentals | hours | rental cost (USD) |
|---|---|---|---|
| GB10 | 3 | 0.0 | 0.0 |
| L40S | 10 | 4.91 | 3.92 |
| RTX 3090 | 30 | 11.8 | 2.97 |
| RTX 4090 | 29 | 6.22 | 4.05 |
| RTX 5090 | 39 | 12.44 | 8.54 |

Rental figures are hourly price x time of each rented instance; marketplace download fees and idle time are in the campaign total, not in this table.

## Reproduce

`bash scripts/run.sh` documents the exact commands used on the rented instance (Ubuntu, CUDA, Python venv). Evidence files are the unmodified measured rows (host location and reliability removed).
