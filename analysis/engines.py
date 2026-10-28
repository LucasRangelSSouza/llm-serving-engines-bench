"""vLLM against SGLang on one RTX 5090: throughput, first-token time, achieved concurrency reconstructed from request timestamps, bootstrap intervals."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from benchkit import bootstrap_ci
from common import BLUE, ORANGE, read, save_fig, save_table, update_summary

cmp_ = read("engine_comparison.csv")
req = read("engine_requests.csv")
e5 = cmp_[cmp_.setup.str.contains("5090")].copy()
e5["engine"] = np.where(e5.setup.str.contains("SGLang"), "SGLang", "vLLM")
save_table(e5[["engine", "users", "agg_out_tps", "ttft_p50_s", "ttft_p95_s", "decode_tps_per_user_p50"]].round(3), "engine_comparison", index=False)

r5 = req[req.setup.str.contains("5090")].copy()
r5["engine"] = np.where(r5.setup.str.contains("SGLang"), "SGLang", "vLLM")
ok = r5[r5.status == 200]


def running(g, step=0.25):
    """Requests that are decoding at each instant: after their first token and before their last."""
    g = g[g.status == 200]
    t0, t1 = g.t_start_s.min(), g.t_end_s.max()
    grid = np.arange(t0, t1, step)
    dec_start = g.t_start_s + g.ttft_s
    return grid - t0, np.array([((dec_start <= x) & (g.t_end_s > x)).sum() for x in grid])


conc = {}
fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
for eng, col in (("vLLM", BLUE), ("SGLang", ORANGE)):
    d = e5[e5.engine == eng].sort_values("users")
    axes[0].plot(d.users, d.agg_out_tps, marker="o", color=col, label=eng)
    axes[1].plot(d.users, d.ttft_p50_s, marker="o", color=col, label=eng)
    g = r5[(r5.engine == eng) & (r5.level == 32)]
    x, y = running(g)
    axes[2].plot(x, y, color=col, label=eng)
    conc[eng] = {"max": int(y.max()), "mean": float(y[(x > 3) & (x < x.max() - 3)].mean())}
axes[0].set_ylabel("Aggregate output (tokens/s)"); axes[1].set_ylabel("Median first token (s)"); axes[1].set_yscale("log")
axes[2].set_ylabel("Requests decoding at once (32 users)"); axes[2].set_xlabel("Time into the run (s)")
for ax in axes[:2]:
    ax.set_xscale("log", base=2); ax.set_xlabel("Concurrent users")
axes[0].legend(frameon=False)
save_fig(fig, "engines_vllm_vs_sglang.png")

rows = []
for (eng, lv), g in ok.groupby(["engine", "level"]):
    dur = r5[(r5.engine == eng) & (r5.level == lv)]
    window = dur.t_end_s.max() - dur.t_start_s.min()
    per_req = g.completion_tokens.values
    lo, pt, hi = bootstrap_ci(per_req, stat=lambda v: v.sum() / len(v) * len(g) / window, B=1000)
    rows.append({"engine": eng, "users": lv, "tps_low": lo, "tps": pt, "tps_high": hi})
ci = pd.DataFrame(rows)
save_table(ci.round(1), "engine_throughput_bootstrap", index=False)

a32 = e5[e5.users == 32].set_index("engine")
update_summary({"vllm_agg_32": a32.loc["vLLM", "agg_out_tps"], "sglang_agg_32": a32.loc["SGLang", "agg_out_tps"], "vllm_ttft_p50_32": a32.loc["vLLM", "ttft_p50_s"], "sglang_ttft_p50_32": a32.loc["SGLang", "ttft_p50_s"],
                "vllm_max_concurrent_32": conc["vLLM"]["max"], "sglang_max_concurrent_32": conc["SGLang"]["max"]})
print(ci.round(1))
