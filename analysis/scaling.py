"""How a 27B model (4-bit AWQ) and a 9B model scale with concurrent users on each GPU, and what a token costs at each load."""
import matplotlib.pyplot as plt
import pandas as pd
from benchkit import usd_per_million_tokens
from common import BLUE, gpu_name, read, save_fig, save_table, update_summary

llm = read("llm_serving.csv")
llm = llm[llm.valid_run & ~llm.thinking.astype(bool)].copy()
llm["gpu"] = llm.gpu.map(gpu_name)
awq = llm[llm.variant == "plain"]

t = awq.groupby(["gpu", "users"]).agg(per_user=("decode_tps_per_user_p50", "median"), agg_tps=("agg_out_tps", "median"), ttft95=("ttft_p95_short_s", "median"), dph=("dph_usd", "median")).reset_index()
t["usd_per_m_tokens"] = [usd_per_million_tokens(r.dph, r.agg_tps) for r in t.itertuples()]
t["meets_sla"] = (t.per_user >= 20) & (t.ttft95 <= 3)
save_table(t.round(3), "awq27b_scaling", index=False)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for g, d in t.groupby("gpu"):
    axes[0].plot(d.users, d.per_user, marker="o", label=g)
    axes[1].plot(d.users, d.agg_tps, marker="o", label=g)
    axes[2].plot(d.users, d.ttft95, marker="o", label=g)
for ax, yl in zip(axes, ("Decode speed per user (tokens/s)", "Aggregate output (tokens/s)", "95th percentile first token (s)")):
    ax.set_xscale("log", base=2); ax.set_xlabel("Concurrent users"); ax.set_ylabel(yl)
axes[2].set_yscale("log"); axes[2].axhline(3, ls="--", color="grey"); axes[0].legend(frameon=False, fontsize=7)
fig.suptitle("27B AWQ build under load, one line per GPU")
save_fig(fig, "awq27b_scaling.png")

best = t[t.meets_sla].sort_values("usd_per_m_tokens").groupby("gpu").head(1).set_index("gpu")
save_table(best[["users", "per_user", "agg_tps", "ttft95", "usd_per_m_tokens"]].round(3), "awq27b_cheapest_within_sla")
fig, ax = plt.subplots(figsize=(7, 3.6))
b = best.sort_values("usd_per_m_tokens")
ax.barh(b.index[::-1], b.usd_per_m_tokens[::-1], color=BLUE)
for i, v in enumerate(b.usd_per_m_tokens[::-1]):
    ax.text(v, i, f" {v:.2f}", va="center", fontsize=8)
ax.set_xlabel("US$ per million output tokens at the cheapest load that meets 20 tok/s/user and 3 s first token")
save_fig(fig, "awq27b_cost_within_sla.png")

peak = awq.groupby("gpu").agg_out_tps.max()
update_summary({
    "best_gpu_within_sla": b.index[0], "best_usd_per_m_within_sla": b.usd_per_m_tokens.iloc[0],
    "worst_gpu_within_sla": b.index[-1], "worst_usd_per_m_within_sla": b.usd_per_m_tokens.iloc[-1],
    "peak_agg_awq_5090": peak["RTX 5090"], "peak_agg_awq_3090": peak["RTX 3090"], "peak_agg_awq_4090": peak["RTX 4090"], "peak_agg_awq_pro6000": peak["RTX PRO 6000"],
})
print(b.round(3))
